# TO solve the problem: Error in magma_getdevice_arch: MAGMA not initialized (call magma_init() first) or bad device
# import cupy

# print(cupy.cuda.runtime.get_build_version())  # Checks if MAGMA is in the build


# cupy.cuda.runtime.magma_init()


import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model_bilevel import BilevelDiffusion
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
# from utils_single import ExponentialMovingAverage as ExponentialMovingAverage_single
import os
import math
import argparse
from copy import deepcopy
from evaluation import FIDCalculator
import csv
import time

import sys
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import json

# torch.cuda.empty_cache()


def seed_everything(seed: int):
    """
    Seed all necessary random number generators for reproducibility.

    Args:
        seed (int): The seed value to use for all libraries.
    """
    torch.manual_seed(seed)  # PyTorch's random

    # For CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensures reproducibility with some operations in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Learnable scheduler
class LearnableScheduler(nn.Module):
# class LearnableScheduler:
    def __init__(self, timesteps, initial = None, device = 'cpu',initial_start=0, initial_end=1, initial_tau=1, initial_epsilon=0.008):
        super().__init__()
        self.timesteps = timesteps
        self.initial = initial
        self.device = device
        
        if initial == "cosine":
            self.start_tensor = nn.Parameter(torch.tensor(initial_start, dtype=torch.float32, requires_grad=True))
            self.end_tensor = nn.Parameter(torch.tensor(initial_end, dtype=torch.float32, requires_grad=True))
            self.tau_tensor = nn.Parameter(torch.tensor(initial_tau, dtype=torch.float32, requires_grad=True))
            self.s_tensor = nn.Parameter(torch.tensor(initial_epsilon, dtype=torch.float32, requires_grad=True))
        elif initial == "sigmoid":
            self.start_tensor = nn.Parameter(torch.tensor(initial_start, dtype=torch.float32, requires_grad=True))
            self.end_tensor = nn.Parameter(torch.tensor(initial_end, dtype=torch.float32, requires_grad=True))
            self.tau_tensor = nn.Parameter(torch.tensor(initial_tau, dtype=torch.float32, requires_grad=True))
            self.s_tensor = nn.Parameter(torch.tensor(initial_epsilon, dtype=torch.float32, requires_grad=True))
        else:
            # Initialize betas using the cosine schedule, then make it learnable
            initial_betas = self._cosine_variance_schedule(timesteps)
            self._learnable_betas = nn.Parameter(initial_betas, requires_grad=True)  # Make betas learnable
        
    # modified from https://arxiv.org/pdf/2301.10972
    def _cosine_schedule(self, start, end, timesteps, tau, epsilon):
        # Define your cosine schedule logic here
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32,device = self.device) 
        steps = steps * (end - start) + start
        
        cos_value = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
        cos_value = torch.clamp(cos_value, min=0, max=1)  # Ensure non-negative for exponentiation
        f_t = cos_value ** (2 * tau)

        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        
        return betas
    

    def _sigmoid_schedule(self, start=-3, end=3, timesteps=1000, tau=1.0, s=0):
        # A gamma function based on sigmoid function.
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32,device = self.device) 
        steps = steps * (end - start) + start
        f_t = torch.sigmoid(((timesteps-steps) / timesteps)/tau+s)

        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    

    @property
    def betas(self):
        if self.initial == "cosine":
            return self._cosine_schedule(self.start_tensor, self.end_tensor, self.timesteps, self.tau_tensor, self.s_tensor)
        elif self.initial == "sigmoid":
            return self._sigmoid_schedule(self.start_tensor, self.end_tensor, self.timesteps, self.tau_tensor, self.s_tensor)
        else:
            return self._learnable_betas

    def get_betas(self):
        # return self.betas
        return self.betas
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--lr_z',type = float ,default=0.001)
    parser.add_argument('--lr_beta',type = float, nargs=4,default=[0.01, 0.01, 0.1, 0.001])
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=5)
    parser.add_argument('--inner_loop',type = int,help = 'steps of U-Net training',default=1)
    parser.add_argument('--inner_loop_z',type = int,help = 'steps z update',default=1)
    parser.add_argument('--initial_epoch',type = int,help = 'initial steps of U-Net training',default=50)
    parser.add_argument('--gamma', type=float, default=1, help="penalty constant starting point")
    parser.add_argument('--gamma_end', type=float, default=1, help="penalty constant end point")
    parser.add_argument('--delta', type=float, default=0.01, help="perturbation")
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'steps of forward pass',default=1000)
    parser.add_argument('--backward_steps',type = int,help = 'sampling steps of DDIM',default=100)
    parser.add_argument('--truncation',type = int,help = 'steps of truncated backpropagation',default=1)
    parser.add_argument('--initial_start',type = float,help = 'initial value of start',default=0)
    parser.add_argument('--initial_end',type = float,help = 'initial value of end',default=1)
    parser.add_argument('--initial_tau',type = float,help = 'initial value of tau',default=1)
    parser.add_argument('--initial_epsilon',type = float,help = 'initial value of epsilon',default=0.008)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--ZO_type',type=str, help = 'one point residual or two point estimator',choices=['one', 'two'],default='two')
    parser.add_argument('--tau_type',type = str,help = 'tau_type for DDIM skiiping',default='linear')
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    # parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use.")

    parser.add_argument(
        '--scheduler',
        choices=['cosine', 'linear', 'sigmoid'],
        default='cosine',
        help='Choose a scheduler type: "cosine", "linear", or "sigmoid"'
    )

    args = parser.parse_args()

    return args


def save_args(args, filepath):
    """
    Save the parsed arguments to a JSON file.

    Args:
        args (argparse.Namespace): Parsed arguments.
        filepath (str): Path to the file where arguments will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)



def zeroth_order_gradient(params, model_ema, delta, fid, ZO_type, loss_pre=None):
    # Determine the parameters to optimize
    if params == 'scheduler':
        param_group = list(model_ema.shadow_model.scheduler.parameters())
    elif params == 'model':
        param_group = list(model_ema.shadow_model.model.parameters())
    else:
        raise ValueError("Invalid params argument. Must be 'scheduler' or 'model'.")

    # Generate perturbations for all parameters
    perturbation = [torch.randn_like(param) for param in param_group]

    # Placeholder for gradients
    grads = [torch.zeros_like(param) for param in param_group]

    # ===== Perturbation 1 =====
    with torch.no_grad():
        for param, noise in zip(param_group, perturbation):
            param.add_(noise * delta)  # Apply forward perturbation in-place

        samples = model_ema.shadow_model.sampling_DDIM(args.n_samples, device=args.device, tau_type=args.tau_type)
        samples = samples.repeat(1, 3, 1, 1)  # Adjust for MNIST or other datasets
        samples = (samples + 1) / 2  # Scale to [0, 1]
        
        fid.reset()
        fid.update(samples, real=False)
        loss_fwd = fid.compute()

    # ===== Perturbation 2 =====
    if ZO_type == 'two':
        with torch.no_grad():
            for param, noise in zip(param_group, perturbation):
                param.add_(-2 * noise * delta)  # Apply backward perturbation in-place
        
            samples = model_ema.shadow_model.sampling_DDIM(args.n_samples, device=args.device, tau_type=args.tau_type)
            samples = samples.repeat(1, 3, 1, 1)  # Adjust for MNIST or other datasets
            samples = (samples + 1) / 2  # Scale to [0, 1]
            
            fid.reset()
            fid.update(samples, real=False)
            loss_bwd = fid.compute()

        # # Revert parameters to original values
        # with torch.no_grad():
            for param, noise in zip(param_group, perturbation):
                param.add_(noise * delta)  # Revert in-place
    elif ZO_type == 'one':
        loss_bwd = loss_pre
        # Revert forward perturbation
        with torch.no_grad():
            for param, noise in zip(param_group, perturbation):
                param.add_(-noise * delta)
    else:
        raise ValueError("ZO_type must be 'two' or 'one'.")

    # ===== Compute Gradients for All Parameters =====
    scale = 1 / (2 * delta)
    for param, grad, noise in zip(param_group, grads, perturbation):
        grad.copy_(((loss_fwd - loss_bwd) * scale) * noise)  # Zeroth-order gradient scaled
        param.grad = grad  # Assign the computed gradient

    if ZO_type == 'two':
        return param_group
    elif ZO_type == 'one':
        return param_group, loss_fwd


class GammaScheduler:
    def __init__(self, start_value, end_value, global_steps, mode="linear"):
        """
        A scheduler for penalty constant gamma that increases linearly or quadratically.
        
        Args:
            start_value (float): Starting value of gamma.
            end_value (float): Ending value of gamma.
            global_steps (int): Total number of steps over which to increase gamma.
            mode (str): "linear" or "quadratic". Determines the type of increase.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.global_steps = global_steps
        self.mode = mode

    def get_gamma(self, step):
        """
        Get the gamma value at a specific step.
        
        Args:
            step (int): Current global step.
        
        Returns:
            float: Gamma value at the given step.
        """
        # Ensure step is within bounds
        step = max(0, min(step, self.global_steps))

        if self.mode == "linear":
            # Linear interpolation
            return self.start_value + (self.end_value - self.start_value) * (step / self.global_steps)
        elif self.mode == "quadratic":
            # Quadratic interpolation
            fraction = step / self.global_steps
            return self.start_value + (self.end_value - self.start_value) * (fraction ** 2)
        else:
            raise ValueError("Mode must be 'linear' or 'quadratic'.")





def main(args):
    # device = "cpu" if args.cpu else "cuda"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if not args.cpu else "cpu")
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)

    initial = args.scheduler  
    print("Using ", initial, " Scheduler")

    # Initialize model and learnable scheduler
    scheduler = LearnableScheduler(args.timesteps, initial, device,args.initial_start,args.initial_end,args.initial_tau,args.initial_epsilon)

        
    # using the torchmetrics FID
    fid = FrechetInceptionDistance(normalize=True,reset_real_features=False).to(device)
    
    inception = InceptionScore(normalize=True).to(device)
    
    
    ZO_type = args.ZO_type

   

    # initializing FID with real features
    for step, (image, _)in enumerate(test_dataloader):

        image = image.to(device)


        # # differntiable FID needs input scale at [-1,1]
        # FID.update_real(image)

        # do this as it is MNIST, no need for CIFAR-10
        image = image.repeat(1, 3, 1, 1)  # Repeat the channels
        image = (image + 1) / 2  # Scale values from [-1, 1] to [0, 1]
        fid.update(image, real=True)


    model = BilevelDiffusion(
        scheduler=scheduler,
        timesteps=args.timesteps,
        back_steps=args.backward_steps,
        truncation = args.truncation, 
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)


    adjust = 0.5 * args.batch_size * args.model_ema_steps / (args.epochs*args.inner_loop)
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    print("adjust:", adjust)
    print("alpha:", alpha)
    model_ema = ExponentialMovingAverage(model, decay=1.0 - alpha, device=device)

    
    back_steps=args.backward_steps
    inner_loop = args.inner_loop
    inner_loop_z = args.inner_loop_z
    

    # Unet for theta_y
    theta_y = deepcopy(model.model)
    # model_forward = deepcopy(model)
    

    
    # Optimizers
    optimizer_theta_z = AdamW(model.model.parameters(), lr=args.lr_z)
    
    
    param_groups = [
    {"params": [scheduler.start_tensor], "lr": args.lr_beta[0]},
    {"params": [scheduler.end_tensor], "lr": args.lr_beta[1]},
    {"params": [scheduler.tau_tensor], "lr": args.lr_beta[2]},
    {"params": [scheduler.s_tensor], "lr": args.lr_beta[3]}
    ]
    
    if initial  =="cosine":
        optimizer_scheduler = AdamW(param_groups)
    elif initial  =="sigmoid":
        optimizer_scheduler = AdamW(param_groups)
    else:
        optimizer_scheduler = AdamW([scheduler._learnable_betas], lr=args.lr_beta[0])
    optimizer_theta_y = AdamW(theta_y.parameters(), lr=args.lr)



    scheduler_theta_z=OneCycleLR(optimizer_theta_z,args.lr_z,total_steps=(args.epochs-args.initial_epoch)*len(train_dataloader)*inner_loop_z,pct_start=0.25,anneal_strategy='cos')
    scheduler_theta_y=OneCycleLR(optimizer_theta_y,args.lr,total_steps=args.epochs*len(train_dataloader)*inner_loop,pct_start=0.25,anneal_strategy='cos')
    lr_scheduler_beta=OneCycleLR(optimizer_scheduler,args.lr_beta,total_steps=(args.epochs-args.initial_epoch)*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')

    
    # Parameters for the scheduler
    start_gamma = args.gamma
    end_gamma = args.gamma_end
    total_steps = (args.epochs-args.initial_epoch)*len(train_dataloader)*args.inner_loop_z
    
    # Create a linear scheduler
    linear_scheduler = GammaScheduler(start_value=start_gamma, end_value=end_gamma, global_steps=total_steps, mode="linear")


    # Loss function
    loss_fn = nn.MSELoss(reduction='mean')

    global_steps = 0
    print("beta_value", torch.mean(scheduler.betas))
    
    scheduler_type = getattr(args, 'scheduler', 'default_value')
    
    output_file = "results/bilevel/gamma_{}_gamma_end_{}_y_loop_{}_z_loop_{}_initial_epoch_{}_epoch_{}_lr_beta_{}_lr_{}_lr_z_{}/{}/output/scores.csv".format(
        args.gamma,
        args.gamma_end,
        args.inner_loop,
        args.inner_loop_z,
        args.initial_epoch,
        args.epochs,
        args.lr_beta,
        args.lr,
        args.lr_z,
        scheduler_type
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    header = ["Epoch", "Step", "Upper Loss", "FID Score", "IS Score","start","end","tau","epsilon"]
    file_exists = os.path.isfile(output_file)
    
    # Save arguments to a file
    directory = "results/bilevel/gamma_{}_gamma_end_{}_y_loop_{}_z_loop_{}_initial_epoch_{}_epoch_{}_lr_beta_{}_lr_{}_lr_z_{}/{}/output/".format(
        args.gamma,
        args.gamma_end,
        args.inner_loop,
        args.inner_loop_z,
        args.initial_epoch,
        args.epochs,
        args.lr_beta,
        args.lr,
        args.lr_z,
        scheduler_type
    )
    
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(directory, "args.json")  # Path for the JSON file
    save_args(args, file_path)
    print(f"Arguments saved to {file_path}")
    
    
    flag=0
    loss_pre_model = 0
    loss_pre_scheduler = 0
    
    total_step_gamma = 0
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header only if the file is new
        writer.writerow(header)
        
        start_time = time.time()

        for epoch in range(args.epochs):
            for step, (image, _) in enumerate(train_dataloader):

                
                model.train()
                model_ema.shadow_model.eval()
                
                image = image.to(device)
                
                gamma_linear = linear_scheduler.get_gamma(total_step_gamma)
                
                
                for _ in range(inner_loop):  # Inner optimization steps
                    # clear gradients
                    optimizer_theta_y.zero_grad()
                    optimizer_theta_z.zero_grad()
                    optimizer_scheduler.zero_grad()
    
    
                    # ===== Phase 1: Solve inner optimization for theta_y =====
                    noise = torch.randn_like(image)
                    t = torch.randint(0, args.timesteps, (image.size(0),)).to(device)
                    x_t = model._forward_diffusion(image, t, noise)
    
                    pred_noise_y = theta_y(x_t, t)
                    lower_loss_y = loss_fn(pred_noise_y, noise)
                    grads = torch.autograd.grad(lower_loss_y, theta_y.parameters())
    
                    with torch.no_grad():
                            for param, grad in zip(theta_y.parameters(), grads):
                                param.grad = grad  # Example of manual gradient update
    
    
                    optimizer_theta_y.step()
                    scheduler_theta_y.step()
                                    
                    
                global_steps += 1
                if epoch>= args.initial_epoch:
                    total_step_gamma += 1
                    # clear gradients
                    optimizer_theta_z.zero_grad()
                    optimizer_theta_y.zero_grad()
                    optimizer_scheduler.zero_grad()
                    
                    
                    model.model = deepcopy(theta_y)
                    
                    for _ in range(inner_loop_z):  # Inner optimization steps for z
                        # clear gradients
                        
                        optimizer_theta_y.zero_grad()
                        optimizer_theta_z.zero_grad()
                        optimizer_scheduler.zero_grad()
                        
                        # ===== Phase 2: Compute upper loss =====
                        # samples = model.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type)
                        samples = model_ema.shadow_model.sampling_DDIM_no_grad(args.n_samples,device=device,tau_type = args.tau_type)

                        fid.reset()
                        inception.reset()
                        
                        
                        # do this as it is MNIST, no need for CIFAR-10
                        samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
                        samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
                        
                        fid.update(samples, real=False)                
                        fid_score = fid.compute()
                        upper_loss = fid_score
                        
                        inception.update(samples)
                        inception_score, std_inception_score = inception.compute()
        
                        
                        # ===== Phase 3: Update theta_z using upper_loss/gamma + lower_loss_z =====
                        noise = torch.randn_like(image)
                        t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
                        pred_noise_z = model(image, t, noise)
        
                        
                        lower_loss_z = loss_fn(pred_noise_z, noise)
                        
                        if ZO_type == 'two':                            
                            param_group = zeroth_order_gradient(params='model', model_ema=model_ema, delta=args.delta, fid=fid, ZO_type=ZO_type)
                        elif ZO_type == 'one':
                            param_group, loss_pre_model = zeroth_order_gradient(params='model', model_ema=model_ema, delta=args.delta, fid=fid, ZO_type=ZO_type, loss_pre = loss_pre_model)
                        else:
                            raise ValueError("ZO_type must be 'two' or 'one'.")
            
                        # Combine upper_loss and lower_loss_z
                        grad_lower = torch.autograd.grad(lower_loss_z, model.model.parameters()) 

                        
                        # Apply gradients manually
                        with torch.no_grad():
                            for param, grad_u, grad_l in zip(model.model.parameters(), param_group,grad_lower):
                                param.grad = grad_u.grad / gamma_linear + grad_l  # Example of manual gradient update

        
                        optimizer_theta_z.step()
                        scheduler_theta_z.step()
                    
                    # ===== Compute gradient of scheduler at y_star
                    noise = torch.randn_like(image)
                    t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
                    x_t = model._forward_diffusion(image, t, noise)
        
                    pred_noise_y_star = theta_y(x_t, t)
                    lower_loss_y_star = loss_fn(pred_noise_y_star, noise)
                    
                    grad_lower_y_star = torch.autograd.grad(lower_loss_y_star, scheduler.parameters())  
        
        
                    # ===== Phase 4: Compute scheduler gradients =====
                    # clear gradients
                    optimizer_theta_z.zero_grad()
                    optimizer_theta_y.zero_grad()
                    optimizer_scheduler.zero_grad()
        
                    # ===== Phase 5: Compute upper loss =====
                    # samples = model.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type, steps=back_steps)
                    samples = model_ema.shadow_model.sampling_DDIM_no_grad(args.n_samples,device=device,tau_type = args.tau_type, steps=back_steps)
        
        
                    # FID.clear_fake()
                    fid.reset()
                    inception.reset()

                    
                    # do this as it is MNIST, no need for CIFAR-10
                    samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
                    samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
                    fid.update(samples, real=False)
            
                    fid_score = fid.compute()
                    upper_loss = fid_score
                    
                    inception.update(samples)
                    inception_score, std_inception_score = inception.compute()
        
        
                    if ZO_type == 'two':
                        param_group = zeroth_order_gradient(params='scheduler', model_ema=model_ema, delta=args.delta, fid=fid, ZO_type = ZO_type)
                    elif ZO_type == 'one':
                        param_group, loss_pre_scheduler = zeroth_order_gradient(params='scheduler', model_ema=model_ema, delta=args.delta, fid=fid, ZO_type = ZO_type, loss_pre=loss_pre_scheduler)
                    else:
                        raise ValueError("ZO_type must be 'two' or 'one'.")
        
        
                    # ==== calculate gradient of z_star
                    # clear gradients
                    optimizer_theta_z.zero_grad()
                    optimizer_theta_y.zero_grad()
                    optimizer_scheduler.zero_grad()
                    
                    noise = torch.randn_like(image)
                    t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
        
                    pred_noise_z_star = model(image, t, noise)
        
                    
                    lower_loss_z_star = loss_fn(pred_noise_z_star, noise)
        
                    grad_lower_z_star = torch.autograd.grad(lower_loss_z_star, scheduler.parameters())
        
                    combined_grad = [
                        g_upper.grad/gamma_linear + (g_z - g_y)
                        for g_upper, g_z, g_y in zip(param_group, grad_lower_z_star, grad_lower_y_star)
                    ]
                    
        
        
                
                    for param, grad in zip(scheduler.parameters(), combined_grad):
                        param.grad = grad
        
                    optimizer_scheduler.step()
                    lr_scheduler_beta.step()
                
                    if initial  == "cosine":
                        scheduler.start_tensor.data.clamp_(0, 1)
                        scheduler.end_tensor.data.clamp_(scheduler.start_tensor + 1e-10, torch.tensor(1.0, dtype=torch.float32).to(device))
                        scheduler.tau_tensor.data.clamp_(1, 1e6)
                        scheduler.s_tensor.data.clamp_(0,1)
                    if initial == "sigmoid":
                        scheduler.end_tensor.data.clamp_(scheduler.start_tensor + 1e-10, torch.tensor(1e6, dtype=torch.float32).to(device))
                        scheduler.tau_tensor.data.clamp_(0.1, 1)
                    else:
                        # Clip beta to the range [0, 1]
                        scheduler.betas.data.clamp_(0., 0.999)  # In-place clamping
        

                    if step % args.log_freq == 0:
                        if initial  == "cosine" or "sigmoid":
                            print(
                                    f"start: {scheduler.start_tensor.data.item():.5f}, "
                                    f"end: {scheduler.end_tensor.data.item():.5f}, "
                                    f"tau: {scheduler.tau_tensor.data.item():.5f}, "
                                    f"epsilon: {scheduler.s_tensor.data.item():.5f}"
                                )

                        else:
                            print("max gradient, ", [max(g) for g in combined_grad])
                            print("beta_value", torch.mean(scheduler.betas))
                   
                    # Update EMA model
                    if global_steps % args.model_ema_steps == 0:
                        # print("update model 2")
                        model_ema.update_parameters(model)

                else:
                    if global_steps % args.model_ema_steps == 0:
                        # print("update model 1")
                        model.model = deepcopy(theta_y)
                        model_ema.update_parameters(model)
                        # model_single.model = deepcopy(theta_y)
                        # model_ema_single.update_parameters(model_single)

                if step % args.log_freq == 0:
                    
                    
                    
                    if epoch>=args.initial_epoch:
                        print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(train_dataloader)}], "
                                  f"Lower Loss Z: {lower_loss_z.item():.5f}, Upper Loss: {upper_loss.item():.5f}, torch FID: {fid_score.item():.5f}, torch IS: {inception_score.item():.5f}, Lower Loss y: {lower_loss_y.item():.5f}, Gamma: {gamma_linear:.5f}")
                        writer.writerow([epoch, step, upper_loss, fid_score, inception_score, scheduler.start_tensor.data.item(),scheduler.end_tensor.data.item(),scheduler.tau_tensor.data.item(),scheduler.s_tensor.data.item()])

                    else:
                        print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(train_dataloader)}], "
                                  f"Lower Loss y: {lower_loss_y.item():.5f}, Gamma: {gamma_linear:.5f}")
                        writer.writerow([epoch, step, "", "", "", scheduler.start_tensor.data.item(),scheduler.end_tensor.data.item(),scheduler.tau_tensor.data.item(),scheduler.s_tensor.data.item()])


    


            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict()
            }
            os.makedirs("results", exist_ok=True)
            # torch.save(ckpt, f"results/bilevel/gamma_{args.gamma:.3f}/{args.scheduler}/steps_{global_steps:08d}.pt")
            torch.save(
                ckpt,
                f"results/bilevel/gamma_{args.gamma}_gamma_end_{args.gamma_end}_y_loop_{args.inner_loop}_z_loop_{args.inner_loop_z}_initial_epoch_{args.initial_epoch}_epoch_{args.epochs}_lr_beta_{args.lr_beta}_lr_{args.lr}_lr_z_{args.lr_z}/{args.scheduler}/steps_{global_steps:08d}.pt"
            )

            
            model_ema.shadow_model.eval()
            samples = model_ema.shadow_model.sampling_DDIM_no_grad(args.n_samples, device=device, tau_type = args.tau_type)
            
                        
            # FID.clear_fake()
            fid.reset()
            inception.reset()
            

            save_image(
                samples,
                f"results/bilevel/gamma_{args.gamma}_gamma_end_{args.gamma_end}_y_loop_{args.inner_loop}_z_loop_{args.inner_loop_z}_initial_epoch_{args.initial_epoch}_epoch_{args.epochs}_lr_beta_{args.lr_beta}_lr_{args.lr}_lr_z_{args.lr_z}/{args.scheduler}/steps_{global_steps:08d}.png",
                nrow=int(math.sqrt(args.n_samples))
            )

            
            # do this as it is MNIST, no need for CIFAR-10
            samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
            samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
            fid.update(samples, real=False)
    
            fid_score = fid.compute()
            upper_loss = fid_score
            
            inception.update(samples)
            inception_score, std_inception_score = inception.compute()

            print("Upper loss (FID differentiable): ", upper_loss, "torch FID: ", fid_score, "torch IS: ", inception_score)

    
            # Write the iteration and scores to the CSV
            writer.writerow([epoch, step, upper_loss, fid_score, inception_score, scheduler.start_tensor.data.item(),scheduler.end_tensor.data.item(),scheduler.tau_tensor.data.item(),scheduler.s_tensor.data.item()])

            
        end_time = time.time()
        execution_time = end_time - start_time
        print("Time: ", execution_time)
    
        writer = csv.writer(f)
        writer.writerow(["Time", execution_time, "", "", "", "", "", ""])
            

if __name__ == "__main__":
    args = parse_args()
    seed_everything(0)
    main(args)

    
    