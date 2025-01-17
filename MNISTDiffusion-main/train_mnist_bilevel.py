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

import sys
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


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
    parser.add_argument('--lr_beta',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=5)
    parser.add_argument('--inner_loop',type = int,help = 'steps of U-Net training',default=1)
    parser.add_argument('--inner_loop_z',type = int,help = 'steps z update',default=1)
    parser.add_argument('--initial_epoch',type = int,help = 'initial steps of U-Net training',default=50)
    parser.add_argument('--gamma', type=float, default=1.0, help="penalty constant")
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



def main(args):
    # device = "cpu" if args.cpu else "cuda"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if not args.cpu else "cpu")
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)

    initial = args.scheduler  
    print("Using ", initial, " Scheduler")

    # Initialize model and learnable scheduler
    scheduler = LearnableScheduler(args.timesteps, initial, device,args.initial_start,args.initial_end,args.initial_tau,args.initial_epsilon)

    # differentiable FID score
    FID = FIDCalculator(device)
        
    # using the torchmetrics FID
    fid = FrechetInceptionDistance(normalize=True,reset_real_features=False).to(device)
    
    inception = InceptionScore(normalize=True).to(device)

   

    # initializing FID with real features
    for step, (image, _)in enumerate(test_dataloader):

        image = image.to(device)


        # differntiable FID needs input scale at [-1,1]
        FID.update_real(image)

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


    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    print("adjust:", adjust)
    print("alpha:", alpha)
    model_ema = ExponentialMovingAverage(model, decay=1.0 - alpha, device=device)

    # model_single = MNISTDiffusion(timesteps=args.timesteps,
    #                      back_steps=args.backward_steps,
    #             image_size=28,
    #             in_channels=1,
    #             base_dim=args.model_base_dim,
    #             dim_mults=[2,4],
    #             start=0.1,
    #             end=1.0,
    #             epsilon=0.02,
    #             tau=3.0).to(device)
    # model_ema_single = ExponentialMovingAverage_single(model_single, device=device, decay=1.0 - alpha)

    
    back_steps=args.backward_steps
    inner_loop = args.inner_loop
    inner_loop_z = args.inner_loop_z
    

    # Unet for theta_y
    theta_y = deepcopy(model.model)
    # model_forward = deepcopy(model)
    
    # Optimizers
    optimizer_theta_z = AdamW(model.model.parameters(), lr=args.lr_z)
    if initial  =="cosine":
        optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor, scheduler.s_tensor], lr=args.lr_beta)
    elif initial  =="sigmoid":
        optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor], lr=args.lr_beta)
    else:
        optimizer_scheduler = AdamW([scheduler._learnable_betas], lr=args.lr_beta)
    optimizer_theta_y = AdamW(theta_y.parameters(), lr=args.lr)



    scheduler_theta_z=OneCycleLR(optimizer_theta_z,args.lr_z,total_steps=(args.epochs-args.initial_epoch)*len(train_dataloader)*inner_loop_z,pct_start=0.25,anneal_strategy='cos')
    scheduler_theta_y=OneCycleLR(optimizer_theta_y,args.lr,total_steps=args.epochs*len(train_dataloader)*inner_loop,pct_start=0.25,anneal_strategy='cos')
    lr_scheduler_beta=OneCycleLR(optimizer_scheduler,args.lr_beta,total_steps=(args.epochs-args.initial_epoch)*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
 

    # Loss function
    loss_fn = nn.MSELoss(reduction='mean')

    global_steps = 0
    print("beta_value", torch.mean(scheduler.betas))
    
    scheduler_type = getattr(args, 'scheduler_type', 'default_value')
    
    output_file = "results/bilevel/gamma_{:.3f}/{}/output/scores.csv".format(args.gamma,scheduler_type)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    header = ["Epoch", "Upper Loss", "FID Score", "IS Score","start","end","tau","epsilon"]
    file_exists = os.path.isfile(output_file)
    
    flag=0
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header only if the file is new
        writer.writerow(header)

        for epoch in range(args.epochs):
            for step, (image, _) in enumerate(train_dataloader):
                # if step>1:
                #     continue
                
                model.train()
                image = image.to(device)
                
                
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
                    # clear gradients
                    optimizer_theta_z.zero_grad()
                    optimizer_theta_y.zero_grad()
                    optimizer_scheduler.zero_grad()
                    
                    if flag == 0:
                        model.model = deepcopy(theta_y)
                        flag = 1
                    
                    for _ in range(inner_loop_z):  # Inner optimization steps for z
                        # clear gradients
                        optimizer_theta_y.zero_grad()
                        optimizer_theta_z.zero_grad()
                        optimizer_scheduler.zero_grad()
                        
                        
                        # ===== Phase 2: Compute upper loss =====
                        # samples = model.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type)
                        samples = model_ema.shadow_model.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type)
                        # clear FID cashe
                        FID.clear_fake()
                        fid.reset()
                        inception.reset()
                        
                        FID.update_fake(samples)
                        upper_loss = FID.compute()
                        
                        # do this as it is MNIST, no need for CIFAR-10
                        samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
                        samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
                        
                        fid.update(samples, real=False)                
                        fid_score = fid.compute()
                        
                        inception.update(samples)
                        inception_score, std_inception_score = inception.compute()
        
                        
                        # ===== Phase 3: Update theta_z using upper_loss/gamma + lower_loss_z =====
                        noise = torch.randn_like(image)
                        t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
                        pred_noise_z = model(image, t, noise)
        
                        
                        lower_loss_z = loss_fn(pred_noise_z, noise)
            
                        # Combine upper_loss and lower_loss_z
                        grad_upper = torch.autograd.grad(upper_loss, model_ema.shadow_model.model.parameters()) 
                        grad_lower = torch.autograd.grad(lower_loss_z, model.model.parameters()) 
                        
                        # Apply gradients manually
                        with torch.no_grad():
                            for param, grad_u, grad_l in zip(model.model.parameters(), grad_upper,grad_lower):
                                param.grad = grad_u / args.gamma + grad_l  # Example of manual gradient update

                        # combined_loss = (upper_loss / args.gamma) + lower_loss_z
                        
                        # grads = torch.autograd.grad(combined_loss, model.model.parameters())
                        # # Apply gradients manually
                        # with torch.no_grad():
                        #     for param, grad in zip(model.model.parameters(), grads):
                        #         param.grad = grad  # Example of manual gradient update
        
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
                    samples = model_ema.shadow_model.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type, steps=back_steps)
        
        
                    FID.clear_fake()
                    fid.reset()
                    inception.reset()
                    FID.update_fake(samples)
                    upper_loss = FID.compute()
                    
                    # do this as it is MNIST, no need for CIFAR-10
                    samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
                    samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
                    fid.update(samples, real=False)
            
                    fid_score = fid.compute()
                    
                    inception.update(samples)
                    inception_score, std_inception_score = inception.compute()
        
                    # print("requires_grad", upper_loss.requires_grad)
                    # grad_upper = torch.autograd.grad(upper_loss, scheduler.parameters())                
                    grad_upper = torch.autograd.grad(upper_loss, model_ema.shadow_model.scheduler.parameters())                
        
        
        
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
                        g_upper/args.gamma + (g_z - g_y)
                        for g_upper, g_z, g_y in zip(grad_upper, grad_lower_z_star, grad_lower_y_star)
                    ]
        
        
                
                    for param, grad in zip(scheduler.parameters(), combined_grad):
                        param.grad = grad
        
                    optimizer_scheduler.step()
                    lr_scheduler_beta.step()
                
                    if initial  == "cosine":
                        scheduler.start_tensor.data.clamp_(0, 1)
                        scheduler.end_tensor.data.clamp_(scheduler.start_tensor + 1e-10, torch.tensor(1.0, dtype=torch.float32).to(device))
                        scheduler.tau_tensor.data.clamp_(0, 1e6)
                        scheduler.s_tensor.data.clamp_(0,1)
                    if initial == "sigmoid":
                        scheduler.end_tensor.data.clamp_(scheduler.start_tensor + 1e-10, torch.tensor(1e6, dtype=torch.float32).to(device))
                        scheduler.tau_tensor.data.clamp_(0, 1e6)
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
                            # print("start: ", scheduler.start_tensor.data.item(), ". end:", scheduler.end_tensor.data.item(), ". tau:", scheduler.tau_tensor.data.item(), ". epsilon", scheduler.s_tensor.data.item())
                            # print("end: ", scheduler.end_tensor.data.item())
                            # print("tau: ", scheduler.tau_tensor.data.item())
                            # print("s: ", scheduler.s_tensor.data.item())

                        else:
                            print("max gradient, ", [max(g) for g in combined_grad])
                            print("beta_value", torch.mean(scheduler.betas))
                   
                    # Update EMA model
                    if global_steps % args.model_ema_steps == 0:
                        # print("update model 2")
                        model_ema.update_parameters(model)
                        # if initial  == "cosine":
                        #     print("****check model schedular****")
                        #     print("start: ", scheduler.start_tensor.data)
                        #     print("end: ", scheduler.end_tensor.data)
                        #     print("tau: ", scheduler.tau_tensor.data)
                        #     print("s: ", scheduler.s_tensor.data)
                        #     print("****check model_ema schedular****")
                        #     print("start: ", model_ema.shadow_model.scheduler.start_tensor.data)
                        #     print("end: ", model_ema.shadow_model.scheduler.end_tensor.data)
                        #     print("tau: ", model_ema.shadow_model.scheduler.tau_tensor.data)
                        #     print("s: ", model_ema.shadow_model.scheduler.s_tensor.data)
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
                                  f"Lower Loss Z: {lower_loss_z.item():.5f}, Upper Loss: {upper_loss.item():.5f}, torch FID: {fid_score.item():.5f}, torch IS: {inception_score.item():.5f}, Lower Loss y: {lower_loss_y.item():.5f}, Gamma: {args.gamma}")
                    else:
                        print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(train_dataloader)}], "
                                  f"Lower Loss y: {lower_loss_y.item():.5f}, Gamma: {args.gamma}")

    

            # try:
            # Save checkpoint and generate samples
            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict()
            }
            os.makedirs("results", exist_ok=True)
            torch.save(ckpt, f"results/bilevel/gamma_{args.gamma:.3f}/steps_{global_steps:08d}.pt")
            model_ema.shadow_model.eval()
            samples = model_ema.shadow_model.sampling_DDIM_no_grad(args.n_samples, device=device, tau_type = args.tau_type)
            
            # model.eval()
            # samples = model.sampling_DDIM_no_grad(args.n_samples, device=device, tau_type = args.tau_type)
            
            FID.clear_fake()
            fid.reset()
            inception.reset()
            
            FID.update_fake(samples)
            upper_loss = FID.compute()
            save_image(samples, f"results/bilevel/gamma_{args.gamma:.3f}/steps_{global_steps:08d}.png", nrow=int(math.sqrt(args.n_samples)))
            # print("Final FID for DDIM:", upper_loss)
            
            # do this as it is MNIST, no need for CIFAR-10
            samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
            samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
            fid.update(samples, real=False)
    
            fid_score = fid.compute()
            
            inception.update(samples)
            inception_score, std_inception_score = inception.compute()

            print("Upper loss (FID differentiable): ", upper_loss, "torch FID: ", fid_score, "torch IS: ", inception_score)

            # model_ema_single.eval()
            # samples = model_ema_single.module.sampling_DDIM(args.n_samples, device=device, tau_type = args.tau_type)
            # FID.clear_fake()
            # fid.reset()
            # FID.update_fake(samples)
            # upper_loss = FID.compute()
            # samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
            # samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
            # fid.update(samples, real=False)
    
            # fid_score = fid.compute()
    
            # print("Single: Upper loss (FID differentiable): ", upper_loss, "torch FID: ", fid_score)
    
            # Write the iteration and scores to the CSV
            writer.writerow([epoch, upper_loss, fid_score, inception_score, scheduler.start_tensor.data.item(),scheduler.end_tensor.data.item(),scheduler.tau_tensor.data.item(),scheduler.s_tensor.data.item()])
            f.flush()
            # except:
            #     ckpt = {
            #         "model": model.state_dict(),
            #     }
            #     os.makedirs("results", exist_ok=True)
            #     torch.save(ckpt, f"results/bilevel/gamma_{args.gamma:.3f}/steps_{global_steps:08d}.pt")
            #     print("No evaluation in initial stages")
            

if __name__ == "__main__":
    args = parse_args()
    seed_everything(0)
    main(args)

    
    