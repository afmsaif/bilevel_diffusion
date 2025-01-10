# TO solve the problem: Error in magma_getdevice_arch: MAGMA not initialized (call magma_init() first) or bad device
# import cupy

# print(cupy.cuda.runtime.get_build_version())  # Checks if MAGMA is in the build


# cupy.cuda.runtime.magma_init()


import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model_bilevel import BilevelDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from copy import deepcopy
from evaluation import FIDCalculator

from torch.nn.parallel.scatter_gather import scatter

# # cmg added for check autograde
# torch.autograd.set_detect_anomaly(True)

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
    def __init__(self, timesteps, initial = None, device = 'cpu'):
        super().__init__()
        self.timesteps = timesteps
        self.initial = initial
        self.device = device
        
        if initial == "cosine":
            self.start_tensor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, requires_grad=True))
            self.end_tensor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
            self.tau_tensor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
            self.s_tensor = nn.Parameter(torch.tensor(0.008, dtype=torch.float32, requires_grad=True))
        elif initial == "sigmoid":
            self.start_tensor = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32, requires_grad=True))
            self.end_tensor = nn.Parameter(torch.tensor(3.0, dtype=torch.float32, requires_grad=True))
            self.tau_tensor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
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

    def _sigmoid_schedule(self, start=-3, end=3, timesteps=1000, tau=1.0):
        # A gamma function based on sigmoid function.
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32,device = self.device) 
        steps = steps * (end - start) + start
        f_t = torch.sigmoid((steps / timesteps)/tau)

        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    

    @property
    def betas(self):
        if self.initial == "cosine":
            return self._cosine_schedule(self.start_tensor, self.end_tensor, self.timesteps, self.tau_tensor, self.s_tensor)
        elif self.initial == "sigmoid":
            return self._sigmoid_schedule(self.start_tensor, self.end_tensor, self.timesteps, self.tau_tensor)
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
            DataLoader(test_dataset,batch_size=512,shuffle=True,num_workers=num_workers),\



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--lr_z',type = float ,default=0.001)
    parser.add_argument('--lr_beta',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=5)
    parser.add_argument('--inner_loop',type = int,help = 'steps of U-Net training',default=1)
    parser.add_argument('--gamma', type=float, default=1.0, help="penalty constant")
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'steps of forward pass',default=1000)
    parser.add_argument('--backward_steps',type = int,help = 'sampling steps of DDIM',default=100)
    parser.add_argument('--truncation',type = int,help = 'steps of truncated backpropagation',default=1)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    parser.add_argument(
        '--scheduler',
        choices=['cosine', 'linear', 'sigmoid'],
        default='cosine',
        help='Choose a scheduler type: "cosine", "linear", or "sigmoid"'
    )

    args = parser.parse_args()

    return args

# Wrap model in DataParallel if more than one GPU is available
def prepare_model(model, device):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
        model = MyDataParallel(model, device_ids=[0, 1])
    else:
        print(torch.cuda.device_count(), flush=True)
    return model.to(device)



def main(args):
    # device = "cpu" if args.cpu else "cuda"
    device = torch.device("cuda:0" if not args.cpu else "cpu")
    # device = torch.device("mps" if not args.cpu else "cpu")
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)

    # initial  =  None
    # initial  =  "cosine"
    initial = args.scheduler  
    print("Using ", initial)

    # Initialize model and learnable scheduler
    scheduler = LearnableScheduler(args.timesteps, initial, device)

    # FID score
    FID = FIDCalculator(device)
    for step, (image, _)in enumerate(test_dataloader):
        # initializing FID
        image = image.to(device)
        FID.update_real(image)


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
    
    back_steps=args.backward_steps
    inner_loop = args.inner_loop
    
    # model = DataParallel(model, device_ids = [0,1])

    # Unet for theta_y
    if isinstance(model, nn.DataParallel):
        theta_y = deepcopy(model.module.model)
        # model_ema = ExponentialMovingAverage(model.module, decay=1.0 - alpha, device=device)
        
        # Optimizers
        optimizer_theta_z = AdamW(model.module.model.parameters(), lr=args.lr_z)
        if initial  =="cosine":
            optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor, scheduler.s_tensor], lr=args.lr_beta)
        elif initial  =="sigmoid":
            optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor], lr=args.lr_beta)
        else:
            optimizer_scheduler = AdamW([scheduler._learnable_betas], lr=args.lr_beta)
        optimizer_theta_y = AdamW(theta_y.parameters(), lr=args.lr)
    else:
        theta_y = deepcopy(model.model)
        # model_ema = ExponentialMovingAverage(model, decay=1.0 - alpha, device=device)
        
        # Optimizers
        optimizer_theta_z = AdamW(model.model.parameters(), lr=args.lr_z)
        if initial  =="cosine":
            optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor, scheduler.s_tensor], lr=args.lr_beta)
        elif initial  =="sigmoid":
            optimizer_scheduler = AdamW([scheduler.start_tensor, scheduler.end_tensor, scheduler.tau_tensor], lr=args.lr_beta)
        else:
            optimizer_scheduler = AdamW([scheduler._learnable_betas], lr=args.lr_beta)
        optimizer_theta_y = AdamW(theta_y.parameters(), lr=args.lr)



    scheduler_theta_z=OneCycleLR(optimizer_theta_z,args.lr_z,total_steps=args.epochs*len(train_dataloader)*inner_loop,pct_start=0.25,anneal_strategy='cos')
    scheduler_theta_y=OneCycleLR(optimizer_theta_y,args.lr,total_steps=args.epochs*len(train_dataloader)*inner_loop,pct_start=0.25,anneal_strategy='cos')
    lr_scheduler_beta=OneCycleLR(optimizer_scheduler,args.lr_beta,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
 

    # Loss function
    loss_fn = nn.MSELoss(reduction='mean')

    global_steps = 0
    print("beta_value", torch.mean(scheduler.betas))
    
    # device = "cuda:1"

    for epoch in range(args.epochs):
        #TODO: check: clear cache in pytorch
        for step, (image, _) in enumerate(train_dataloader):
            # print("In epoch ",epoch, "step", step)
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
                if isinstance(model, nn.DataParallel):
                    x_t = model.module._forward_diffusion(image, t, noise)
                else:
                    x_t = model._forward_diffusion(image, t, noise)

                pred_noise_y = theta_y(x_t, t)
                lower_loss_y = loss_fn(pred_noise_y, noise)
                grads = torch.autograd.grad(lower_loss_y, theta_y.parameters())

                # lower_loss_y.backward()

                with torch.no_grad():
                        for param, grad in zip(theta_y.parameters(), grads):
                            param.grad = grad  # Example of manual gradient update


                optimizer_theta_y.step()
                scheduler_theta_y.step()
                
                
                # ===== Phase 2: Compute upper loss =====
                if isinstance(model, nn.DataParallel):
                    samples = model.module.sampling_DDIM(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device,tau_type = "linear")
                else:
                    samples = model.sampling_DDIM(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device,tau_type = "linear")
                
                # print("***********Samples***********")
                # print(samples[0][0])
                FID.clear_fake()
                FID.update_fake(samples)
                upper_loss = FID.compute()
                # print("upper_loss", upper_loss.requires_grad)
                
                # ===== Phase 3: Update theta_z using upper_loss/gamma + lower_loss_z =====
                noise = torch.randn_like(image)
                t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
                if isinstance(model, nn.DataParallel):
                    pred_noise_z = model(image, t, noise)
                else:
                    pred_noise_z = model(image, t, noise)

                
                lower_loss_z = loss_fn(pred_noise_z, noise)
    
                # Combine upper_loss and lower_loss_z
                combined_loss = (upper_loss / args.gamma) + lower_loss_z
                # combined_loss = lower_loss_z
                # combined_loss = upper_loss / args.gamma

                if isinstance(model, nn.DataParallel):
                    # combined_loss.backward()
                    # print(scheduler.start_tensor.grad)
                    # print(scheduler.end_tensor.grad)
                    torch.autograd.grad(combined_loss, model.module.model.parameters())

                    with torch.no_grad():
                        for param, grad in zip(model.model.parameters(), grads):
                            param.grad = grad  # Example of manual gradient update
                            if torch.isnan(grad).any():
                                print("There is nan in the grad")

                else:
                    grads = torch.autograd.grad(combined_loss, model.model.parameters())
                    # Apply gradients manually
                    with torch.no_grad():
                        for param, grad in zip(model.model.parameters(), grads):
                            param.grad = grad  # Example of manual gradient update

                optimizer_theta_z.step()
                scheduler_theta_z.step()

            # clear gradients
            optimizer_theta_z.zero_grad()
            optimizer_theta_y.zero_grad()
            optimizer_scheduler.zero_grad()
            
            # ===== Compute gradient of scheduler at y_star
            noise = torch.randn_like(image)
            t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 
            if isinstance(model, nn.DataParallel):
                x_t = model.module._forward_diffusion(image, t, noise)
            else:
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
            if isinstance(model, nn.DataParallel):
                samples = model.module.sampling_DDIM(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device,tau_type = "linear", steps=back_steps)
            else:
                samples = model.sampling_DDIM(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device,tau_type = "linear", steps=back_steps)


            FID.clear_fake()
            FID.update_fake(samples)
            upper_loss = FID.compute()


            grad_upper = torch.autograd.grad(upper_loss, scheduler.parameters())                



            # ==== calculate gradient of z_star
            # clear gradients
            optimizer_theta_z.zero_grad()
            optimizer_theta_y.zero_grad()
            optimizer_scheduler.zero_grad()
            
            noise = torch.randn_like(image)
            t = torch.randint(0, args.timesteps, (image.size(0),)).to(device) 

            if isinstance(model, nn.DataParallel):
                pred_noise_z_star = model(image, t, noise)
            else:
                pred_noise_z_star = model(image, t, noise)

            
            lower_loss_z_star = loss_fn(pred_noise_z_star, noise)

            grad_lower_z_star = torch.autograd.grad(lower_loss_z_star, scheduler.parameters())

            combined_grad = [
                g_upper/args.gamma + (g_z - g_y)
                for g_upper, g_z, g_y in zip(grad_upper, grad_lower_z_star, grad_lower_y_star)
            ]


        
            for param, grad in zip(scheduler.parameters(), combined_grad):
                # print(param,grad)
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


            # # Update EMA model
            # if global_steps % args.model_ema_steps == 0:
            #     model_ema.update_parameters(model)

            global_steps += 1
            if step % args.log_freq == 0:
                if initial  == "cosine":
                    print("start: ", scheduler.start_tensor.data)
                    print("end: ", scheduler.end_tensor.data)
                    print("tau: ", scheduler.tau_tensor.data)
                    print("s: ", scheduler.s_tensor.data)
                elif initial  == "sigmoid":
                    print("start: ", scheduler.start_tensor.data)
                    print("end: ", scheduler.end_tensor.data)
                    print("tau: ", scheduler.tau_tensor.data)
                else:
                    print("max gradient, ", [max(g) for g in combined_grad])
                    print("beta_value", torch.mean(scheduler.betas))

                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(train_dataloader)}], "
                          f"Lower Loss Z: {lower_loss_z.item():.5f}, Upper Loss: {upper_loss.item():.5f}, Lower Loss y: {lower_loss_y.item():.5f}, Gamma: {args.gamma}")



        # Save checkpoint and generate samples
        ckpt = {
            "model": model.state_dict(),
            # "model_ema": model_ema.state_dict()
        }
        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, f"results/bilevel/steps_{global_steps:08d}.pt")

        model.eval()
        samples = model.sampling_no_grad(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        FID.clear_fake()
        FID.update_fake(samples)
        save_image(samples, f"results/bilevel/steps_{global_steps:08d}.png", nrow=int(math.sqrt(args.n_samples)))
        print("Final FID:", FID.compute())

        samples = model.sampling_DDIM_no_grad(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        FID.clear_fake()
        FID.update_fake(samples)
        save_image(samples, f"results/bilevel/steps_{global_steps:08d}.png", nrow=int(math.sqrt(args.n_samples)))
        print("Final FID for DDIM:", FID.compute())

if __name__ == "__main__":
    args = parse_args()
    seed_everything(0)
    main(args)

    
    