import torch.nn as nn
import torch
# import math
from unet import Unet
# from tqdm import tqdm



class BilevelDiffusion(nn.Module):
    def __init__(self,scheduler,image_size,in_channels,time_embedding_dim=256,timesteps=1000,back_steps=50,truncation=5,base_dim=32,dim_mults= [1, 2, 4, 8]):
        super().__init__()
        self.timesteps=timesteps
        self.back_steps=back_steps
        self.truncation=truncation
        self.in_channels=in_channels
        self.image_size=image_size
        self.scheduler=scheduler
        
        #save for deepcopy
        self.time_embedding_dim = time_embedding_dim
        self.base_dim= base_dim
        self.dim_mults = dim_mults

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    @property
    def betas(self):
        return self.scheduler.get_betas()

    @property
    def alphas(self):
        return 1.- self.betas

    @property
    def alphas_cumprod(self):
        return torch.cumprod(self.alphas,dim=-1)

    @property
    def sqrt_alphas_cumprod(self):
        return torch.sqrt(self.alphas_cumprod)

    @property
    def sqrt_one_minus_alphas_cumprod(self):
        return torch.sqrt(1.-self.alphas_cumprod)

    def forward(self,x,t,noise):
        # x:NCHW
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t)


        return pred_noise


    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    # Changed sampling to DDIM according to https://github.com/huggingface/diffusers/blob/v0.11.0/src/diffusers/schedulers/scheduling_ddim.py#L78    
    def sampling_DDIM(self,n_samples,device="cuda", tau_type= "None", steps=None, truncation=None):

        steps = steps if steps is not None else self.back_steps
        truncation = truncation if truncation is not None else self.truncation

        if steps is not None and tau_type in ["linear", "quadratic"]:
            if tau_type == "linear":
                self.tau = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long).to(device)
            elif tau_type == "quadratic":
                self.tau = (torch.linspace(0, torch.sqrt(torch.tensor(self.timesteps * 0.8)), steps) ** 2).to(torch.int64)
            else:
                raise ValueError("Invalid tau_type. Choose 'linear' or 'quadratic'.")

        self.tau_prev = torch.cat([torch.tensor([-1]).to(self.tau.device), self.tau[:-1]])

        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)

        # for i in tqdm(range(steps-1,-1,-1),desc="Sampling"):
        for i in range(steps-1,-1,-1):
            if steps is not None and tau_type in ["linear", "quadratic"]:
                t_indices = self.tau[i]  # Map t to tau indices
                t_indices_prev = self.tau_prev[i]
            else:
                t_indices = i  # Default to standard timesteps
                t_indices_prev = i-1

            noise=torch.randn_like(x_t).to(device)
            t = torch.tensor([t_indices for _ in range(n_samples)]).to(device)
            t_prev = torch.tensor([t_indices_prev for _ in range(n_samples)]).to(device)

            x_t=self._reverse_diffusion_DDIM(x_t,t,t_prev,noise)
            
            if i>=truncation:
                x_t = x_t.detach()

        # x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        x_t = x_t.clamp(-1., 1.)

        return x_t

    def _reverse_diffusion_DDIM(self,x_t,t,t_prev,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t_cumprod = self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1,t_prev).reshape(x_t.shape[0],1,1,1) if t_prev.min()>=0 else torch.tensor(1.).to(x_t.device)
        

        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * pred)/ torch.sqrt(alpha_t_cumprod)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = torch.sqrt(1 - alpha_t_cumprod_prev) * pred

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = torch.sqrt(alpha_t_cumprod_prev) * x_0_pred + pred_sample_direction

        return prev_sample

    @torch.no_grad()
    def sampling_DDIM_no_grad(self,n_samples,clipped_reverse_diffusion=True,device="cuda", tau_type= "None", steps=None):
        steps = steps if steps is not None else self.timesteps

        if steps is not None and tau_type in ["linear", "quadratic"]:
            if tau_type == "linear":
                self.tau = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long).to(device)
            elif tau_type == "quadratic":
                self.tau = (torch.linspace(0, torch.sqrt(torch.tensor(self.timesteps * 0.8)), steps) ** 2).to(torch.int64)
            else:
                raise ValueError("Invalid tau_type. Choose 'linear' or 'quadratic'.")

        self.tau_prev = torch.cat([torch.tensor([-1]).to(self.tau.device), self.tau[:-1]])

        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)

        # for i in tqdm(range(steps-1,-1,-1),desc="Sampling"):
        for i in range(steps-1,-1,-1):
            if steps is not None and tau_type in ["linear", "quadratic"]:
                t_indices = self.tau[i]  # Map t to tau indices
                t_indices_prev = self.tau_prev[i]
            else:
                t_indices = i  # Default to standard timesteps
                t_indices_prev = i-1


            t = torch.tensor([t_indices for _ in range(n_samples)]).to(device)
            t_prev = torch.tensor([t_indices_prev for _ in range(n_samples)]).to(device)

            x_t=self._reverse_diffusion_DDIM_no_grad(x_t,t,t_prev)

        # x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        x_t = x_t.clamp(-1., 1.)

        return x_t
    

    @torch.no_grad()
    def _reverse_diffusion_DDIM_no_grad(self,x_t,t,t_prev): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t_cumprod = self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1,t_prev).reshape(x_t.shape[0],1,1,1) if t_prev.min()>=0 else torch.tensor(1.).to(x_t.device)
        

        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * pred)/ torch.sqrt(alpha_t_cumprod)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = torch.sqrt(1 - alpha_t_cumprod_prev) * pred

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = torch.sqrt(alpha_t_cumprod_prev) * x_0_pred + pred_sample_direction

        return prev_sample
    
