import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm

class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8],back_steps=50,start=0, end=1, tau=1, epsilon=0.008):
        super().__init__()
        self.timesteps=timesteps
        self.back_steps=back_steps
        self.in_channels=in_channels
        self.image_size=image_size

        betas=self._cosine_variance_schedule(start,end,timesteps,tau,epsilon)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self,x,noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t)

        return pred_noise

    @torch.no_grad()
    def sampling_DDIM(self,n_samples,clipped_reverse_diffusion=True,device="cuda", tau_type= "None", steps=None):
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
    
    def _cosine_variance_schedule(self, start=0, end=1, timesteps=1000, tau=1, epsilon=0.008):
        
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32) 
        steps = steps * (end - start) + start
        
        cos_value = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
        cos_value = torch.clamp(cos_value, min=0, max=1)  # Ensure non-negative for exponentiation
        f_t = cos_value ** (2 * tau)

        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

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
    
    # @torch.no_grad()
    # def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"):
    #     x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
    #     for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
    #         noise=torch.randn_like(x_t).to(device)
    #         t=torch.tensor([i for _ in range(n_samples)]).to(device)

    #         if clipped_reverse_diffusion:
    #             x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
    #         else:
    #             x_t=self._reverse_diffusion(x_t,t,noise)

    #     x_t=(x_t+1.)/2. #[-1,1] to [0,1]

    #     return x_t
    # @torch.no_grad()
    # def _reverse_diffusion(self,x_t,t,t_prev,noise):
    #     '''
    #     p(x_{t-1}|x_{t})-> mean,std

    #     pred_noise-> pred_mean and pred_std
    #     '''
    #     pred=self.model(x_t,t)

    #     alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

    #     if t.min()>0:
    #         alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t_prev).reshape(x_t.shape[0],1,1,1) if t_prev.min()>=0 else torch.tensor(1.).to(x_t.device)
    #         std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
    #     else:
    #         std=0.0

    #     return mean+std*noise 



    # @torch.no_grad()
    # def _reverse_diffusion_with_clip(self,x_t,t,t_prev,noise): 
    #     '''
    #     p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

    #     pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
    #     '''
    #     pred=self.model(x_t,t)
    #     alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
    #     x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
    #     x_0_pred.clamp_(-1., 1.)

    #     if t.min()>0:
    #         alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t_prev).reshape(x_t.shape[0],1,1,1) if t_prev.min()>=0 else torch.tensor(1.).to(x_t.device)
    #         mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
    #              ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

    #         std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
    #     else:
    #         mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
    #         std=0.0

    #     return mean+std*noise 
    