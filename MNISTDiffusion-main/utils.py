import torch

# #torchvision ema implementation
# #https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
# class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
#     """Maintains moving averages of model parameters using an exponential decay.
#     ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
#     `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
#     is used to compute the EMA.
#     """

#     def __init__(self, model, decay, device="cpu"):
#         def ema_avg(avg_model_param, model_param, num_averaged):
#             return decay * avg_model_param + (1 - decay) * model_param

#         super().__init__(model, device, ema_avg, use_buffers=True)

from copy import deepcopy
import torch
from model_bilevel import BilevelDiffusion

class ExponentialMovingAverage:
    """Maintains moving averages of model parameters using an exponential decay."""

    def __init__(self, model, decay=0.995, device="cpu"):
        self.decay = decay
        self.device = device

        # Initialize a shadow model using state_dict
        self.shadow_model = self._initialize_shadow_model(model)
        self.shadow_model.to(self.device)

        self.n_averaged = torch.tensor(0, dtype=torch.long, device=device)


    def _initialize_shadow_model(self, model):
        # Create a new instance of BilevelDiffusion with the same parameters
        shadow_model = BilevelDiffusion(
            scheduler=deepcopy(model.scheduler),  # Reuse the scheduler instance
            image_size=model.image_size,
            in_channels=model.in_channels,
            timesteps=model.timesteps,
            base_dim=model.base_dim,  # Directly access the Unet attribute
            dim_mults=model.dim_mults,  # Access dim_mults from Unet
        )

        shadow_model.load_state_dict(model.state_dict())  # Copy weights

        return shadow_model

    def update_parameters(self, model):
        # Update the parameters of the EMA model

        # for (ema_name, ema_param), (name, param) in zip(
        #     self.shadow_model.named_parameters(), model.named_parameters()
        # ):
        #     # ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        #     ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data
        if self.n_averaged > 0:
            for ema_param, param in zip(self.shadow_model.parameters(), model.parameters()):
                # print("before:", torch.mean(ema_param.data), torch.mean(param.data))
                # ema_param.detach()
                # ema_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)
                # ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                ema_param.detach().copy_(self.decay * ema_param.detach() + (1 - self.decay) * param)
                # ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                # print("after:", torch.mean(ema_param.data))

            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.shadow_model.buffers(), model.buffers()):
                # print(b_swa.data.dtype, b_model.data.dtype)
                b_swa.detach().copy_(self.decay * b_swa.detach() + (1 - self.decay) * b_model)
                # b_swa.data.mul_(self.decay).add_(b_model.data, alpha=1.0 - self.decay)
        else:
            for ema_param, param in zip(self.shadow_model.parameters(), model.parameters()):
                # print("before:", torch.mean(ema_param.data), torch.mean(param.data))
                # ema_param.detach()
                # ema_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)
                ema_param.detach().copy_(param.detach().to(ema_param.device))
                # ema_param.data.mul_(0.0).add_(param.data)

            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.shadow_model.buffers(), model.buffers()):
                # print(b_swa.data.dtype, b_model.data.dtype)
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1
        

    # if self.n_averaged > 0:
    # else:
    #         for ema_param, param in zip(self.shadow_model.parameters(), model.parameters()):
    #             ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    #         for p_averaged, p_model in zip(  # type: ignore[assignment]
    #                 self_param_detached, model_param_detached
    #             ):
    #                 n_averaged = self.n_averaged.to(p_averaged.device)
    #                 p_averaged.detach().copy_(
    #                     self.avg_fn(p_averaged.detach(), p_model, n_averaged)
    #                 )



    def state_dict(self):
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow_model.load_state_dict(state_dict)



