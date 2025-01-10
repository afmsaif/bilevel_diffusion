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
        for ema_param, param in zip(self.shadow_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow_model.load_state_dict(state_dict)



