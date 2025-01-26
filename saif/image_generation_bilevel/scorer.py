import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import os
from torchmetrics.functional.multimodal import clip_score
from functools import partial

class RCGDMScorer(torch.nn.Module):
    ### Reward model borrowed from RCGDM
    def __init__(self):
        super().__init__()
        
        ## load the synthetic reward model which is based on a ResNet-18 model
        reward_model_path = "reward_model.pth"
        
        if os.path.exists(reward_model_path):
            self.reward_model = torch.load(reward_model_path)
        
        else:
            model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

            # Replace the final layer with a linear layer of scalar output
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            torch.save(model, reward_model_path)
        
            self.reward_model = torch.load(reward_model_path) 
        
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def __call__(self, images, prompts=[]):
        ## input: generated images
        target_size = 224
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
        
        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]
        ## image transform
        im_pix = torchvision.transforms.Resize(target_size)(images)
        im_pix = normalize(im_pix).to(images.dtype)
        
        return self.reward_model(im_pix)


class PenaltyScorer(torch.nn.Module):
    ### Reward model borrowed from RCGDM
    def __init__(self, gemma):
        super().__init__()
        
        ## load the synthetic reward model which is based on a ResNet-18 model
        reward_model_path = "reward_model.pth"
        self.reward_model = torch.load(reward_model_path) 
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()
        self.gemma = gemma
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def __call__(self, images, prompts):
        ## input: generated images
        target_size = 224
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
        
        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]
        ## image transform
        im_pix = torchvision.transforms.Resize(target_size)(images)
        im_pix = normalize(im_pix).to(images.dtype)

        ## calculate clip score
        clip = self.clip_score_fn(images, prompts).detach()
        
        return self.reward_model(im_pix)+ self.gemma * round(float(clip), 4)
           



    