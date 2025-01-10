import numpy as np
import torch
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from typing import Any, Optional, Union
from torch import Tensor
from torch.nn import Module

import math

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple


import torch
import torchvision.models as models

from pytorch_fid.inception import InceptionV3

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]




class FIDCalculator:
    def __init__(self, real_features):
        # Compute mean and covariance of real features during initialization
        self.model__ = InceptionV3([block_idx]).cuda()
        self.model__.eval()
        real_features = self._process_features(real_features)
        self.mean_real = real_features.mean(dim=0)
        self.cov_real = torch.cov(real_features.T)
        self.trace_real = self.cov_real.trace()

    def _process_features(self, features):
        """Process input features by passing through the model and squeezing dimensions."""
        features = self.model__(features.repeat(1, 3, 1, 1))[0]
        return features.squeeze(3).squeeze(2)

    def compute(self, gen_features):
        """Compute FID score given generated features."""
        # Process generated features
        self.model__.train()
        gen_features = self._process_features(gen_features)
        
        # Compute mean and covariance of generated features
        mean_fake = gen_features.mean(dim=0)
        cov_fake = torch.cov(gen_features.T)
        
        # Compute FID components
        a = (self.mean_real.squeeze(0) - mean_fake.squeeze(0)).square().sum(dim=-1)
        b = self.trace_real + cov_fake.trace()
        try:
            c = torch.linalg.eigvals(self.cov_real @ cov_fake).sqrt().real.sum(dim=-1)
        except Exception as e:
            print("Error during eigenvalue computation:", e)
            print("cov_real:", self.cov_real)
            print("cov_fake:", cov_fake)
            print("gen_features:", gen_features)
            print("Product:", self.cov_real @ cov_fake)

        return a + b - 2 * c


# def My_fid(real_features, gen_features): 
#     real_features = model__(real_features.repeat(1, 3, 1, 1))[0]
#     gen_features = model__(gen_features.repeat(1, 3, 1, 1))[0]

#     real_features = real_features.squeeze(3).squeeze(2)
#     gen_features = gen_features.squeeze(3).squeeze(2)


#     mean_real = real_features.mean(dim=0)
#     mean_fake = gen_features.mean(dim=0)
    

#     # Compute covariances
#     # print("dim", real_features.shape, gen_features.shape)
#     cov_real = torch.cov(real_features.T)
#     cov_fake = torch.cov(gen_features.T)


#     a = (mean_real.squeeze(0) - mean_fake.squeeze(0)).square().sum(dim=-1)
#     b = cov_real.trace() + cov_fake.trace()
#     try:
#         c = torch.linalg.eigvals((cov_real) @ (cov_fake)).sqrt().real.sum(dim=-1)
#     except:
#         print((cov_real))
#         print((cov_fake))
#         print("gen_features", gen_features)
#         print((cov_real) @ (cov_fake))


#     return a + b - 2 * c   