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

from torchmetrics.metric import Metric


class FIDCalculator():
    # def __init__(self, real_features):
    def __init__(self, device):
        self.device = device
        # Compute mean and covariance of real features during initialization
        self.model__ = InceptionV3([block_idx],normalize_input=False).to(device)
        self.model__.train()
        # self.model__.eval()


        # Initialize accumulators
        self.sum_real = torch.zeros(2048, device=device)  # Shape: (D,)
        self.cov_sum_real = torch.zeros(2048, 2048, device=device)  # Shape: (D, D)
        self.real_features_num_samples = 0  # Total number of samples

        self.sum_fake = torch.zeros(2048, device=device)  # Shape: (D,)
        self.cov_sum_fake = torch.zeros(2048, 2048, device=device)  # Shape: (D, D)
        self.fake_features_num_samples = 0  # Total number of samples

        # self.model__.eval()
        # real_features = self._process_features(real_features)
        # self.mean_real = real_features.mean(dim=0)
        # self.cov_real = torch.cov(real_features.T)
        # self.trace_real = self.cov_real.trace()

    def clear_fake(self):
        self.sum_fake = torch.zeros(2048, device=self.device)  # Shape: (D,)
        self.cov_sum_fake = torch.zeros(2048, 2048, device=self.device)  # Shape: (D, D)
        self.fake_features_num_samples = 0  # Total number of samples

    def update_real(self,real_features):
        with torch.enable_grad():
            real_features = self._process_features(real_features)
            # print("print(real_features.requires_grad)", real_features.requires_grad)
            
            B, D = real_features.shape

            self.sum_real += real_features.sum(dim=0)
            self.cov_sum_real += real_features.t().mm(real_features)
            self.real_features_num_samples += B

    def update_fake(self,fake_features):
        # # self.model__.train()
        # for para in self.model__.parameters():
        #     para.requires_grad = True
        with torch.enable_grad():
            fake_features = self._process_features(fake_features)
            # print("print(fake_features.requires_grad)", fake_features.requires_grad)
            
            B, D = fake_features.shape

            self.sum_fake += fake_features.sum(dim=0)
            self.cov_sum_fake += fake_features.t().mm(fake_features)
            self.fake_features_num_samples += B

    def _process_features(self, features):
        """Process input features by passing through the model and squeezing dimensions."""
        features = self.model__(features.repeat(1, 3, 1, 1))[0]
        return features.squeeze(3).squeeze(2)

    def compute(self):
        """Compute FID score given generated features."""
        with torch.enable_grad():
            self.mean_real = (self.sum_real / self.real_features_num_samples).unsqueeze(0)

            cov_real_num = self.cov_sum_real - self.real_features_num_samples * self.mean_real.t().mm(self.mean_real)
            self.cov_real = cov_real_num / (self.real_features_num_samples - 1)
            self.trace_real = self.cov_real.trace()

            self.mean_fake = (self.sum_fake / self.fake_features_num_samples).unsqueeze(0)

            cov_fake_num = self.cov_sum_fake - self.fake_features_num_samples * self.mean_fake.t().mm(self.mean_fake)
            self.cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
            self.trace_fake = self.cov_fake.trace()
            
            # Compute FID components
            a = (self.mean_real.squeeze(0) - self.mean_fake.squeeze(0)).square().sum(dim=-1)
            b = self.trace_real + self.trace_fake
            try:
                c = torch.linalg.eigvals(self.cov_real @ self.cov_fake).sqrt().real.sum(dim=-1)
            except Exception as e:
                print("Error during eigenvalue computation:", e)
                # print("cov_real:", self.cov_real)
                # print("cov_fake:", self.cov_fake)
                # print("gen_features:", self.gen_features)
                # print("Product:", self.cov_real @ self.cov_fake)

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