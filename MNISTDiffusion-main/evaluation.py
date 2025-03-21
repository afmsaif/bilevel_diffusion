import numpy as np
import torch
from scipy.linalg import sqrtm
# from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from typing import Any, Optional, Union
from torch import Tensor
from torch.nn import Module

import math

from torch.nn.modules.utils import _ntuple


import torchvision.models as models

from pytorch_fid.inception import InceptionV3

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

from torchmetrics.metric import Metric


from collections.abc import Sequence



# from torchmetrics.image.fid import NoTrainInceptionV3
# from torchmetrics.utilities import rank_zero_warn
# from torchmetrics.utilities.data import dim_zero_cat
# from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
# from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

# import torch
# from torch import nn
# from torchvision.models import inception_v3

class FIDCalculator:
    def __init__(self, device):
        self.device = device

        # Load pre-trained InceptionV3 model and configure for penultimate layer extraction
        self.model__ = InceptionV3([block_idx],normalize_input=False).to(device)
        self.model__.fc = torch.nn.Identity()  # Remove final classification layer to extract penultimate features
        self.model__.train()
        # self.model__.eval()  # Set model to evaluation mode 

        # Initialize accumulators for real and fake features
        self.sum_real = torch.zeros(2048, device=device)  # Feature dimension is 2048
        self.cov_sum_real = torch.zeros(2048, 2048, device=device)
        self.real_features_num_samples = 0

        self.sum_fake = torch.zeros(2048, device=device)
        self.cov_sum_fake = torch.zeros(2048, 2048, device=device)
        self.fake_features_num_samples = 0

    def clear_fake(self):
        """Clear accumulated fake features."""
        self.sum_fake = torch.zeros(2048, device=self.device)
        self.cov_sum_fake = torch.zeros(2048, 2048, device=self.device)
        self.fake_features_num_samples = 0

    def update_real(self, real_images):
        """Update accumulators with features from real images."""
        real_features = self._extract_features(real_images)
        B, D = real_features.shape

        self.sum_real += real_features.sum(dim=0)
        self.cov_sum_real += real_features.t().mm(real_features)
        self.real_features_num_samples += B

    def update_fake(self, fake_images):
        """Update accumulators with features from fake images."""
        fake_features = self._extract_features(fake_images)
        B, D = fake_features.shape

        self.sum_fake += fake_features.sum(dim=0)
        self.cov_sum_fake += fake_features.t().mm(fake_features)
        self.fake_features_num_samples += B

    def _extract_features(self, features):
        """Extract features using the penultimate layer of InceptionV3."""
        
        # min_size = 299  # Default size for InceptionV3
        # if features.shape[2] < min_size or features.shape[3] < min_size:
        #     resize = Resize((min_size, min_size))
        #     features = resize(features)
        
        # Convert grayscale to RGB if needed
        if features.size(1) == 1:
            features = features.repeat(1, 3, 1, 1)
        
        # print(features.shape)
        features = self.model__(features)[0]
        
        # features = self.model__(features.repeat(1, 3, 1, 1))[0]  # Forward pass through the modified model
        
        # return features.squeeze(3).squeeze(2)
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

        return a + b - 2 * c



