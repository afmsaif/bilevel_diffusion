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

from torch.nn.modules.utils import _ntuple


import torchvision.models as models

from pytorch_fid.inception import InceptionV3

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

from torchmetrics.metric import Metric


from collections.abc import Sequence



from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

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


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["InceptionScore.plot"]


__doctest_requires__ = {("InceptionScore", "InceptionScore.plot"): ["torch_fidelity"]}


class IS_Calculator(Metric):


    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    features: list
    inception: Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[str, int, Module] = "logits_unbiased",
        splits: int = 10,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `InceptionScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "InceptionScore metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.splits = splits
        self.add_state("features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        # if imgs.dtype == torch.uint8:
        #     imgs_float = imgs.to(dtype=torch.float32) / 255.0
        # elif imgs.dtype == torch.float32:
        #     imgs_float = imgs
        # else:
        #     raise ValueError("Expecting image as torch.Tensor with dtype torch.uint8 or torch.float32")

        # imgs_float = imgs.to(dtype=torch.float32) / 255.0
        # imgs_float.requires_grad_(True)
        # with torch.enable_grad():  # Ensure gradients are enabled even if Inception disables them
        #     features = self.inception(imgs_float)
        # features = features.clone().detach().requires_grad_(True)
        # Extract features
        features = self.inception(imgs)
        self.features.append(features)

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute metric."""
        features = dim_zero_cat(self.features)
        features.requires_grad_(True)
        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.inception import InceptionScore
            >>> metric = InceptionScore()
            >>> metric.update(torch.randint(0, 255, (50, 3, 299, 299), dtype=torch.uint8))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the mean value by default

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.inception import InceptionScore
            >>> metric = InceptionScore()
            >>> values = [ ]
            >>> for _ in range(3):
            ...     # we index by 0 such that only the mean value is plotted
            ...     values.append(metric(torch.randint(0, 255, (50, 3, 299, 299), dtype=torch.uint8))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]  # by default we select the mean to plot
        return self._plot(val, ax)
