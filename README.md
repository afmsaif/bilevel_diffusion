# A First-order Generative Bilevel Optimization Framework for Diffusion Models

## Introduction

Diffusion models have emerged as a leading class of generative models, capable of synthesizing high-fidelity images by reversing a gradual noising process. Despite their empirical success, tuning diffusion samplers for specific objectives—such as aligning outputs with downstream reward functions or designing optimal noise schedules—remains an open challenge. Traditional approaches rely on manual heuristics or multi-stage training, which can be unstable and inefficient.

In this work, we introduce a unified bilevel optimization framework that directly integrates generative sampling with task-specific objectives. By formulating diffusion sampling as a lower-level problem with entropic regularization and treating hyperparameters (e.g., entropy strength or noise schedule) as upper-level variables, our approach provides principled gradients for fine-tuning. Specifically:

1. **Reward Fine-Tuning (Section 3.1):** We propose a bilevel scheme for adapting sampling trajectories to maximize arbitrary reward functions (e.g., CLIP-based aesthetic scores), learning an optimal entropy parameter that balances fidelity with diversity.

2. **Noise Scheduling (Section 3.2):** We derive first-order updates for continuous noise schedules on MNIST, enabling automatic discovery of schedules that improve generative quality under a fixed budget of reverse diffusion steps.

Our contributions are threefold:

1. A general bilevel formulation for diffusion-based generative modeling, compatible with both score-based SDEs and discrete denoising flows.  
2. Scalable, Monte Carlo estimators for upper-level gradients, allowing end-to-end fine-tuning without nested optimization loops.  
3. Empirical validation on aesthetic reward tuning and noise schedule discovery, demonstrating improved sample quality and reduced generation cost.  

This repository provides modular implementations for both tasks, facilitating reproducibility and extension to new datasets and reward functions.

This repository contains the code to reproduce all experiments presented in our ICML 2025 paper:

> *A First-order Generative Bilevel Optimization Framework for Diffusion Models*  
> Quan Xiao, Hui Yuan, A. F. M. Saif, Gaowen Liu, Ramana Kompella, Mengdi Wang, Tianyi Chen


<p float="left">
  <img src="https://github.com/afmsaif/bilevel_diffusion/blob/main/MNISTDiffusion-main/docs/exp1-2.png" width="45%" />
  <img src="https://github.com/afmsaif/bilevel_diffusion/blob/main/MNISTDiffusion-main/docs/exp2.png" width="45%" />
</p>

---

## 📂 Repository Structure

```text
├── MNISTDiff/            # Section 3.2: Noise scheduling experiments on MNIST
│   ├── train_mnist.py/          
│   ├── train_mnist_bayesian.py/         
│   └── train_mnist_bilevel_ZO.py/   
├── saif/                 # Section 3.1: Reward fine-tuning modules
│   ├── scripts/          # Entry-point for reward fine-tuning
│   ├── configs/          # Hyperparameter YAML files
│   └── outputs/          # Learned λ, samples, and logs
├── requirements.txt      # Python dependencies
└── README.md             # This overview
```

---

## 🛠️ Installation

Ensure you have Python 3.8+ and install the required packages:

```bash
pip install -r requirements.txt
```

Dependencies include:

* PyTorch (>=1.10)
* diffusers
* transformers
* torchvision
* tqdm
* pyyaml
* CLIP (for reward models)

---

## ⚙️ Usage

### Section 3.1: Reward Fine-Tuning (`saif`)

Reproduce the bilevel reward fine-tuning from Section 3.1:

```bash
cd saif
python scripts/reward_finetune.py \
  --config configs/reward_config.yaml \
  --pretrained_model_path "CompVis/stable-diffusion-v1-5" \
  --lower_reward_model models/resnet18_reward.pth \
  --upper_reward clip \
  --num_samples 256 \
  --lambda_init 0.01 \
  --gamma 1e3 \
  --steps 500 \
  --output_dir outputs/reward_finetune
```

Outputs are stored under `saif/outputs/reward_finetune`:

* `lambda_final.txt`: learned entropy strength λ
* `samples/`: generated images under the fine-tuned policy
* `logs/`: training curves and gradient trajectories

---

### Section 3.2: Noise Scheduling (`MNISTDiff`)

Run the bilevel noise schedule optimization for Section 3.2 on the MNIST dataset:

```bash
cd MNISTDiff
python train_mnist_bilevel_ZO.py \
  --inner_loop 10 \
  --inner_loop_z 1 \
  --gamma 1 \
  --gamma_end 1 \
  --lr_beta 0.05 0.05 1 0 \
```

* optimized noise schedule parameters
* sampled trajectories and denoised images
* training logs for upper- and lower-level objectives

---

## 📖 Algorithmic Summary

Both modules implement our generative bilevel framework:

1. **Lower-level**: SDE-based sampling adjusted by a surrogate loss (reward or scheduling penalty) with entropic regularization.
2. **Upper-level**: First-order update of the regularization strength (λ) or schedule parameters via Monte Carlo gradient estimates.

Refer to Algorithms 2 & 5 in the paper for detailed pseudo‑code and mathematical derivations.

---

## 📑 Citation

If you use this code, please cite:

If you find our work interesting, please consider citing this paper:
```
@inproceedings{xiao2025first,
  title={A First-order Generative Bilevel Optimization Framework for Diffusion Models},
  author={Xiao, Quan and Yuan, Hui and Saif, AFM and Liu, Gaowen and Kompella, Ramana and Wang, Mengdi and Chen, Tianyi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```
