## Reward Fine-Tuning in Diffusion Models (Section 3.1)

This directory implements the inference-only bilevel approach for reward fine-tuning in diffusion models as described in Section 3.1 of **"A First-order Generative Bilevel Optimization Framework for Diffusion Models"**.

### 📦 Requirements

* Python >= 3.8
* PyTorch >= 1.10
* [diffusers](https://github.com/huggingface/diffusers)
* [transformers](https://github.com/huggingface/transformers)
* [torchvision](https://github.com/pytorch/vision)
* [tqdm](https://github.com/tqdm/tqdm)
* CLIP model (from OpenAI)

Install the required packages:

```bash
pip install -r requirements.txt
```

### 🗂️ Directory Structure

```
scripts/
├── main.py      # Training for bilevel reward fine-tuning

├── reward_config.yaml      # Default hyperparameters for Section 3.1

├── scorer_bi.py     # Reward functions

└── gradguided_sdpipeline.py/        # Checkpoints and generated samples
```

### 🚀 Usage

Run the fine-tuning procedure:

```bash
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

where:

* `--num_samples` is the number of Monte Carlo samples from the pre-trained distribution.
* `--lambda_init` initializes the entropy strength $\lambda$ for the lower-level problem.
* `--gamma` is the penalty constant $\gamma$ controlling approximation accuracy.
* `--steps` is the number of bilevel update iterations for $\lambda$.

### 🔍 Algorithm Details

Section 3.1 formulates reward fine-tuning as a bilevel problem:

<pre> Minimize over λ ∈ ℝ₊ and p ∈ 𝒮(λ): f(λ, p) := –E_{u ~ p}[r₁(u)] Subject to: 𝒮(λ) = argmin_{p′ ∈ 𝒫} { –E_{u ~ p′}[r₂(u)] + λ · KL(p′ || p_data) } </pre>

* **Lower level** adjusts sampling via guided backward SDE (Algorithm 5) with entropy regularized reward $r_2$.
* **Upper level** updates $\lambda$ by Monte Carlo estimation of the closed-form gradient (Eq. 13) and projected gradient descent (Algorithm 2).

### 📈 Output

* Learned entropy strength $\lambda^*$ saved to `outputs/reward_finetune/lambda_final.txt`
* Generated images in `outputs/reward_finetune/samples/`
* Training logs and gradient trajectories in `outputs/reward_finetune/logs/`

### 📝 Citation

If you use this code, please cite:

```
@inproceedings{xiao2025first,
  title={A First-order Generative Bilevel Optimization Framework for Diffusion Models},
  author={Xiao, Quan and Yuan, Hui and Saif, AFM and Liu, Gaowen and Kompella, Ramana and Wang, Mengdi and Chen, Tianyi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

