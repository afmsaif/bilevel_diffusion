# MNIST Diffusion
![60 epochs training from scratch](assets/demo.gif "60 epochs training from scratch")

## Training
Install packages
```bash
pip install -r requirements.txt
```
Start default setting training 
```bash
python train_mnist_grid_search.py 
python train_mnist_bilevel.py --inner_loop 1 --gamma 1e-3 --lr 0.01 --lr_z 0.01 --lr_beta 0.01 --epochs 1 --truncation 2
```
Feel free to tuning training parameters, type `python train_mnist.py -h` to get help message of arguments.

For train_mnist_grid_search.py, tune hyperparameter tau first and then start, end, and epsilon. Implement Bayesian search for tuning tau. 
For train_mnist_bilevel.py, tune gamma and learning rates. 

## Reference
A neat blog explains how diffusion model works(must read!): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

A pytorch version of DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch

