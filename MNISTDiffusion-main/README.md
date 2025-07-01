# Noise optimizatino for training MNIST Diffusion
![60 epochs training from scratch](assets/demo.gif "60 epochs training from scratch")

## Training
Install packages
```bash
pip install -r requirements.txt
```
Start default setting training 
```bash
python train_mnist.py
python train_mnist_bayesian.py 
python train_mnist_bilevel_ZO.py --inner_loop 10 --inner_loop_z 1 --gamma 1 --gamma_end 1 --lr_beta 0.05 0.05 1 0.05 --initial_epoch 2 --epochs 3
```
Feel free to tuning training parameters, type `python train_mnist.py -h` to get help message of arguments.

For train_mnist.py, tune hyperparameter tau first and then start, end, and epsilon. 
For train_mnist_bilevel.py, tune gamma and learning rates for beta, and initial epoch (without learning hyperparmater) and epochs. 

## Reference
A neat blog explains how diffusion model works: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

A pytorch version of DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch

