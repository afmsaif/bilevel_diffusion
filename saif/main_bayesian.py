import os
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

from gradguided_sdpipeline import GradGuidedSDPipeline
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import wandb
import argparse
from scorer import RCGDMScorer
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--target", type=float, default=0., help="target reward value y")
    parser.add_argument("--guidance", type=float, default=1., help="guidance strength beta(t)")
    parser.add_argument("--opt_steps", type=int, default=100, help="number of optimization steps")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="text prompt for image generation")
    parser.add_argument("--out_dir", type=str, default="results_fox", help="output directory")
    parser.add_argument("--bs", type=int, default=5, help="batch size for optimization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat_epoch", type=int, default=1, help="repeat the optimization for multiple times")
    parser.add_argument("--image_size", type=int, default=512, help="image size for optimization")
    parser.add_argument("--wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--save_samples", action="store_true", default=True, help="save the generated images")
    args = parser.parse_args()
    return args

def get_grad_eval(ims, reward_model, device='cuda'):    
    ims = ims.to(device)
    ims.requires_grad = True
    rewards = reward_model(ims)
    rewards_sum = rewards.sum()
    rewards_sum.backward()  # get the gradients for each sample in the batch
    grads = ims.grad
    biases = -torch.einsum('bijk,bijk->b', grads, ims) + rewards  # r(x) - <grad, x>
    return grads, biases, rewards

def concat_from_2d_list(lst, size=128):
    img = Image.new('RGB', (size * len(lst[0]), size * len(lst)))
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            img.paste(lst[i][j], (size * j, size * i))
    return img

def run_for_lambda(args, lambda_scale):
    device = args.device if torch.cuda.is_available() else "cpu"
    total_bs = args.bs * args.repeat_epoch
    remark = f'target{args.target}guidance{args.guidance}lambda{lambda_scale}seed{args.seed}totalbs{total_bs}_{args.prompt}'
    out_dir = args.out_dir + '/' + remark
    os.makedirs(out_dir, exist_ok=True)

    print("-" * 50)
    print("target:", args.target)
    print("guidance:", args.guidance)
    print("lambda_scale:", lambda_scale)
    print("opt_steps:", args.opt_steps)
    print("prompt:", args.prompt)
    print("bs:", args.bs)
    print("repeat_epoch:", args.repeat_epoch)
    print("seed:", args.seed)
    print("device:", device)
    print("out_dir:", out_dir)
    print("-" * 50)

    if args.wandb:
        wandb.init(
            project="gradient_guided_sd", 
            name=remark,
            config={
                'target': args.target,
                'guidance': args.guidance,
                'lambda_scale': lambda_scale,
                'seed': args.seed
            }
        )

    image_size_kwargs = {"height": args.image_size, "width": args.image_size}

    ### Set up prompts
    prompts = [args.prompt] * args.repeat_epoch

    ### Load Reward Model
    reward_model = RCGDMScorer().to(device)

    ### Pretrained Stable Diffusion Model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to(device)
    pretrain_gen_start = time.time()
    pretrain_images = pipe([prompts[0]] * args.bs, **image_size_kwargs).images
    pretrain_gen_end = time.time()
    pretrain_gen_time = pretrain_gen_end - pretrain_gen_start

    ### Gradient Guided Optimization
    targets = [args.target] * args.opt_steps
    guidances = [args.guidance / lambda_scale] * args.opt_steps
    sd_model = GradGuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
    sd_model.to(device)
    image_rewards = np.zeros([args.opt_steps, args.repeat_epoch, args.bs])

    torch.manual_seed(args.seed)
    latent_size = int(args.image_size) // sd_model.vae_scale_factor
    shape = (args.repeat_epoch, args.bs, 4, latent_size, latent_size)
    init_latents = torch.randn(shape, device=device)

    opt_start_time = time.time()
    opt_images_step = defaultdict(list)
    for n in range(args.repeat_epoch):
        sd_model.set_linear_reward_model(is_init=True, batch_size=args.bs, **image_size_kwargs)
        prompt = prompts[n]
        for k in range(args.opt_steps):
            sd_model.set_target(targets[k])
            sd_model.set_guidance(guidances[k])
            init_i = init_latents[n]
            image_, image_eval_ = sd_model(prompt, num_images_per_prompt=args.bs, latents=init_i, **image_size_kwargs)
            image_ = image_.images
            grads, biases, rewards = get_grad_eval(image_eval_, reward_model)
            grads, biases = grads.clone().detach(), biases.clone().detach()
            sd_model.set_linear_reward_model(gradients=grads, biases=biases, **image_size_kwargs)
            rewards = rewards.detach().cpu().numpy()
            if len(rewards.shape) > 1:
                rewards = rewards.squeeze()
            image_rewards[k, n] = rewards
            if args.wandb:
                wandb.log({'step': k, 'reward_mean': rewards.mean(), 'reward_std': rewards.std()})
            if args.save_samples:
                opt_images_step[k].append([img.resize([args.image_size // 4, args.image_size // 4]) for img in image_])

    if args.save_samples:
        os.makedirs(out_dir + "/generated_images", exist_ok=True)
        imgs = concat_from_2d_list([[img.resize([args.image_size // 4, args.image_size // 4]) for img in pretrain_images]], size=args.image_size // 4)
        imgs.save(out_dir + "/generated_images/pretrain.png")
        for k in range(args.opt_steps):
            img = concat_from_2d_list(opt_images_step[k], size=args.image_size // 4)
            img.save(out_dir + f"/generated_images/step_{k}.png")

    if args.save_samples:
        # Create directories for real and fake images
        real_dir = os.path.join(out_dir, "real_images")
        fake_dir = os.path.join(out_dir, "fake_images")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        # Sanitize the prompt for filenames
        sanitized_prompt = args.prompt.replace(" ", "_").lower()

        # Save real (pretrained) images
        for i, img in enumerate(pretrain_images):
            img.save(os.path.join(real_dir, f"{sanitized_prompt}_real_{i}.png"))

        # Save generated (fake) images for each optimization step
        for k in range(args.opt_steps):
            step_dir = os.path.join(fake_dir, f"{sanitized_prompt}_step_{k}")
            os.makedirs(step_dir, exist_ok=True)  # Create sub-folder for each step
            for n, images in enumerate(opt_images_step[k]):
                for j, img in enumerate(images):
                    img.save(os.path.join(step_dir, f"{sanitized_prompt}_fake_step_{k}_repeat_{n}_img_{j}.png"))


    end_time = time.time()
    total_optim_time = end_time - opt_start_time
    total_optim_round = args.repeat_epoch * args.opt_steps


    print(f"------------Time Summary-----------------")
    print(f"Total rounds: {total_optim_round}")
    print(f"The total optimizing time is: {total_optim_time}")
    print(f"Each round:\n optimizing time: {total_optim_time / total_optim_round}")
    print(f"Time for unguided generation with pretrained model (Single Round): {pretrain_gen_time}")

    image_rewards_mean = np.mean(image_rewards, axis=2)
    np.savetxt(out_dir + '/rewards_mean_repeat.csv', image_rewards_mean, delimiter=',')
    image_rewards_std = np.std(image_rewards_mean, axis=1)
    np.savetxt(out_dir + '/rewards_std_in_repeats.csv', image_rewards_std, delimiter=',')

    image_rewards = image_rewards.reshape(args.opt_steps, -1)
    np.savetxt(out_dir + '/rewards.csv', image_rewards, delimiter=',')
    mean_rewards = np.mean(image_rewards, axis=1)                       ######## want to maximize
    std_rewards = np.std(image_rewards, axis=1)
    np.savetxt(out_dir + '/mean_rewards.csv', mean_rewards, delimiter=',')
    np.savetxt(out_dir + '/std_rewards.csv', std_rewards, delimiter=',')

    x = np.arange(args.opt_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_rewards, color='dodgerblue')
    plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color='dodgerblue', alpha=0.2)
    plt.xlabel('Optimization Steps')
    plt.ylabel('Reward')
    plt.savefig(out_dir + '/reward_plot.png')
    plt.close()
    time.sleep(1)

    return mean_rewards

# def main():
#     args = parse_args()
#     # lambda_values = [0.01,0.1,1,10,100]
#     lambda_values = [0.01]
#     for lambda_scale in lambda_values:
#         run_for_lambda(args, lambda_scale)

# if __name__ == "__main__":
#     main()

# define search space  

search_space = [Real(0.01, 100, "log-uniform", name="lambda_scale")]

# Define the objective function
# @use_named_args(search_space)
# def objective(**params):
#     lambda_scale = params["lambda_scale"]
#     args = parse_args()
#     start_time = time.time()
#     mean_rewards = run_for_lambda(args, lambda_scale)
#     end_time = time.time()

#     # Print details for each iteration
#     print(f"Lambda: {lambda_scale:.4f}, Mean Rewards: {mean_rewards:.4f}, Time: {end_time - start_time:.2f} seconds")

#     # Log the time taken for each iteration
#     iteration_times.append(end_time - start_time)
#     lambda_values.append(lambda_scale)

#     # Negate mean_rewards since we are minimizing
#     return -mean_rewards


@use_named_args(search_space)
def objective(**params):
    lambda_scale = params["lambda_scale"]
    args = parse_args()
    start_time = time.time()

    # Run the optimization process for the given lambda_scale
    mean_rewards = run_for_lambda(args, lambda_scale)

    # Ensure mean_rewards is a scalar for printing
    if isinstance(mean_rewards, np.ndarray):
        mean_rewards = mean_rewards.mean()

    end_time = time.time()

    # Log details for this iteration
    iteration_time = end_time - start_time
    print(f"Lambda: {lambda_scale:.4f}, Mean Rewards: {mean_rewards:.4f}, Time: {iteration_time:.2f} seconds")
    
    # Store details for logging and analysis
    lambda_values.append(lambda_scale)
    iteration_times.append(iteration_time)
    # mean_rewards_list.append(mean_rewards)

    # Return negative mean_rewards for minimization
    return -mean_rewards




# List to store lambda values and times for each iteration
lambda_values = []
iteration_times = []

if __name__ == "__main__":
    args = parse_args()
    start_total_time = time.time()

    # Perform Bayesian Optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,  # Number of evaluations
        random_state=42
    )

    # Calculate total time
    total_time = time.time() - start_total_time

    # Print final results
    print("\nOptimization Completed")
    print(f"Best Lambda: {result.x[0]:.4f}")
    print(f"Best Mean Rewards: {-result.fun:.4f}")
    print(f"Total Time: {total_time:.2f} seconds")

    # Print all lambda values and their corresponding times
    print("\nLambda Values and Times:")
    for i, (lambda_val, time_taken) in enumerate(zip(lambda_values, iteration_times)):
        print(f"Iteration {i+1}: Lambda = {lambda_val:.4f}, Time = {time_taken:.2f} seconds")

    # Save lambda values and iteration times
    np.savetxt("lambda_values.csv", lambda_values, delimiter=",")
    np.savetxt("iteration_times.csv", iteration_times, delimiter=",")

    # Plot lambda vs. mean rewards (inverted objective function)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, [-objective(lambda_scale=val) for val in lambda_values], marker="o")
    plt.xlabel("Lambda Scale")
    plt.ylabel("Mean Rewards")
    plt.title("Lambda Scale vs. Mean Rewards")
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("lambda_vs_mean_rewards.png")
    plt.show()