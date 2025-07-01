import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils_single import ExponentialMovingAverage
import os
import math
import argparse
import csv
import time

import sys
from evaluation import FIDCalculator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import json

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):   
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'forward steps of DDIM',default=1000)
    parser.add_argument('--backward_steps',type = int,help = 'sampling steps of DDIM',default=100)
    parser.add_argument('--start',type = float,help = 'start of noise scheduler',default=0)
    parser.add_argument('--end',type = float,help = 'end of noise scheduler',default=1)
    parser.add_argument('--epsilon',type = float,help = 'epsilon for noise scheduler',default=0.008)
    parser.add_argument('--tau',type = float,help = 'tau for noise scheduler',default=1)
    parser.add_argument('--tau_type',type = str,help = 'tau_type for DDIM skipping',default='linear')
    parser.add_argument('--scheduler_type',type = str,help = 'parameterization of noise scheduler',default='cosine')
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use.")

    args = parser.parse_args()
    

    return args

def save_args(args, filepath):
    """
    Save the parsed arguments to a JSON file.

    Args:
        args (argparse.Namespace): Parsed arguments.
        filepath (str): Path to the file where arguments will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                         back_steps=args.backward_steps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4],
                scheduler_type=args.scheduler_type,
                start=args.start,
                end=args.end,
                epsilon=args.epsilon,
                tau=args.tau).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)


    # using the differntiable FID (with conti)
    FID = FIDCalculator(device)

    # using the torchmetrics FID
    fid = FrechetInceptionDistance(normalize=True,reset_real_features=False).to(device)
    
    inception = InceptionScore(normalize=True).to(device)

   

    # initializing FID with real features
    for step, (image, _)in enumerate(test_dataloader):

        # # for trial
        # if step>2:
        #     continue

        image = image.to(device)


        # differntiable FID needs input scale at [-1,1]
        FID.update_real(image)

        # do this as it is MNIST, no need for CIFAR-10
        image = image.repeat(1, 3, 1, 1)  # Repeat the channels
        image = (image + 1) / 2  # Scale values from [-1, 1] to [0, 1]
        fid.update(image, real=True)


    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean') 

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        
        
    scheduler_type = getattr(args, 'scheduler_type', 'default_value')
        
        
    output_file = "results/DDIM/{}/tau_{:.3f}_start_{:.3f}_end_{:.3f}_s_{:.3f}/output/scores.csv".format(scheduler_type,args.tau,args.start,args.end,args.epsilon)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    header = ["Epoch", "Upper Loss", "FID Score", "IS Score"]
    file_exists = os.path.isfile(output_file)
    
    
    # Save arguments to a file
    directory = "results/DDIM/{}/tau_{:.3f}_start_{:.3f}_end_{:.3f}_s_{:.3f}/output/".format(scheduler_type,args.tau,args.start,args.end,args.epsilon)
    
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(directory, "args.json")  # Path for the JSON file
    save_args(args, file_path)
    print(f"Arguments saved to {file_path}")
    
    start_time = time.time()
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header only if the file is new
        writer.writerow(header)
        

        global_steps=0
        for i in range(args.epochs):
            model.train()
            for j,(image,target) in enumerate(train_dataloader):
                noise=torch.randn_like(image).to(device)
                image=image.to(device)
                pred=model(image,noise)
                loss=loss_fn(pred,noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if global_steps%args.model_ema_steps==0:
                    model_ema.update_parameters(model)
                global_steps+=1
                if j%args.log_freq==0:
                    print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                        loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
            ckpt={"model":model.state_dict(),
                    "model_ema":model_ema.state_dict()}
    
            # os.makedirs("results",exist_ok=True)
            os.makedirs("results/DDIM/{}/tau_{:.3f}_start_{:.3f}_end_{:.3f}_s_{:.3f}",exist_ok=True)
            torch.save(ckpt,"results/DDIM/{}/tau_{:.3f}_start_{:.3f}_end_{:.3f}_s_{:.3f}/steps_{:0>8}.pt".format(scheduler_type,args.tau,args.start,args.end,args.epsilon,global_steps))
            # torch.save(ckpt,"results/DDIM/steps_{:0>8}_tau_{}_start_{}_end_{}_s_{}.pt".format(global_steps,args.tau,args.start,args.end,args.epsilon))
    
            model_ema.eval()
            samples=model_ema.module.sampling_DDIM(args.n_samples,device=device,tau_type = args.tau_type)
            save_image(samples,"results/DDIM/{}/tau_{:.3f}_start_{:.3f}_end_{:.3f}_s_{:.3f}/steps_{:0>8}.png".format(scheduler_type,args.tau,args.start,args.end,args.epsilon,global_steps),nrow=int(math.sqrt(args.n_samples)))
    
            #clear the previous fake feature first
            FID.clear_fake()
            fid.reset()
            inception.reset()
            
            # differntiable FID needs input scale at [-1,1]
            FID.update_fake(samples)
            upper_loss = FID.compute()
            
    
            # do this as it is MNIST, no need for CIFAR-10
            samples = samples.repeat(1, 3, 1, 1)  # Repeat the channels
            samples= (samples + 1) / 2  # Scale values from [-1, 1] to [0, 1]
            fid.update(samples, real=False)
    
            fid_score = fid.compute()
            
            inception.update(samples)
            
            inception_score, std_inception_score = inception.compute()
    
            print("Upper loss (FID differentiable): ", upper_loss, "torch FID: ", fid_score, "IS: ", inception_score)
    
            # Write the iteration and scores to the CSV
            writer.writerow([i, upper_loss, fid_score, inception_score])
            # f.flush()
    
        end_time = time.time()
        execution_time = end_time - start_time
        print("Time: ", execution_time)
    
        writer = csv.writer(f)
        writer.writerow(["Time", execution_time, "", ""])

if __name__=="__main__":
    args=parse_args()
    main(args)
