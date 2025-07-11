import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import os
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from evaluation_fid import FIDCalculator
from torchmetrics.image.fid import FrechetInceptionDistance

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RCGDMScorer(torch.nn.Module):
    ### Reward model borrowed from RCGDM
    def __init__(self):
        super().__init__()
        
        ## load the synthetic reward model which is based on a ResNet-18 model
        reward_model_path = "reward_model.pth"
        
        if os.path.exists(reward_model_path):
            self.reward_model = torch.load(reward_model_path)
        
        else:
            model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

            # Replace the final layer with a linear layer of scalar output
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            torch.save(model, reward_model_path)
        
            self.reward_model = torch.load(reward_model_path) 
        
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def __call__(self, images, prompts=[]):
        ## input: generated images
        target_size = 224
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
        
        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]
        ## image transform
        im_pix = torchvision.transforms.Resize(target_size)(images)
        im_pix = normalize(im_pix).to(images.dtype)
           
        return self.reward_model(im_pix)


class PenaltyScorer(torch.nn.Module):
    ### Reward model borrowed from RCGDM
    def __init__(self, gemma):
        super().__init__()
        
        ## load the synthetic reward model which is based on a ResNet-18 model
        reward_model_path = "reward_model.pth"
        self.reward_model = torch.load(reward_model_path) 
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()
        self.gemma = gemma
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def __call__(self, images, prompts):
        ## input: generated images
        target_size = 224
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
        
        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]
        ## image transform
        im_pix = torchvision.transforms.Resize(target_size)(images)
        im_pix = normalize(im_pix).to(images.dtype)

        ## calculate clip score
        clip = self.clip_score_fn(images, prompts).detach()
        
        return self.reward_model(im_pix)+ self.gemma * round(float(clip), 4)
           


class Weighted_FID_Scorer(torch.nn.Module):
    """Weighted FID Scorer with Reward Model."""
    
    def __init__(self, device):
        super().__init__()
        reward_model_path = "reward_model.pth"
        
        # Load the reward model
        self.reward_model = torch.load(reward_model_path, map_location=device)
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()
        self.device = device
        
        # Initialize FIDCalculator and torchmetrics FID
        self.FID = FIDCalculator(device)
        self.fid = FrechetInceptionDistance(normalize=True, reset_real_features=False).to(device)

    # def img_process(self, images):

    #     if not isinstance(images, torch.Tensor):
    #         # Convert PIL Image to Tensor
    #         images = transforms.ToTensor()(images).to(self.device)

    #     target_size = 224
    #     normalize = transforms.Normalize(
    #         mean=[0.48145466, 0.4578275, 0.40821073],
    #         std=[0.26862954, 0.26130258, 0.27577711]
    #     )
        
    #     # Convert from [-1, 1] to [0, 1] if necessary
    #     images = ((images + 1.0) / 2.0).clamp(0, 1)
        
    #     # Apply resizing and normalization
    #     transform = transforms.Compose([
    #         transforms.Resize(target_size),
    #         normalize
    #     ])
        
    #     return transform(images).to(self.device)

    def img_process_reward(self, images):

        if isinstance(images, list):  # If input is a list of images (e.g., PIL)
            # Convert each image to a tensor and stack them into a batch
            images = torch.stack([transforms.ToTensor()(img) for img in images]).to(self.device)
    
        target_size = 224
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]

        return images.to(images.dtype).to(self.device)

    def img_process_fid(self, images):
        
        if isinstance(images, list):  # If input is a list of images (e.g., PIL)
            # Convert each image to a tensor and stack them into a batch
            images = torch.stack([transforms.ToTensor()(img) for img in images]).to(self.device)

        # Ensure the tensor is float before applying transforms
        images = images.to(torch.float32)

        # Normalize and resize
        target_size = 224
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        # Apply the transform to each image in the batch
        images = torch.stack([transform(img) for img in images])

        return images.to(images.dtype).to(self.device)



    def forward(self, pretrained_images, fake_image, generated_images):
        """
        Compute the weighted FID score and reward model score.
        """
        # Process images
        generated_images_p = self.img_process_reward(generated_images)
        pretrained_images_p = self.img_process_fid(pretrained_images)
        fake_image = self.img_process_fid(fake_image)

        # Ensure tensors are 4D: Add batch dimension if necessary
        # if pretrained_images_p.ndim == 3:  # Single image: [C, H, W]
        #     pretrained_images_p = pretrained_images_p.unsqueeze(0)
        # if generated_images_p.ndim == 3:  # Single image: [C, H, W]
        #     generated_images_p = generated_images_p.unsqueeze(0)

        # print(pretrained_images_p.shape)
        # print(generated_images_p.shape)


        # Reset FIDCalculator
        self.FID.clear_fake()
        self.fid.reset()
        
        
        self.FID.update_real(pretrained_images_p)
        self.fid.update(pretrained_images_p, real=True)

        scalar = torch.tensor([1.0], device=self.device, requires_grad=True)

        self.FID.update_fake(fake_image * scalar)
        self.fid.update(fake_image, real=False)

        # Compute FID loss
        FID_loss = self.FID.compute()

        # Reset and use torchmetrics FID
        
        
        
        fid_score = self.fid.compute()
        

        return self.reward_model(generated_images_p) - .5*round(float(FID_loss), 4), self.reward_model(generated_images_p), round(float(fid_score), 4)




# class Weighted_Clip_Scorer(torch.nn.Module):
#     """Weighted FID Scorer with Reward Model."""
    
#     def __init__(self, device):
#         super().__init__()
#         reward_model_path = "reward_model.pth"
        
#         # Load the reward model
#         self.reward_model = torch.load(reward_model_path, map_location=device)
#         self.reward_model.requires_grad_(False)
#         self.reward_model.eval()
#         self.device = device
        
#     def img_process(self, images):
    
#         if isinstance(images, list):  # If input is a list of images (e.g., PIL)
#             # Convert each image to a tensor and stack them into a batch
#             images = torch.stack([transforms.ToTensor()(img) for img in images]).to(self.device)

#         elif isinstance(images, torch.Tensor) and images.ndim == 3:  # Single tensor image: [C, H, W]
#             images = images.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

#         # Normalize and resize
#         target_size = 224
#         transform = transforms.Compose([
#             transforms.Resize(target_size),
#             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                 std=[0.26862954, 0.26130258, 0.27577711])
#         ])
#         # Apply the transform to each image in the batch
#         images = torch.stack([transform(img) for img in images])

#         images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]

#         return images.to(images.dtype).to(self.device)


#     def forward(self, generated_images, prompts):
#         """
#         Compute the weighted CLIP score and reward model score.
#         """
#         # Process images
#         generated_images_p = self.img_process(generated_images)

#         print(generated_images_p.shape)
#         print(prompts*generated_images_p[0])
        

#         clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

#         def calculate_clip_score(image,prompts):
#             print(image)
#             images_init = (images).astype("uint8")
#             clip_score = clip_score_fn(torch.from_numpy(images_init), prompts)
#             return round(float(clip_score), 4)


#         sd_clip_score = calculate_clip_score(generated_images_p, prompts*generated_images_p[0])


#         return self.reward_model(generated_images_p) + round(float(sd_clip_score), 4), self.reward_model(generated_images_p), round(float(sd_clip_score), 4)


class Weighted_Clip_Scorer(torch.nn.Module):
    """Weighted CLIP Scorer with Reward Model."""
    
    def __init__(self, device):
        super().__init__()
        reward_model_path = "reward_model.pth"
        
        # Load the reward model
        self.reward_model = torch.load(reward_model_path, map_location=device)
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()
        self.device = device
        

    def img_process(self, images):
        """
        Preprocess the input images:
        - Normalize for the reward model
        """
        # if isinstance(images, list):  # If input is a list of images (e.g., PIL)
        #     # Convert each image to a tensor and stack them into a batch
        #     images = torch.stack([transforms.ToTensor()(img) for img in images]).to(self.device)
        # elif isinstance(images, torch.Tensor) and images.ndim == 3:  # Single tensor image: [C, H, W]
        #     images = images.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

        # Normalize and resize
        target_size = 224
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        # Apply the transform to each image in the batch
        images = torch.stack([transform(img) for img in images])

        images = ((images + 1.0) / 2.0).clamp(0, 1) # convert from [-1, 1] to [0, 1]

        return images.to(images.dtype).to(self.device)


    def forward(self, generated_images, clip_images, prompts):
        """
        Compute the weighted CLIP score and reward model score.
        """

        # Process images for reward score
        processed_images = self.img_process(generated_images)

        # Compute reward model score
        reward_scores = self.reward_model(processed_images)

        # Additional processing for CLIP score

        # Compute CLIP score
        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        sd_clip_scores = clip_score_fn(clip_images, prompts)

        print(sd_clip_scores)

        # Combine scores
        combined_score = reward_scores + sd_clip_scores

        return combined_score, reward_scores, sd_clip_scores    