import os
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import random
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the prompts
prompts = [
    # "A fox",
    # "A horse",
    # "A rabbit",
    "horse standing on lush green grass"
    ]

lambda_ = [0.01,0.1,1,10,100]

for _, lm in enumerate(lambda_):
    for i in range(0,8):

        folder_paths = [
        # f"/home/saif/ASR/diffusion/GGDMOptim/image_generation/results_fox/target10.0guidance1.0lambda{lm}seed5totalbs12_fox/fake_images/fox_step_{i}", 
        # f"/home/saif/ASR/diffusion/GGDMOptim/image_generation/results_horse/target10.0guidance1.0lambda{lm}seed5totalbs12_horse/fake_images/horse_step_{i}", 
        # f"/home/saif/ASR/diffusion/GGDMOptim/image_generation/results_rabbit/target10.0guidance1.0lambda{lm}seed5totalbs12_rabbit/fake_images/rabbit_step_{i}"
        f"/home/saif/ASR/diffusion/GGDMOptim/image_generation/results_hourse_stand/target10.0guidance1.0lambda{lm}seed5totalbs12_horse standing on lush green grass/fake_images/horse_standing_on_lush_green_grass_step_{i}"
        ]

        # Initialize an empty list to hold the images
        images = []

        for folder in folder_paths:
            # List all files in the folder
            files = os.listdir(folder)
            
            # Filter only image files (optional, depending on your folder content)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Randomly choose an image
            chosen_image = random.choice(image_files)
            
            # Full path to the chosen image
            image_path = os.path.join(folder, chosen_image)
            
            # Load the image
            img = Image.open(image_path).convert("RGB")  # Ensure RGB format
            
            # Convert to NumPy array
            img_array = np.array(img)

            # print(img_array)
            
            # Append to the list
            images.append(img_array)

        images = np.stack(images)

        # print(images.shape)
        # print(f"Images saved to the folder: {output_folder}")

        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

        def calculate_clip_score(image,prompts):
            images_init = (images).astype("uint8")
            clip_score = clip_score_fn(torch.from_numpy(images_init).permute(0, 3, 1, 2), prompts).detach()
            return round(float(clip_score), 4)


        sd_clip_score = calculate_clip_score(images, prompts)
        print(f"Lambda: {lm}, Step: {i}, CLIP Score: {sd_clip_score}")