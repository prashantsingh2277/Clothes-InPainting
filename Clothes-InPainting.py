### Install Libraries
!pip install -q regex tqdm
!pip install -q diffusers transformers accelerate scipy 
!pip install -q -U xformers
!pip install -q opencv-python


### Import libraries
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

import PIL, cv2
from PIL import Image

from io import BytesIO
from IPython.display import display
import base64, json, requests
from matplotlib import pyplot as plt

import numpy as np
import copy

from numpy import asarray


### Import Clip and ClipSeg repositories

# Install Clip from OpenAI ## MIT Licence
!pip install -q git+https://github.com/openai/CLIP.git 

# Install ClipSeg Repo ## MIT Licence
!git clone https://github.com/timojl/clipseg  

# Important: Move into the ClipSeg folder
%cd clipseg

### Get the weights for the ClipSeg model
### These weights don't have MIT licence, they can only be used for research, not for commercial purposes
! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
! unzip -d weights -j weights.zip

# Import Clip and ClipSeg model
import clip
from models.clipseg import CLIPDensePredT

# load clipseg model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# non-strict mode: decoder weights only (no CLIP weights)
model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);

### Import Stable Diffusion model
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

model_dir="stabilityai/stable-diffusion-2-inpainting"

### The scheduler determine the algorithm used to produce new samples during the denoising process
scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, 
                                                   scheduler=scheduler,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

### if you receive a triton error message, that's normal on windows systems


### Example image from unsplash.com
### Photo by Lac McGregor, Canada
### Free to use under the Unsplash License
### Link: https://unsplash.com/photos/AsJirOOLN_s

### IMPORTANT: Upload the unsplash image inside the clipseg folder

target_width, target_height = 512,512
source_image = Image.open('mix909-AsJirOOLN_s-unsplash.jpg')

width, height = source_image.size
print(f"Source image size: {source_image.size}")

source_image = source_image.crop((0, height-width , width , height))  # box=(left, upper, right, lower)
source_image = source_image.resize((target_width, target_height), Image.LANCZOS )
print(f"Target image size: {source_image.size}")

### Setup transformations to be applied to the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tensor_image = transform(source_image).unsqueeze(0)

display(source_image)


#### Create masks for the parts of the clothes to be identified

prompts = ['a hat', 'a dark skirt', 'shoes', 'a white shirt']

# Use ClipSeg to identify elements in picture
with torch.no_grad():
    preds = model(tensor_image.repeat(len(prompts),1,1,1), prompts)[0]

#### Create masks for the parts of the clothes to be identified

prompts = ['a hat', 'a dark skirt', 'shoes', 'a white shirt']

# Use ClipSeg to identify elements in picture
with torch.no_grad():
    preds = model(tensor_image.repeat(len(prompts),1,1,1), prompts)[0]


# Decide which mask you want to do inpainting with. In this case we pick the skirt which is mask number 1
mask_number = 1

# Normalize mask values by computing the area under Gaussan probability density function, calculating the cumulative distribution with ndtr
processed_mask = torch.special.ndtr(preds[mask_number][0])

stable_diffusion_mask = transforms.ToPILImage()(processed_mask)
display(stable_diffusion_mask)


### Setup transformation prompts
num_images_per_prompt = 4
inpainting_prompts = ["a skirt full of text",  "blue flowers", "white flowers", "a zebra skirt"]

generator = torch.Generator(device="cuda").manual_seed(77) # 155, 77, 

### Run Stable Difussion pipeline in inpainting mode
encoded_images = []
for i in range(num_images_per_prompt):
        image = pipe(prompt=inpainting_prompts[i], guidance_scale=7.5, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]
        encoded_images.append(image)
        
        
        
        
def create_image_plus_masks_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)  # Create a copy of the names list to avoid modifying the external variable
    images = copy.copy(images)  # Create a copy of the images list to avoid modifying the external variable

    # Check if images is a tensor
    if torch.is_tensor(images):
        # Check if the number of tensor images and names is equal
        assert images.size(0) == len(names), "Number of images and names should be equal"

        # Check if there are enough images for the specified grid size
        assert images.size(0) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

        # Convert the original image to a NumPy array
        original_image_np = np.array(original_image)

        # Normalize tensor images, convert them to 3-channel format, and blend them with the original image
        images = [
            cv2.add(
                original_image_np,
                cv2.applyColorMap((torch.sigmoid(img).squeeze(0).numpy() * 255).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,[2,1,0]] * (torch.sigmoid(img).squeeze(0).numpy() > 0.5).astype(np.uint8)[:, :, np.newaxis]
            )
            for img in images
        ]

        # Convert the blended images back to PIL format
        images = [to_pil_image(img) for img in images]
    else:
        # Check if the number of PIL images and names is equal
        assert len(images) == len(names), "Number of images and names should be equal"

    # Check if there are enough images for the specified grid size
    assert len(images) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

    # Add the original image to the beginning of the images list
    images.insert(0, original_image)

    # Add an empty name for the original image to the beginning of the names list
    names.insert(0, '')

    # Create a figure with specified rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15))

    # Iterate through the images and names
    for idx, (img, name) in enumerate(zip(images, names)):
        # Calculate the row and column index for the current image
        row, col = divmod(idx, columns)

        # Add the image to the grid
        axes[row, col].imshow(img)

        # Set the title (name) for the subplot
        axes[row, col].set_title(name)

        # Turn off axes for the subplot
        axes[row, col].axis('off')

    # Iterate through unused grid cells
    for idx in range(len(images), rows * columns):
        # Calculate the row and column index for the current cell
        row, col = divmod(idx, columns)

        # Turn off axes for the unused grid cell
        axes[row, col].axis('off')

    # Adjust the subplot positions to eliminate overlaps
    plt.tight_layout()

    # Display the grid of images with their names
    plt.show()




def create_image_plus_masks_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)  # Create a copy of the names list to avoid modifying the external variable
    images = copy.copy(images)  # Create a copy of the images list to avoid modifying the external variable

    # Check if images is a tensor
    if torch.is_tensor(images):
        # Check if the number of tensor images and names is equal
        assert images.size(0) == len(names), "Number of images and names should be equal"

        # Check if there are enough images for the specified grid size
        assert images.size(0) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

        # Convert the original image to a NumPy array
        original_image_np = np.array(original_image)

        # Normalize tensor images, convert them to 3-channel format, and blend them with the original image
        images = [
            np.where(
                (torch.sigmoid(img).squeeze(0).numpy() > 0.5)[:, :, np.newaxis],
                cv2.applyColorMap((torch.sigmoid(img).squeeze(0).numpy() * 255).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,[2,1,0]],
                original_image_np
            )
            for img in images
        ]

        # Convert the blended images back to PIL format
        images = [to_pil_image(img) for img in images]
    else:
        # Check if the number of PIL images and names is equal
        assert len(images) == len(names), "Number of images and names should be equal"

    # Check if there are enough images for the specified grid size
    assert len(images) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

    # Add the original image to the beginning of the images list
    images.insert(0, original_image)

    # Add an empty name for the original image to the beginning of the names list
    names.insert(0, '')

    # Create a figure with specified rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15))

    # Iterate through the images and names
    for idx, (img, name) in enumerate(zip(images, names)):
        # Calculate the row and column index for the current image
        row, col = divmod(idx, columns)

        # Add the image to the grid
        axes[row, col].imshow(img)

        # Set the title (name) for the subplot
        axes[row, col].set_title(name)

        # Turn off axes for the subplot
        axes[row, col].axis('off')

    # Iterate through unused grid cells
    for idx in range(len(images), rows * columns):
        # Calculate the row and column index for the current cell
        row, col = divmod(idx, columns)

        # Turn off axes for the unused grid cell
        axes[row, col].axis('off')

    # Adjust the subplot positions to eliminate overlaps
    plt.tight_layout()

    # Display the grid of images with their names
    plt.show()



def create_image_plus_masks_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)  # Create a copy of the names list to avoid modifying the external variable
    images = copy.copy(images)  # Create a copy of the images list to avoid modifying the external variable

    # Check if images is a tensor
    if torch.is_tensor(images):
        # Check if the number of tensor images and names is equal
        assert images.size(0) == len(names), "Number of images and names should be equal"

        # Check if there are enough images for the specified grid size
        assert images.size(0) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

        # Convert the original image to a NumPy array
        original_image_np = np.array(original_image)

        # Create a solid yellow color mask with the same size as the original image
        yellow_mask = np.zeros_like(original_image_np)
        yellow_mask[..., 0] = 255  # Red channel
        yellow_mask[..., 1] = 255  # Green channel

        # Normalize tensor images, create a binary mask, and blend them with the original image
        images = [
            np.where(
                (torch.sigmoid(img).squeeze(0).numpy() > 0.5)[:, :, np.newaxis],
                yellow_mask,
                original_image_np
            )
            for img in images
        ]

        # Convert the blended images back to PIL format
        images = [to_pil_image(img) for img in images]
    else:
        # Check if the number of PIL images and names is equal
        assert len(images) == len(names), "Number of images and names should be equal"

    # Check if there are enough images for the specified grid size
    assert len(images) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

    # Add the original image to the beginning of the images list
    images.insert(0, original_image)

    # Add an empty name for the original image to the beginning of the names list
    names.insert(0, '')

    # Create a figure with specified rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15))

    # Iterate through the images and names
    for idx, (img, name) in enumerate(zip(images, names)):
        # Calculate the row and column index for the current image
        row, col = divmod(idx, columns)

        # Add the image to the grid
        axes[row, col].imshow(img)

        # Set the title (name) for the subplot
        axes[row, col].set_title(name)

        # Turn off axes for the subplot
        axes[row, col].axis('off')

    # Iterate through unused grid cells
    for idx in range(len(images), rows * columns):
        # Calculate the row and column index for the current cell
        row, col = divmod(idx, columns)

        # Turn off axes for the unused grid cell
        axes[row, col].axis('off')

    # Adjust the subplot positions to eliminate overlaps
    plt.tight_layout()

    # Display the grid of images with their names
    plt.show()





























