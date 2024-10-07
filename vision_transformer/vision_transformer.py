import torch
import torch.nn as nn
from skimage import io
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import timm # image model Library (vit_base_patch32_224_in21k)
import os
import cv2  # For image processing
from PIL import Image

# Image path and model settings
img_path = "0.jpg" # Path to the input image
image_size = 1024 # Set the image size

preprocess_img_folder = "outputs/preprocess_img"
model_path = "vit_base_patch16_224_in21k" # Vision Transformer vit_base_patch16_224_in21k
attention_map_folder = "outputs/attention_map" # Transformer attention map

attention_map_overlay_folder = "outputs/attention_map_overlay"

# Create directories if they do not exist
os.makedirs(attention_map_folder, exist_ok=True)
os.makedirs(preprocess_img_folder, exist_ok=True)
os.makedirs(attention_map_overlay_folder, exist_ok=True)

# Function to load and preprocess the image
def get_image(im_path, shape=image_size):
    img = io.imread(im_path)[:,:,:3] # HxWxC: Load image and use only the first 3 channels
    org_img = img.copy()

    # Normalize the image
    img = A.Normalize()(image=img)["image"] # HxWxC 이미지 픽셀 값 -> Normalize the image using the mean and std
    norm_img = img.copy() # Copy the normalized image

    # Convert image to CxHxW format
    img = np.transpose(img,(2,0,1)) # CxHxW Albumentations의 Normalize를 사용하여 이미지 정규화 (평균 및 표준편차 사용)
    img = np.expand_dims(img,0) # B=1, Add batch dimension (BxCxHxW)

    img = torch.tensor(img) # Convert NumPy array to PyTorch tensor
    img = nn.Upsample((shape,shape),mode='bilinear')(img) # Upsample using bilinear interpolation
 
    return img, norm_img, org_img # Return upsampled image, normalized image, and original image

# Load and preprocess the image
img, norm_img, org_img = get_image(img_path)
# Set up a figure for visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5)) # Create a subplot with 1 row and 3 columns

# Display the original image
axs[0].imshow(org_img)
axs[0].set_title("Original Image")
axs[0].axis('off')

# Display the normalized image
axs[1].imshow(norm_img)
axs[1].set_title("Normalized Image")
axs[1].axis('off')

# Display the upsampled image
upsampled_img = img[0].permute(1, 2, 0).numpy()
axs[2].imshow(upsampled_img)
axs[2].set_title("Upsampled Image")
axs[2].axis('off')

# Save the entire figure as a single image
preprocess_img_output_path = os.path.join(preprocess_img_folder, "preprocessing_images.png")
plt.savefig(preprocess_img_output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
plt.close()

# Load the Vision Transformer model
model = timm.create_model(model_path,pretrained=True,
                          img_size=image_size,
                          dynamic_img_size=True)

# Move the model to GPU
model = model.cuda()

# Dictionary to store output values
outputs = {}

# Define a hook function to save output values
def get_outputs(name:str):
    def hook(model, input, output):
        outputs[name] = output.detach()
    
    return hook

# Register hooks to specific layers of the model
model.blocks[-1].attn.q_norm.register_forward_hook(get_outputs('Q')) # Register hook for query
model.blocks[-1].attn.k_norm.register_forward_hook(get_outputs('K')) # Register hook for key
model.blocks[-1].register_forward_hook(get_outputs('features')) # Register hook for features

# Pass the image to the model and perform forward pass
model(img.cuda())

# Save the scale value
scale = model.blocks[-1].attn.scale

# Compute attention map and apply softmax
outputs["attention"] = (outputs["Q"]@ outputs["K"].transpose(-2, -1))
outputs["attention"] = outputs["attention"].softmax(dim=1)

# Print the shapes of Q and K
print("Q shape:", outputs["Q"].shape)
print("K shape:", outputs["K"].shape)

# Print the shape of the attention map
print(outputs["attention"].shape)

# Set variables related to the attention map
b, num_heads, num_patches_1, num_patches_2 = outputs["attention"].shape
# Calculate map size
map_size = int(np.sqrt(num_patches_1-1))

attention_maps = []  # List to store attention maps for selection

# Visualize and save attention maps for each attention head
for attention_head in range(num_heads):
    attention_map = outputs["attention"][:,attention_head,0,1:] # Extract 1x4096 attention
    attention_map = attention_map.view(1,1,map_size,map_size) # Convert attention map to 2D

    attention_map = nn.Upsample(size=(image_size, image_size))(attention_map) # Upsample the attention map
    attention_map = attention_map[0,0,:,:].detach().cpu().numpy() # Convert to NumPy array
    
    attention_maps.append(attention_map)  # Save the attention map

    # Save the first attention map as the best attention map for overlaying
    if attention_head == 8:
        best_attention_map = attention_map


    # Visualize the attention map
    plt.imshow(attention_map, cmap='viridis')
    plt.axis('off')
    plt.title(f"Attention Head {attention_head + 1}")

    # Save the attention map as an image file
    attention_map_output_path = os.path.join(attention_map_folder, f"attention_head_{attention_head + 1}.png")
    plt.savefig(attention_map_output_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
print(f"Save the attention map")

# Ask the user to select the best attention map
chosen_attention_map = int(input("Please select the attention map number for overlay: ")) - 1

# Set the best attention map based on the user's choice
best_attention_map = attention_maps[chosen_attention_map]

# Normalize the best attention map and resize it to match the original image size
resized_attention_map = cv2.resize(best_attention_map, (org_img.shape[1], org_img.shape[0]))
resized_attention_map = (resized_attention_map - np.min(resized_attention_map)) / \
                             (np.max(resized_attention_map) - np.min(resized_attention_map))

# Convert the attention map to heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * resized_attention_map), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Overlay the attention map on the original image
overlay_img = Image.blend(Image.fromarray(org_img).convert('RGBA'), Image.fromarray(heatmap).convert('RGBA'), alpha=0.5)

# Save the overlaid image
overlay_output_path = os.path.join(attention_map_overlay_folder, "overlay_attention_map.png")
overlay_img.save(overlay_output_path)

# Display the overlaid image
plt.figure(figsize=(10, 5))
plt.imshow(overlay_img)
plt.axis('off')
plt.title('Attention Map Overlay')
plt.show()