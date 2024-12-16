
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Function to apply random transformations to images and masks
def augment_image_and_mask(image, mask):
    # Resize image and mask to the desired size
    target_size = (5472, 3648)  # Resize to 256x256, modify this as needed
    
    # Check if the images are not empty
    if image is None or mask is None:
        raise ValueError("Image or Mask is empty. Check the file paths.")

    # Resize images and masks
    image_resized = cv2.resize(image, target_size) if image is not None else None
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) if mask is not None else None

    # Random rotation between -20 and +20 degrees
    angle = np.random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((target_size[0] / 2, target_size[1] / 2), angle, 1)
    image_rotated = cv2.warpAffine(image_resized, M, target_size) if image_resized is not None else None
    mask_rotated = cv2.warpAffine(mask_resized, M, target_size, flags=cv2.INTER_NEAREST) if mask_resized is not None else None

    # Random flip horizontally and vertically
    if np.random.rand() > 0.5 and image_rotated is not None:  # 50% chance of horizontal flip
        image_rotated = cv2.flip(image_rotated, 1)
        mask_rotated = cv2.flip(mask_rotated, 1) if mask_rotated is not None else None
    
    if np.random.rand() > 0.5 and image_rotated is not None:  # 50% chance of vertical flip
        image_rotated = cv2.flip(image_rotated, 0)
        mask_rotated = cv2.flip(mask_rotated, 0) if mask_rotated is not None else None
    
    # Random brightness adjustment
    image_bright = cv2.convertScaleAbs(image_rotated, alpha=1.2, beta=np.random.uniform(-30, 30)) if image_rotated is not None else None
    
    # Random zoom
    zoom_factor = np.random.uniform(1.0, 1.2)
    h, w = image_bright.shape[:2] if image_bright is not None else (target_size[1], target_size[0])
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    image_zoom = cv2.resize(image_bright, (new_w, new_h)) if image_bright is not None else None
    mask_zoom = cv2.resize(mask_rotated, (new_w, new_h), interpolation=cv2.INTER_NEAREST) if mask_rotated is not None else None
    
    # Crop back to the original size after zoom
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    image_final = image_zoom[start_y:start_y + h, start_x:start_x + w] if image_zoom is not None else None
    mask_final = mask_zoom[start_y:start_y + h, start_x:start_x + w] if mask_zoom is not None else None

    return image_final, mask_final


# Function to augment the dataset and save the new images
def augment_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir, num_augmented_images):
    # Make output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # List all images in the directory
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Make sure there are equal numbers of images and masks
    assert len(image_files) == len(mask_files), "The number of images and masks must match."

    augmented_images = 0
    
    for image_file, mask_file in zip(image_files, mask_files):
        # Read image and mask
        image = cv2.imread(os.path.join(image_dir, image_file))
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {image_file} or {mask_file} due to loading error.")
            continue  # Skip the file if there was an issue loading it

        while augmented_images < num_augmented_images:
            # Apply augmentations
            augmented_image, augmented_mask = augment_image_and_mask(image, mask)

# Save the augmented image and mask
            augmented_image_path = os.path.join(output_image_dir, f"aug_{augmented_images}.png")
            augmented_mask_path = os.path.join(output_mask_dir, f"aug_{augmented_images}.png")
            
            cv2.imwrite(augmented_image_path, augmented_image)
            cv2.imwrite(augmented_mask_path, augmented_mask)

            augmented_images += 1

            if augmented_images >= num_augmented_images:
                break

        # If we have generated enough augmented images, stop
        if augmented_images >= num_augmented_images:
            break

    print(f"Generated {augmented_images} augmented images.")

# Example usage
image_dir = "dataset/images"  # Replace with your image directory
mask_dir = "dataset/masks"    # Replace with your mask directory
output_image_dir = "dataset/images"
output_mask_dir = "dataset/masks"

# How many augmented images you want to create
num_augmented_images = 100

augment_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir, num_augmented_images)