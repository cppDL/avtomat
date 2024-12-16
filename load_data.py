import cv2
import os
import numpy as np

def load_data(image_dir, mask_dir, img_size=(128, 128)):
    images = []
    masks = []
    
    # Get sorted list of files to ensure matching order
    image_files = sorted(os.listdir(image_dir))
    
    for img_file in image_files:
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Read and process image
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_file}")
                continue
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            images.append(img)
            
            # Read and process corresponding mask
            mask_path = os.path.join(mask_dir, img_file)  # Assuming same filename
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {img_file}")
                continue
            mask = cv2.resize(mask, img_size)
            mask = (mask > 128).astype(np.float32)  # Convert to binary and float32
            masks.append(mask)
    
    return np.array(images), np.array(masks)