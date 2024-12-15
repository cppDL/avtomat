import cv2
import os
import numpy as np

def load_data(image_dir, mask_dir, img_size=(128, 128)):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        img = cv2.resize(img, img_size)
        images.append(img)

        mask_path = os.path.join(mask_dir, filename.replace('.jpg', '_mask.png'))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        masks.append(mask)

    return np.array(images), np.array(masks)

image_dir = "dataset/images/"
mask_dir = "dataset/masks/"
images, masks = load_data(image_dir, mask_dir)
masks = masks / 255.0  # Normalize masks to [0, 1]