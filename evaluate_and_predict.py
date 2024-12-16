import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from load_data import load_data
from sklearn.model_selection import train_test_split
import os
import cv2

def plot_sample(image, true_mask, pred_mask, index):
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('True Mask')
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    # Add overlay visualization
    plt.subplot(1, 4, 4)
    plt.title('Overlay')
    plt.imshow(image)
    plt.imshow(pred_mask, cmap='Reds', alpha=0.5)  # Red overlay with 50% transparency
    plt.axis('off')
    
    plt.suptitle(f'Sample {index}')
    plt.show()

# Create output directories if they don't exist
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
image_dir = "dataset/images/"
mask_dir = "dataset/masks/"
images, masks = load_data(image_dir, mask_dir)

# Split data (use same split as training)
_, val_images, _, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Load model
print("Loading model...")
model = load_model('unet_model.h5')

# Make predictions
print("Making predictions...")
predictions = model.predict(val_images)

# Print statistics
print("\nPrediction statistics:")
print(f"Shape: {predictions.shape}")
print(f"Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"Mean: {predictions.mean():.3f}")
print(f"Std: {predictions.std():.3f}")

# Create binary predictions
binary_predictions = (predictions > 0.5).astype(np.float32)
binary_predictions = binary_predictions.squeeze(axis=-1)

# Calculate accuracy for each prediction
accuracies = np.mean((binary_predictions == val_masks).astype(np.float32), axis=(1,2))
print("\nAccuracies for each prediction:", accuracies)

# Sort predictions by accuracy
sorted_indices = np.argsort(accuracies)[::-1]  # Reverse order to get highest first
num_best = 5  # Number of best predictions to show

print(f"\nShowing {num_best} best predictions...")
for rank, idx in enumerate(sorted_indices[:num_best]):
    image = val_images[idx]
    pred_mask = binary_predictions[idx]
    true_mask = val_masks[idx]
    accuracy = accuracies[idx]
    
    # Save visualization
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.title('Input Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('True Mask')
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Overlay')
    plt.imshow(image)
    plt.imshow(pred_mask, cmap='Reds', alpha=0.5)
    plt.axis('off')
    
    plt.suptitle(f'Best Prediction {rank+1} (Accuracy: {accuracy:.3f})')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'best_prediction_{rank+1}.png'))
    plt.show()
    plt.close()

print(f"\nPredictions saved to {output_dir}/")

# Calculate and print overall metrics
accuracy = np.mean(accuracies)
print(f"\nOverall validation accuracy: {accuracy:.3f}")
