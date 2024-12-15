from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("unet_model.h5")

# Predict on validation images
predictions = model.predict(val_images)

# Apply a threshold to get binary masks
threshold = 0.5
binary_masks = (predictions > threshold).astype(np.uint8)

# Function to visualize results
def plot_results(image, true_mask, predicted_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(true_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')

    plt.show()

# Visualize the first image and its predicted mask
plot_results(val_images[0], val_masks[0], binary_masks[0])
