from load_data import load_data
from unet_model import unet_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load dataset
image_dir = "dataset/images/"
mask_dir = "dataset/masks/"
images, masks = load_data(image_dir, mask_dir)
masks = masks / 255.0  # Normalize masks to [0, 1]

# Split dataset into train and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Define the U-Net model
model = unet_model(input_size=(128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=50,
    batch_size=16
)

# Save the trained model
model.save("unet_model.h5")
print("Model saved as unet_model.h5")
