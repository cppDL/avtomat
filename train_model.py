import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from unet_model import unet_model
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and prepare data
print("Loading data...")
image_dir = "dataset/images/"
mask_dir = "dataset/masks/"
images, masks = load_data(image_dir, mask_dir)

print("Data shapes:")
print(f"Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
print(f"Masks: {masks.shape}, range: [{masks.min():.3f}, {masks.max():.3f}]")

# Split data
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Create and compile model
model = unet_model(input_size=(128, 128, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
]

# Train the model
history = model.fit(
    train_images,
    train_masks,
    batch_size=16,
    epochs=100,
    validation_data=(val_images, val_masks),
    callbacks=callbacks
)

# Save the model
model.save('unet_model.h5')
print("Model saved as unet_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
