!pip install tensorflow opencv-python tensorflow-datasets matplotlib --quiet

# Import libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# 1. Data Collection and Preparation

IMG_SIZE = 64
BATCH_SIZE = 32

def preprocess(image, label):
    # Resize, normalize, convert to 3 channels (if needed)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Load EuroSAT RGB dataset (as SAR proxy)
(ds_train, ds_test), ds_info = tfds.load(
    'eurosat/rgb',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

ds_train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
class_names = ds_info.features['label'].names

# 2. CNN Model with VGG19 Feature Extraction


# Load VGG19 without top layers, use as feature extractor
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Training

EPOCHS = 8  # Increase for better accuracy

history = model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test
)

# 4. Evaluation and Visualization

# Evaluate on test data
loss, accuracy = model.evaluate(ds_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Plot accuracy curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training/Validation Accuracy')
plt.show()

# Visualize predictions on test images
for images, labels in ds_test.take(1):
    preds = model.predict(images)
    plt.figure(figsize=(15,6))
    for i in range(10):
        ax = plt.subplot(2, 5, i+1)
        plt.imshow(images[i].numpy())
        pred_label = np.argmax(preds[i])
        true_label = labels[i].numpy()
        plt.title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 5. Example: OpenCV Preprocessing (Optional, for SAR-like effect)

def add_sar_noise(image):
    # Convert to grayscale, add speckle noise, and back to 3 channels
    image = (image.numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
    noisy = cv2.add(gray, noise)
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_GRAY2RGB)
    return noisy_rgb / 255.0

# Visualize OpenCV SAR-like preprocessing
for images, labels in ds_test.take(1):
    plt.figure(figsize=(8,4))
    for i in range(5):
        ax = plt.subplot(1, 5, i+1)
        noisy_img = add_sar_noise(images[i])
        plt.imshow(noisy_img)
        plt.axis('off')
    plt.suptitle("SAR-like Noisy Images (OpenCV)", fontsize=12)
    plt.show()
    break

# =======================
# End of pipeline
# =======================
