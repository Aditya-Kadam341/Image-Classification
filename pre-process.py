import os
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image

# Dataset Paths
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"

# Image settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

# Normalize images using MobileNetV2 preprocessing
normalization_layer = tf.keras.applications.mobilenet_v2.preprocess_input
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load Pretrained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze weights

# Custom Classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate on Test Data
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")

# Save Model
model.save("image_classifier_model.keras")
print("Model saved successfully!")


# Function to Classify an Image
def classify_image(image_path, model_path="image_classifier_model.keras"):
    """Loads a saved model and classifies an image."""
    model = load_model(model_path)  # Load trained model

    # Load and preprocess image
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    return predicted_class, predictions[0]


# Example Usage
image_path = "/Users/mayurideshmukh/Desktop/Image-Classification/test1.jpg"
predicted_class, probabilities = classify_image(image_path)
print(f"Predicted Class: {predicted_class}")
print(f"Prediction Probabilities: {probabilities}")
