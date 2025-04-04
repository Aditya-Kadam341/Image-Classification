import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore



#Dataset
pred_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_pred/seg_pred"
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"

# Set image dimensions and batch size
IMG_SIZE = (150, 150)  # You can also try (224, 224) for larger models
BATCH_SIZE = 32

# load datasets

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2, subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


#Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Check dataset structure
for image_batch, label_batch in train_dataset.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")


# Load the MobileNetV2 model (without top layers)
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3),
                                               include_top=False,  # Remove original classifier
                                               weights='imagenet')  # Use pretrained weights

# Freeze the base model (so we don’t retrain ImageNet weights)
base_model.trainable = False

# Add custom classifier on top
model = models.Sequential([
    base_model,  # Pretrained feature extractor
    layers.GlobalAveragePooling2D(),  # Reduce feature map to vector
    layers.Dense(128, activation='relu'),  # Custom dense layer
    layers.Dropout(0.3),  # Prevent overfitting
    layers.Dense(6, activation='softmax')  # Output layer (6 classes)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)  # Adjust epochs as needed

# Evaluate on test dataset
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")