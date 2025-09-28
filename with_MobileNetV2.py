import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
import glob
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore



#Dataset
pred_dir = "image_dataset/seg_pred/seg_pred"
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"

 
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
SEED = 123


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode = "int",
    seed = SEED,
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE,
    validation_split = 0.2,
    subset = "training"
)



val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode = "int",
    seed = SEED,
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE,
    validation_split = 0.2,
    subset = "validation"
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode = "int",
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)



# Pre-trained base
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)



# Freeze base
base_model.trainable = False

# Add our classification head
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()


callbacks = [
    tf.keras.callbacks.ModelCheckpoint("mobilenet_best.keras", save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# Unfreeze from a certain layer (fine-tune deeper layers)
base_model.trainable = True


# Freeze earlier layers, fine-tune later ones
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layers.trainable= False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)


test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)