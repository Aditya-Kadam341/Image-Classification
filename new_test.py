import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models # type: ignore



#Dataset
pred_dir = "image_dataset/seg_pred/seg_pred"
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"

for p in (train_dir, test_dir, pred_dir):
    print(p, "=>", "FOUND" if os.path.exists(p) else "MISSING")

def count_images_by_class(directory) : 
    counts ={}
    if not os.path.exists(directory) : 
        return counts
    for cls in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            counts[cls] = sum(1 for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png')))
    return counts

# print("Train counts:", count_images_by_class(train_dir))
# print("Test counts: ", count_images_by_class(test_dir))



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
print("Classes:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)


plt.figure(figsize=(10,8))
for images, labels in train_ds.take(1):
    images = images.numpy().astype("uint8")
    labels = labels.numpy()
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()


num_classes = len(class_names)

data_augentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08)
])

model = models.Sequential([
    layers.Rescaling(1./255, input_shape= (IMG_HEIGHT, IMG_WIDTH,3)),
    data_augentation,
    layers.Conv2D(32,3, activation = "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, activation = "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3, activation = "relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation = "softmax")
])

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

# model.summary()




callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only = True, monitor = "val_loss"),
    tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience=5,restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 20,
    callbacks=callbacks
)

