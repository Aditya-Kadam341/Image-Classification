import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np



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


