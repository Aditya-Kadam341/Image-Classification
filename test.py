import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


#Dataset
pred_dir = "image_dataset/seg_pred/seg_pred"
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"


# List only valid class directories (ignore hidden/system files)
train_classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls)) and not cls.startswith(".")]

# Calculate rows and columns dynamically
num_classes = len(train_classes)
cols = 3  # Set columns (adjustable)
rows = math.ceil(num_classes / cols)  # Compute required rows

# Plot images
plt.figure(figsize=(15, rows * 3))
for idx, class_name in enumerate(train_classes):
    class_path = os.path.join(train_dir, class_name)
    sample_img_name = next((img for img in os.listdir(class_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))), None)
    
    if sample_img_name:  # Ensure a valid image is found
        sample_img_path = os.path.join(class_path, sample_img_name)
        img = Image.open(sample_img_path)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

plt.tight_layout()
plt.show()



