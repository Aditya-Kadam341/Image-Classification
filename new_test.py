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




# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only = True, monitor = "val_loss"),
#     tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience=5,restore_best_weights=True)
# ]

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs = 20,
#     callbacks=callbacks
# )

model = tf.keras.models.load_model("best_model.keras")
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.title("Loss")
# plt.show()

# plt.figure()
# plt.plot(history.history['accuracy'], label='train_acc')
# plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.title("Accuracy")
# plt.show()


# Evaluate numeric
loss, acc = model.evaluate(test_ds)
print("Test loss:", loss, "Test accuracy:", acc)

# Gather true labels and predictions
y_true = np.concatenate([y.numpy() for x,y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)


# recursively collect image files
img_extensions = ('*.jpg','*.jpeg','*.png')
pred_paths = []
for ext in img_extensions:
    pred_paths += glob.glob(os.path.join(pred_dir, '**', ext), recursive=True)

print("Found", len(pred_paths), "images to predict.")

X = []
paths_order = []
for p in pred_paths:
    try:
        img = load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH))
        arr = img_to_array(img) / 255.0
        X.append(arr)
        paths_order.append(p)
    except Exception as e:
        print("Skipping", p, ":", e)

if len(X) == 0:
    print("No images found in pred_dir. Check path or file types.")
else:
    X = np.stack(X)
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    for path, idx, prob in zip(paths_order, preds, np.max(probs, axis=1)):
        print(os.path.basename(path), "->", class_names[idx], f"({prob:.3f})")

model.save("final_model.keras") 