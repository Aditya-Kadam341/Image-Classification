import os
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

# -----------------------
# Paths
# -----------------------
train_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_train/seg_train"
test_dir = "/Users/mayurideshmukh/Desktop/Image-Classification/image_dataset/seg_test/seg_test"

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
SEED = 123

# -----------------------
# Datasets
# -----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="int",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="int",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="int",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# -----------------------
# Build MobileNetV2 model
# -----------------------
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

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

# -----------------------
# Callbacks
# -----------------------
checkpoint_full = ModelCheckpoint(
    "models/mobilenet_best.keras",
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

checkpoint_weights = ModelCheckpoint(
    "models/mobilenet_best_weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint_full, checkpoint_weights, early_stop]

# -----------------------
# Train (only 5 epochs)
# -----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# -----------------------
# Evaluate on test set
# -----------------------
test_loss, test_acc = model.evaluate(test_ds)
print("✅ Test accuracy:", test_acc)
