import os
import time
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models # type: ignore

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Classes (same order as training)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ---------------------------
# Load CNN model
# ---------------------------
cnn_model = load_model("/Users/mayurideshmukh/Desktop/Image-Classification/project/models/best_model.keras")

# ---------------------------
# Load MobileNetV2 with weights
# ---------------------------
IMG_HEIGHT, IMG_WIDTH = 180, 180

def build_mobilenet(num_classes):
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
    return model

mobilenet_model = None
try:
    mobilenet_model = build_mobilenet(len(class_names))
    mobilenet_model.load_weights("/Users/mayurideshmukh/Desktop/Image-Classification/project/models/mobilenet_best.weights.h5")
    print("✅ MobileNetV2 loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load MobileNetV2: {e}")

# ---------------------------
# Image Preprocessing
# ---------------------------
def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------
# Routes
# ---------------------------


UPLOAD_FOLDER = "/Users/mayurideshmukh/Desktop/Image-Classification/project/static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    filename = None
    model_choice = "cnn"

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Ensure static folder exists
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            # Make filename unique to avoid overwrites / caching
            original_filename = file.filename
            filename = f"{int(time.time())}_{original_filename}"

            # Save inside static folder
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Capture selected model
            model_choice = request.form.get("model_choice", "cnn")

            # Prepare image for prediction
            img_array = prepare_image(filepath)

            # Make prediction
            if model_choice == "cnn":
                preds = cnn_model.predict(img_array)
                label = class_names[np.argmax(preds)]
            elif model_choice == "mobilenet" and mobilenet_model:
                preds = mobilenet_model.predict(img_array)
                label = class_names[np.argmax(preds)]
            else:
                label = "⚠️ MobileNetV2 not available yet."

    # Pass filename to template for displaying
    return render_template("index.html", filename=filename, label=label, model_choice=model_choice)




@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename=filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
