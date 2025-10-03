import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore

# Settings
IMG_HEIGHT, IMG_WIDTH = 180, 180
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Load models once at startup
cnn_model = load_model("/Users/mayurideshmukh/Desktop/Image-Classification/project/models/best_model.keras")
try:
    mobilenet_model = load_model("/Users/mayurideshmukh/Desktop/Image-Classification/project/models/mobilenet_best.keras")
except Exception as e:
    print("⚠️ Could not load MobileNetV2 yet:", e)
    mobilenet_model = None
app = Flask(__name__)


print("🔍 Testing MobileNetV2 model load...")
if mobilenet_model:
    print(mobilenet_model.summary())  # check structure
else:
    print("⚠️ MobileNetV2 did not load")


def predict_image (model,img_path):
    img = load_img(img_path,target_size=(IMG_HEIGHT,IMG_WIDTH))
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr,axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(np.max(preds))

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# make sure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_file = None
    label, confidence = None, None   # ✅ always initialize

    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            if model_choice == "cnn":
                label, confidence = predict_image(cnn_model, filepath)

            elif model_choice == "mobilenet":
                if mobilenet_model is None:
                    prediction = "⚠️ MobileNetV2 model not loaded yet."
                else:
                    label, confidence = predict_image(mobilenet_model, filepath)

            # ✅ only format prediction if label was set
            if label is not None and confidence is not None:
                prediction = f"Prediction: {label} (Confidence: {confidence:.2f})"

            uploaded_file = filepath

    return render_template("index.html", prediction=prediction, uploaded_file=uploaded_file)


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

