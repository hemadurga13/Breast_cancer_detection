from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Explicit import for Pillow
from PIL import Image

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)  # Enable CORS

# Load the trained model (ensure the .h5 file is in the same folder)
MODEL_PATH = "breast_cancer_detection_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Configure the uploads folder and allowed file extensions
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return (
        "." in filename and 
        filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            print("File saved to:", file_path)

            try:
                img = image.load_img(file_path, target_size=(224, 224))
                print("Image loaded successfully")
            except Exception as e:
                print("Error loading image:", e)
                return jsonify({"error": "Error processing image", "message": str(e)}), 500

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            try:
                prediction = model.predict(img_array)
                print("Prediction output:", prediction)
            except Exception as e:
                print("Error during model prediction:", e)
                return jsonify({"error": "Error during prediction", "message": str(e)}), 500

            try:
                # Use threshold 0.5 to interpret prediction
                result = "Malignant (Cancer Detected) 😞" if prediction[0][0] > 0.5 else "Benign (No Cancer) 😊"
            except Exception as e:
                print("Error interpreting prediction:", e)
                result = "Error interpreting prediction"

            return jsonify({"prediction": result, "image_url": file_path})
        else:
            return jsonify({"error": "Invalid file format"}), 400

    except Exception as e:
        print("Error occurred in /upload:", e)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
