from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model
MODEL_PATH = 'catdogmodel.h5'
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# Ensure that the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for the homepage
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        # Preprocess the image
        img = load_img(filepath, target_size=(64, 64))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Scale the image

        # Predict the class
        prediction = model.predict(img)
        print(f"Prediction: {prediction}")

        if prediction[0][0] > 0.5:
            result = "Dog"
        else:
            result = "Cat"

        return render_template('result.html', prediction=result, filename=filename)
    return redirect(url_for('index'))

# Route to display the image and result
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)