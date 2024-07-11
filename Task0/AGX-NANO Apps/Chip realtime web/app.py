from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('agxnano.h5')

# Get the input size expected by the model
input_shape = model.input_shape[1:3]


class_labels = ['agx','nano']
# Define a function to preprocess the input image
def preprocess_image(img):
    img = img.resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

# Define a function to decode the prediction
def decode_predictions(pred):
    predicted_class = np.argmax(pred, axis=1)
    return class_labels[predicted_class[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    preprocessed_image = preprocess_image(img)
    prediction = model.predict(preprocessed_image)
    predicted_label = decode_predictions(prediction)
    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
