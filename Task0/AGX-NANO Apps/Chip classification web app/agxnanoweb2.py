# %% [code] {"jupyter":{"outputs_hidden":false}}
from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# %% [code] {"jupyter":{"outputs_hidden":false}}
app = Flask(__name__)

# Load the trained model
model = load_model('agxnano.h5')

# Get the input size expected by the model
input_shape = model.input_shape[1:3]

class_labels=['agx','nano']

# %% [code] {"jupyter":{"outputs_hidden":false}}
def preprocess_image(img_path, target_size=input_shape):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

# Define a function to decode the prediction
def decode_predictions(pred):
    predicted_class = np.argmax(pred, axis=1)
    return class_labels[predicted_class[0]]

# %% [code] {"jupyter":{"outputs_hidden":false}}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        preprocessed_image = preprocess_image(filepath)
        prediction = model.predict(preprocessed_image)
        predicted_label = decode_predictions(prediction)
        return render_template('result.html', label=predicted_label)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)