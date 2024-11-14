from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image

from keras import models

import numpy as np


block_size = (500, 500)

app = Flask(__name__)
CORS(app)

model = models.load_model('models/ev_slices.keras')

def classifyImage(image):
    image = image.convert("RGB")
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(predicted_class)
    return int(predicted_class)

def process_image(image):
    classification = classifyImage(image)
    return {"classification": classification}

@app.route('/classify-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    result = process_image(image)

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
