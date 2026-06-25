"""
Image Classification CNN — Flask Web App
Run: python app.py  (open http://localhost:5001)
"""

import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
CLASS_ICONS = ['✈️','🚗','🐦','🐱','🦌','🐶','🐸','🐴','🚢','🚛']

# Load model once
model = None
try:
    import tensorflow as tf
    if os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
        print("✓ Model loaded")
    else:
        print("⚠  No model found — run train.py first")
except ImportError:
    print("⚠  TensorFlow not installed — demo mode")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((32, 32)).convert('RGB')
    arr = np.array(img, dtype='float32') / 255.0
    return arr[np.newaxis, ...]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Make a base64 thumbnail for display
    thumb = img.copy()
    thumb.thumbnail((200, 200))
    buf = io.BytesIO()
    thumb.save(buf, format='PNG')
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    if model is None:
        # Demo: random prediction for UI showcase
        probs = np.random.dirichlet(np.ones(10) * 0.5)
    else:
        probs = model.predict(preprocess(img), verbose=0)[0]

    top3_idx = probs.argsort()[-3:][::-1]
    results = [
        {
            'class': CLASS_NAMES[i],
            'icon': CLASS_ICONS[i],
            'confidence': round(float(probs[i]) * 100, 1)
        }
        for i in top3_idx
    ]
    return jsonify({'predictions': results, 'thumbnail': thumb_b64})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
