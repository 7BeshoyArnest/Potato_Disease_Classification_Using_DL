from flask import Flask, request, jsonify
from flasgger import Swagger
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import traceback

# --- Configuration ---
MODEL_PATH = r'C:\Users\AL-MASA\End_to_End_projects_in_machine_learning\Potato_disease_classification\potato_disease_classification_saved_model.h5'
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 256))
CLASS_NAMES = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]

# --- App init ---
app = Flask(__name__)
swagger = Swagger(app)

# --- Load model once at startup ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    traceback.print_exc()

def preprocess_image(image_bytes):
    """Preprocess image bytes for model prediction.
    Resizes to IMAGE_SIZE x IMAGE_SIZE, converts to float32 and rescales by 1/255.
    Returns a batch tensor shape (1, IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict potato leaf disease from an uploaded image file.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file to classify (jpg, png)
    responses:
      200:
        description: Prediction results
        schema:
          type: object
          properties:
            predicted_label:
              type: string
            confidence:
              type: number
      400:
        description: Bad request (e.g., no file provided)
      500:
        description: Server error
    """
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        img_bytes = file.read()
        input_tensor = preprocess_image(img_bytes)
        preds = model.predict(input_tensor)
        preds = preds[0]  # single sample
        top_idx = int(np.argmax(preds))
        label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
        confidence = float(np.max(preds))
        #return jsonify({'predicted_label': label, 'confidence': round(confidence, 4)})

        return jsonify([
             {'predicted_label': label},
             {'confidence': round(confidence, 4)}
        ])

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


@app.route('/')
def index():
    return jsonify({'message': 'Potato disease classification API. See /apidocs/ for Swagger UI.'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)