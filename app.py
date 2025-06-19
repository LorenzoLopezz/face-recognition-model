#!/usr/bin/env python3
# app.py

import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Mapea índice de salida a etiqueta
CLASS_MAPPING = {
    0: 'real',
    1: 'spoof'
}

# Inicializa la app y carga el modelo una sola vez
app = Flask(__name__)
model = load_model('model/fasnet_trained.h5')

def preprocess_image_from_base64(base64_str, target_size=(224, 224)):
    """
    Decodifica una imagen en Base64, la redimensiona y normaliza
    para pasar al modelo.
    """
    # Decodificar Base64 a bytes
    img_data = base64.b64decode(base64_str)
    # Cargar imagen con PIL
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    # Redimensionar
    img = img.resize(target_size)
    # Convertir a array y normalizar
    arr = np.array(img, dtype=np.float32) / 255.0
    # Añadir dimensión de batch
    return np.expand_dims(arr, axis=0)

def preprocess_image_from_bytes(img_bytes, target_size=(224, 224)):
    """
    Procesa una imagen desde bytes directamente, la redimensiona y normaliza
    para pasar al modelo.
    """
    # Cargar imagen con PIL desde bytes
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Redimensionar
    img = img.resize(target_size)
    # Convertir a array y normalizar
    arr = np.array(img, dtype=np.float32) / 255.0
    # Añadir dimensión de batch
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Acepta imágenes en dos formatos:
    1. JSON con clave "image" cuyo valor sea el string Base64 de la imagen
    2. Form-data con un archivo de imagen en el campo "image"
    
    Devuelve JSON con:
      - prediction: "real" o "spoof"
      - confidence: probabilidad asociada
    """
    try:
        # Verificar si es multipart/form-data (blob/archivo)
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            if 'image' not in request.files:
                return jsonify({'error': 'Falta archivo "image" en form-data'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó archivo'}), 400
            
            # Leer bytes del archivo
            img_bytes = file.read()
            x = preprocess_image_from_bytes(img_bytes)
            
        # Si no es form-data, asumir que es JSON con base64
        else:
            data = request.get_json(force=True)
            if 'image' not in data:
                return jsonify({'error': 'Falta clave "image" en el JSON'}), 400
            
            # Limpiar el string base64 si tiene prefijo data:image
            base64_str = data['image']
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            x = preprocess_image_from_base64(base64_str)
        
        # Inferencia
        probs = model.predict(x)[0]
        idx = int(np.argmax(probs))
        label = CLASS_MAPPING.get(idx, 'unknown')
        confidence = float(probs[idx])
        
        return jsonify({
            'prediction': label,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ejecutar en modo debug en localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
