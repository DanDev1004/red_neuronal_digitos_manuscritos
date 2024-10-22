from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from dotenv import load_dotenv

#cargando variables de entorno
load_dotenv()
app_url = os.getenv("FLASK_APP_URL")

#indicamos que no usaremos GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#url generada por render
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://red_neuronal_digitos_dan.onrender.com"}})

#cargando el modelo preentrenado
modelo = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def home():
    return render_template('index.html', app_url=app_url)

@app.route('/predict', methods=['POST'])
def predecir():
    try:
        datos = request.form['pixeles']
        arr = np.fromstring(datos, np.float32, sep=",").reshape(1, 28, 28, 1)
        valores_prediccion = modelo.predict(arr)
        prediccion = int(np.argmax(valores_prediccion))
        return jsonify({'prediccion': prediccion})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False) 
