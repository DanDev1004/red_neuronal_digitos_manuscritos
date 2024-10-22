from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import r2_score
import numpy as np
import math
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
app_url = os.getenv("FLASK_APP_URL")

# Deshabilitar el uso de la GPU para evitar conflictos
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Crear la app y habilitar CORS
app = Flask(__name__)
CORS(app)

# Cargar dataset MNIST
dataset, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
conjunto_entrenamiento, conjunto_prueba = dataset['train'], dataset['test']

# Normalizar imágenes
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas

# Aplicar normalización
conjunto_entrenamiento = conjunto_entrenamiento.map(normalizar)
conjunto_prueba = conjunto_prueba.map(normalizar)

# Definir el modelo de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Configuración de lotes
TAMANO_BATCH = 32
num_ejemplos_entrenamiento = metadatos.splits['train'].num_examples
conjunto_entrenamiento = conjunto_entrenamiento.repeat().shuffle(num_ejemplos_entrenamiento).batch(TAMANO_BATCH)

# Entrenamiento del modelo (reducido para ahorrar memoria)
modelo.fit(
    conjunto_entrenamiento, epochs=1,  # Reducir épocas para minimizar uso de memoria
    steps_per_epoch=math.ceil(num_ejemplos_entrenamiento / TAMANO_BATCH)
)

# Evaluar el modelo
def evaluar_modelo(conjunto_prueba):
    etiquetas_reales = []
    etiquetas_predichas = []

    for imagenes, etiquetas in conjunto_prueba.batch(TAMANO_BATCH):
        predicciones = modelo.predict(imagenes)
        etiquetas_predichas_temp = np.argmax(predicciones, axis=1)

        etiquetas_reales.extend(etiquetas.numpy())
        etiquetas_predichas.extend(etiquetas_predichas_temp)

    r2 = r2_score(etiquetas_reales, etiquetas_predichas)
    return r2

r2 = evaluar_modelo(conjunto_prueba)
print(f"Puntuación R²: {r2:.4f}")

# Ruta raíz
@app.route('/')
def home():
    return render_template('index.html', app_url=app_url)

# Ruta para predicción
@app.route('/predict', methods=['POST'])
def predecir():
    try:
        datos = request.form['pixeles']
        arr = np.fromstring(datos, np.float32, sep=",").reshape(1, 28, 28, 1)

        # Predicción
        valores_prediccion = modelo.predict(arr)
        prediccion = int(np.argmax(valores_prediccion))

        return jsonify({'prediccion': prediccion})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
