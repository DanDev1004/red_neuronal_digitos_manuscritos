from flask import Flask, request, jsonify, render_template #Flask para usar python en la web y manejar solicitudes HTTP
from flask_cors import CORS #permisos para que los template interactuen con el backend flask
import tensorflow as tf #libreria para crear nuestros modelos 
import tensorflow_datasets as tfds #conjunto de datos 
from sklearn.metrics import r2_score #calificación de la precisión en nuestros modelos
import numpy as np #para manejar arrays
import math #matematica basica 


#Creamos la app y habilitamos CORS
app = Flask(__name__)
CORS(app)  


#cargamos dataset MNIST (colección de 70,000 imágenes de dígitos manuscritos (0-9) en escala de grises)
dataset, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
conjunto_entrenamiento, conjunto_prueba = dataset['train'], dataset['test']



#Normalizamos las imagenes(digitos que ingresa el usuario), es decir, convertimos a datos float32
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #establecemos escala de valores en base a pixeles (0 representa negro, 255 blanco)
    return imagenes, etiquetas

#Separamos conjuntos de entrenamiento y prueba
conjunto_entrenamiento = conjunto_entrenamiento.map(normalizar)
conjunto_prueba = conjunto_prueba.map(normalizar)


#Capas de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #Convertimos la entrada 2D en un vector 1D
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') #Una salida con 10 neuronas
])

#Compilación del modelo
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Configuración de lotes(el modelo usará 32 imagenes a la vez, mezclandolas aleatoriamente)
TAMANO_BATCH = 32
num_ejemplos_entrenamiento = metadatos.splits['train'].num_examples
conjunto_entrenamiento = conjunto_entrenamiento.repeat().shuffle(num_ejemplos_entrenamiento).batch(TAMANO_BATCH)

#entrenamiento
modelo.fit(
    conjunto_entrenamiento, epochs=5,
    steps_per_epoch=math.ceil(num_ejemplos_entrenamiento / TAMANO_BATCH)
)


#evaluación
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


#ruta raiz 
@app.route('/')
def home():
    return render_template('index.html')



#ruta para la predicción la cual será usada por AJAX mediante JQuery
@app.route('/predict', methods=['POST'])
def predecir():
    try:
        datos = request.form['pixeles']
        arr = np.fromstring(datos, np.float32, sep=",").reshape(1, 28, 28, 1)

        #predicción
        valores_prediccion = modelo.predict(arr)
        prediccion = int(np.argmax(valores_prediccion))


        return jsonify({'prediccion': prediccion})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
