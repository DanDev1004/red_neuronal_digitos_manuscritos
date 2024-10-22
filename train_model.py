import tensorflow as tf
import tensorflow_datasets as tfds
import math

def train_and_save_model():
    dataset, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
    conjunto_entrenamiento, _ = dataset['train'], dataset['test']

    def normalizar(imagenes, etiquetas):
        imagenes = tf.cast(imagenes, tf.float32)
        imagenes /= 255
        return imagenes, etiquetas

    conjunto_entrenamiento = conjunto_entrenamiento.map(normalizar)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    TAMANO_BATCH = 32
    num_ejemplos_entrenamiento = metadatos.splits['train'].num_examples
    conjunto_entrenamiento = conjunto_entrenamiento.repeat().shuffle(num_ejemplos_entrenamiento).batch(TAMANO_BATCH)

    modelo.fit(conjunto_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ejemplos_entrenamiento / TAMANO_BATCH))

    modelo.save('mnist_model.h5')#guardando modelo

if __name__ == "__main__":
    train_and_save_model()
