# Usa la imagen oficial de Python 3.12 como base
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Asegurarse de que el modelo MNIST está incluido y en la ubicación correcta
COPY mnist_model.h5 .


# Exponer el puerto que utiliza la aplicación Flask
EXPOSE 8000

# Comando para ejecutar la aplicación usando Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]