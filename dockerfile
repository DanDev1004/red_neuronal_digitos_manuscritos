
FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos
COPY . .


COPY mnist_model.h5 .


EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]