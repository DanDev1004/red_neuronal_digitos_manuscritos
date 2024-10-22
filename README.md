# Red Neuronal para Identificar Dígitos Manuscritos

Una app hecha con Python y Flask donde implementamos algoritmos de deep learning, concretamente, redes neuronales densas.

El usuario puede dibujar en un canvas que está hecho con HTML, CSS y JavaScript. Dicha información es enviada al modelo a través de una solicitud HTTP con AJAX y jQuery.

Puede verificar el resultado aquí: [Link de la app](https://red-neuronal-digitos-dan.onrender.com/)

Tal vez sea un poco lenta debido a que está desplegado en un servidor gratuito.

Esta versión está estructurada para generar un modelo preentrenado, de manera que solo tenga que cargarse la imagen generada por Docker. Esta imagen se encuentra en Docker Hub:

[Imagen en Docker Hub](https://hub.docker.com/repository/docker/danlukae1004/red_neuronal_digitos_dan/general)

Podrás ejecutarlo sin necesidad de instalar todas las dependencias:

```bash
docker pull danlukae1004/red_neuronal_digitos_dan



