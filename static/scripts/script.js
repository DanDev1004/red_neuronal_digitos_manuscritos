//Configuración del lienzo donde el usuario dibujará los dígitos 
var lienzo = document.getElementById("canvas");
var ctx = lienzo.getContext("2d");
ctx.strokeStyle = "#000000";
ctx.lineWidth = 10;
var mousePresionado = false;


//Indicamos como va a interactuar el lienzo segun los eventos del mouse
lienzo.onmousedown = function (e) {
    var pos = corregirPosicion(e, lienzo);
    ctx.clearRect(0, 0, lienzo.width, lienzo.height);
    mousePresionado = true;
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    return false;
};

lienzo.onmousemove = function (e) {
    var pos = corregirPosicion(e, lienzo);
    if (mousePresionado) {
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }
};

lienzo.onmouseup = function () {
    mousePresionado = false;

    var pixeles = [];

    //Cada imagen en el mnist es un arreglo 28x28
    for (var x = 0; x < 28; x++) {
        for (var y = 0; y < 28; y++) {
            var imgData = ctx.getImageData(y * 10, x * 10, 10, 10); //Extraemos la imagen en 10x10 ya que el lienzo tiene un tamaño de 280x280
            var data = imgData.data;


            var totalAlpha = 0;
            for (var i = 0; i < data.length; i += 4) {
                totalAlpha += data[i + 3]; //Sumar el valor de alpha
            }
            var promedioAlpha = totalAlpha / (data.length / 4); //Promedio de alpha
            var color = promedioAlpha / 255; //Normalizamos de 0-1

            color = (Math.round(color * 100) / 100).toFixed(2);
            pixeles.push(color);
        }
    }

    console.log(pixeles); //imprimimos la información numerica de los pixeles en consola


    //mandamos la información en una solicitud POST
    $.post(`${appUrl}/predict`, { pixeles: pixeles.join(",") },
        function (respuesta) {
            //imprimimos resultado en consola y en la etiqueta html
            console.log("Resultado: " + respuesta.prediccion);
            $("#resultado").html("Predicción: " + respuesta.prediccion);
        }
    );
};

//Corregimos posiciones durante el dibujo para una mejor calidad de imagen
function corregirPosicion(e, gCanvasElement) {
    var x, y;
    if (e.pageX || e.pageY) {
        x = e.pageX;
        y = e.pageY;
    } else {
        x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
        y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    }
    x -= gCanvasElement.offsetLeft;
    y -= gCanvasElement.offsetTop;
    return { x: x, y: y };
}
