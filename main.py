
import cv2
import numpy as np
import time
import pytesseract
from google.colab.patches import cv2_imshow
from google.colab.output import eval_js
from IPython.display import display, Javascript
from base64 import b64decode

# Abrir webcam e capturar fotografia
def tirar_foto():

    display(Javascript("""
    async function abrirCamera() {

        const video = document.createElement('video');

        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;
        video.style.width = '640px';

        document.body.appendChild(video);

        await video.play();

        await new Promise(resolve => setTimeout(resolve, 10000));

        const canvas = document.createElement('canvas');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getTracks().forEach(track => track.stop());

        video.remove();

        return canvas.toDataURL('image/jpeg');
    }
    """))

    imagem_base64 = eval_js("abrirCamera()")

    imagem_bytes = b64decode(imagem_base64.split(',')[1])

    imagem_array = np.frombuffer(
        imagem_bytes,
        dtype=np.uint8
    )

    imagem = cv2.imdecode(
        imagem_array,
        cv2.IMREAD_COLOR
    )

    return imagem


# Executar captura
print("Abrindo webcam...")
print("Aguarde 10 segundos para a fotografia.")

imagem = tirar_foto()

# Salvar fotografia
cv2.imwrite('/capturas/foto.jpg', imagem)
#img1 = cv2.imread('/capturas/foto.jpg')
rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
print("Fotografia capturada!")
#texto = pytesseract.image_to_string(rgb)
#config_tesseract = '--tessdata-dir tessdata'
#texto = pytesseract.image_to_string(rgb)

texto = pytesseract.image_to_string(rgb)
print("do->")
print(texto)
# Exibir fotografia
cv2_imshow(imagem)


