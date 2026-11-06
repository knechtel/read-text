
import cv2
import numpy as np
import time
import pytesseract
from pathlib import Path


# Configurações
TEMPO_CAPTURA = 11
CAMERA = 0

PASTA = Path("capturas")
PASTA.mkdir(exist_ok=True)

ARQUIVO = PASTA / "foto.jpg"


# Abrir webcam e capturar fotografia
def tirar_foto():

    camera = cv2.VideoCapture(CAMERA)

    if not camera.isOpened():
        print("Erro ao abrir a webcam!")
        return None

    print("Webcam aberta!")
    print(f"Fotografia em {TEMPO_CAPTURA} segundos...")

    inicio = time.time()
    imagem = None

    try:

        while True:

            sucesso, frame = camera.read()

            if not sucesso:
                print("Erro ao capturar imagem!")
                break

            # Calcular tempo restante
            restante = TEMPO_CAPTURA - int(time.time() - inicio)

            if restante <= 0:

                imagem = frame.copy()
                print("Fotografia capturada!")
                break

            # Mostrar contagem regressiva
            cv2.putText(
                frame,
                f"Foto em: {restante}s",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Webcam", frame)

            # Pressionar Q para cancelar
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Captura cancelada!")
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

    return imagem


# Processamento da imagem
def processar_imagem(imagem):

    # Converter para escala de cinza
    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # Aumentar resolução
    altura, largura = cinza.shape

    ampliada = cv2.resize(
        cinza,
        (largura * 2, altura * 2),
        interpolation=cv2.INTER_CUBIC
    )

    # Melhorar contraste
    contraste = cv2.convertScaleAbs(
        ampliada,
        alpha=1.5,
        beta=0
    )

    # Reduzir ruídos
    suavizada = cv2.GaussianBlur(
        contraste,
        (3, 3),
        0
    )

    # Binarização
    _, binaria = cv2.threshold(
        suavizada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binaria


# Reconhecimento de texto OCR
def reconhecer_texto(imagem):

    processada = processar_imagem(imagem)

    config = "--oem 3 --psm 6"

    texto = pytesseract.image_to_string(
        processada,
        lang="por",
        config=config
    )

    return texto, processada


# Programa principal
def main():

    print("\n===== SISTEMA OCR =====\n")

    # Capturar fotografia
    imagem = tirar_foto()

    if imagem is None:
        return

    # Salvar fotografia
    cv2.imwrite(str(ARQUIVO), imagem)

    print(f"\nImagem salva: {ARQUIVO}")

    # Executar OCR
    print("\nProcessando imagem...")

    texto, processada = reconhecer_texto(imagem)

    # Mostrar resultado
    print("\n===== TEXTO RECONHECIDO =====\n")

    print(texto)

    print("\n=============================\n")

    # Salvar texto reconhecido
    arquivo_txt = PASTA / "texto.txt"
    arquivo_txt.write_text(texto, encoding="utf-8")

    print(f"Texto salvo em: {arquivo_txt}")

    # Exibir imagens
    cv2.imshow("Imagem Original", imagem)

    cv2.imshow("Imagem Processada", processada)

    print("Pressione qualquer tecla para fechar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

