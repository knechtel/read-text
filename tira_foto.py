import time
from pathlib import Path

import cv2


TEMPO_ESPERA = 10
CAMERA = 0
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_SAIDA = PASTA_PROJETO / "capturas" / "download1.png"


def main():
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(CAMERA)

    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    inicio = time.monotonic()
    foto = None

    try:
        while True:
            sucesso, frame = camera.read()
            if not sucesso:
                raise RuntimeError("Não foi possível capturar imagem da webcam.")

            restante = max(0, TEMPO_ESPERA - int(time.monotonic() - inicio))
            if restante == 0:
                foto = frame.copy()
                break

            cv2.putText(
                frame,
                f"Prepare-se: foto em {restante}s",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Captura da webcam", frame)

            # Q cancela a captura antes de completar os 10 segundos.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Captura cancelada.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not cv2.imwrite(str(ARQUIVO_SAIDA), foto):
        raise RuntimeError(f"Não foi possível salvar a foto em {ARQUIVO_SAIDA}.")

    print(f"Foto salva em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
