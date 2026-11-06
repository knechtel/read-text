from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


# Imagem armazenada na pasta capturas, relativa à pasta deste script.
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_IMAGEM = PASTA_PROJETO / "capturas" / "download1.png"
TESSDATA_DIR = "/usr/share/tessdata"

def preparar_imagem(img):
    """Aumenta e realça a imagem para ajudar o OCR com escrita à mão."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    altura, largura = cinza.shape
    ampliada = cv2.resize(
        cinza,
        (largura * 2, altura * 2),
        interpolation=cv2.INTER_CUBIC,
    )
    ampliada = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ampliada)
    suavizada = cv2.GaussianBlur(ampliada, (3, 3), 0)
    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binaria


def texto_das_linhas(resultado):
    """Recompõe o texto preservando as quebras de linha do OCR."""
    linhas = []
    chave_anterior = None
    palavras = []

    for i, palavra in enumerate(resultado["text"]):
        palavra = palavra.strip()
        if not palavra:
            continue

        chave = (
            resultado["block_num"][i],
            resultado["par_num"][i],
            resultado["line_num"][i],
        )
        if chave_anterior is not None and chave != chave_anterior:
            linhas.append(" ".join(palavras))
            palavras = []
        palavras.append(palavra)
        chave_anterior = chave

    if palavras:
        linhas.append(" ".join(palavras))
    return "\n".join(linhas)


def main():
    img = cv2.imread(str(ARQUIVO_IMAGEM))
    if img is None:
        raise FileNotFoundError(
            f"Não foi possível abrir {ARQUIVO_IMAGEM}. Confira o caminho da imagem."
        )

    # Equivalente ao cv2.cvtColor do exemplo; útil também para exibição.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processada = preparar_imagem(img)
    rgb_ocr = cv2.cvtColor(processada, cv2.COLOR_GRAY2RGB)

   # config_tesseract = "--tessdata-dir tessdata --oem 3 --psm 6"
    config_tesseract = f'--tessdata-dir "{TESSDATA_DIR}" --oem 3 --psm 6'
    resultado = pytesseract.image_to_data(
        rgb_ocr,
        config=config_tesseract,
        lang="por",
        output_type=Output.DICT,
    )
    texto = texto_das_linhas(resultado)

    print("Texto reconhecido:\n")
    print(texto)
    print("\nConfiança por palavra:")
    for palavra, confianca in zip(resultado["text"], resultado["conf"]):
        if palavra.strip() and confianca != "-1":
            print(f"{palavra}: {confianca}%")

    # Salva o texto ao lado da imagem.
    arquivo_saida = ARQUIVO_IMAGEM.with_name("texto_reconhecido.txt")
    arquivo_saida.write_text(texto, encoding="utf-8")
    print(f"\nTexto salvo em: {arquivo_saida}")

    # Para exibir a imagem em execução local (não necessário para o OCR):
    cv2.imshow("Imagem original", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Imagem preparada para OCR", processada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
