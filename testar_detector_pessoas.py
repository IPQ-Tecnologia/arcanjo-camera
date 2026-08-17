from pathlib import Path

import cv2
from ultralytics import YOLO


ENTRADA = Path("amostras/teste_pessoas.jpg")
SAIDA = Path("amostras/deteccoes_yolo.jpg")


def main() -> None:
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"Imagem de teste não encontrada: {ENTRADA}"
        )

    # Na primeira execução, o peso será baixado automaticamente.
    modelo = YOLO("yolo11n.pt")

    ids_pessoa = [
        indice
        for indice, nome in modelo.names.items()
        if str(nome).lower() == "person"
    ]

    if not ids_pessoa:
        raise RuntimeError(
            "A classe 'person' não foi encontrada no modelo."
        )

    resultados = modelo.predict(
        source=str(ENTRADA),
        classes=ids_pessoa,
        conf=0.20,
        imgsz=960,
        device="cpu",
        verbose=False,
    )

    resultado = resultados[0]
    caixas = resultado.boxes

    quantidade = 0 if caixas is None else len(caixas)

    print(f"Pessoas encontradas pelo YOLO: {quantidade}")

    if caixas is not None:
        for indice, caixa in enumerate(caixas, start=1):
            x1, y1, x2, y2 = [
                round(valor)
                for valor in caixa.xyxy[0].tolist()
            ]

            confianca = float(caixa.conf[0])

            print(
                f"Pessoa {indice}: "
                f"x1={x1}, y1={y1}, "
                f"x2={x2}, y2={y2}, "
                f"confiança={confianca:.2f}"
            )

    imagem_anotada = resultado.plot()

    if not cv2.imwrite(str(SAIDA), imagem_anotada):
        raise RuntimeError(
            f"Não foi possível salvar a imagem: {SAIDA}"
        )

    print(f"Imagem marcada salva em: {SAIDA}")


if __name__ == "__main__":
    main()
