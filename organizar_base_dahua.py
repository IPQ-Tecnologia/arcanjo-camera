import csv
import json
import shutil
from pathlib import Path


PASTA_CAPTURAS = Path("capturas_dahua")
PASTA_BASE = Path("base_dahua")
ARQUIVO_RESUMO = PASTA_BASE / "resumo.csv"


def valor_json(valor):
    if isinstance(valor, (list, dict)):
        return json.dumps(valor, ensure_ascii=False)

    return valor


def main():
    PASTA_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros = []
    total = 0

    pastas_analise = sorted(
        PASTA_CAPTURAS.glob("analise_*")
    )

    for pasta_analise in pastas_analise:
        arquivos_json = sorted(
            pasta_analise.glob("parte_*.json")
        )

        arquivos_imagem = sorted(
            pasta_analise.glob("parte_*.jpg")
        )

        if not arquivos_json or not arquivos_imagem:
            continue

        arquivo_json = arquivos_json[0]
        arquivo_imagem = arquivos_imagem[0]

        try:
            dados = json.loads(
                arquivo_json.read_text(
                    encoding="utf-8"
                )
            )

        except (OSError, json.JSONDecodeError) as erro:
            print(
                f"Não foi possível ler "
                f"{arquivo_json}: {erro}"
            )
            continue

        eventos = dados.get("Events") or []

        if not eventos:
            continue

        evento = eventos[0]
        dados_evento = evento.get("Data") or {}
        objeto = dados_evento.get("Object") or {}

        codigo = (
            evento.get("Code")
            or "EventoDesconhecido"
        )

        identificador = (
            pasta_analise.name
            .removeprefix("analise_")
            .removesuffix("_pacote")
        )

        pasta_destino = (
            PASTA_BASE
            / codigo
            / identificador
        )

        pasta_destino.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            arquivo_json,
            pasta_destino / "metadados.json",
        )

        shutil.copy2(
            arquivo_imagem,
            pasta_destino / "frame.jpg",
        )

        registro = {
            "identificador": identificador,
            "tipo_evento": codigo,
            "acao_evento": evento.get("Action"),
            "acao_objeto": dados_evento.get("Action"),
            "nome_regra": dados_evento.get("Name"),
            "id_regra": dados_evento.get("RuleID"),
            "id_evento": dados_evento.get("EventID"),
            "id_grupo": dados_evento.get("GroupID"),
            "tipo_objeto": objeto.get("ObjectType"),
            "id_objeto": objeto.get("ObjectID"),
            "bounding_box": valor_json(
                objeto.get("BoundingBox")
            ),
            "centro": valor_json(
                objeto.get("Center")
            ),
            "cor_inferior": valor_json(
                objeto.get("LowerBodyColor")
            ),
            "mascara": objeto.get("HasMask"),
            "velocidade": objeto.get("Speed"),
            "regiao_deteccao": valor_json(
                dados_evento.get("DetectRegion")
            ),
            "horario": dados.get("Time"),
            "resolucao": valor_json(
                dados.get("Resolution")
            ),
            "tipo_imagem": (
                dados.get("Image", [{}])[0].get("Type")
                if dados.get("Image")
                else None
            ),
            "tamanho_imagem": dados.get("Length"),
            "transferencia": dados.get("Transfer"),
            "pasta": str(pasta_destino),
        }

        registros.append(registro)
        total += 1

    if registros:
        with ARQUIVO_RESUMO.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as arquivo_csv:
            escritor = csv.DictWriter(
                arquivo_csv,
                fieldnames=registros[0].keys(),
            )

            escritor.writeheader()
            escritor.writerows(registros)

    print()
    print("===== BASE DAHUA =====")
    print(f"Exemplos organizados: {total}")
    print(f"Pasta: {PASTA_BASE}")
    print(f"Resumo: {ARQUIVO_RESUMO}")


if __name__ == "__main__":
    main()