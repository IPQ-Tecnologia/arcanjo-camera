import argparse
import json
from pathlib import Path
from typing import Any


PASTA_CAPTURAS = Path("capturas_dahua")

CHAVES_INTERESSANTES = (
    "event",
    "code",
    "type",
    "action",
    "rule",
    "channel",
    "time",
    "utc",
    "date",
    "object",
    "class",
    "direction",
    "region",
    "name",
    "ack",
)


def localizar_pacote_mais_recente() -> Path:
    arquivos = sorted(
        PASTA_CAPTURAS.glob("*_pacote.bin"),
        key=lambda caminho: caminho.stat().st_mtime,
        reverse=True,
    )

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum arquivo *_pacote.bin foi encontrado "
            "na pasta capturas_dahua."
        )

    return arquivos[0]


def interpretar_headers(
    dados: bytes,
) -> dict[str, str]:
    headers: dict[str, str] = {}

    texto = dados.decode(
        "iso-8859-1",
        errors="replace",
    )

    for linha in texto.splitlines():
        if ":" not in linha:
            continue

        nome, valor = linha.split(
            ":",
            1,
        )

        headers[
            nome.strip().lower()
        ] = valor.strip()

    return headers


def separar_headers_conteudo(
    bloco: bytes,
) -> tuple[bytes, bytes]:
    separadores = (
        b"\r\n\r\n",
        b"\n\n",
    )

    for separador in separadores:
        if separador in bloco:
            return bloco.split(
                separador,
                1,
            )

    return b"", bloco


def remover_quebra_final(
    conteudo: bytes,
) -> bytes:
    if conteudo.endswith(b"\r\n"):
        return conteudo[:-2]

    if conteudo.endswith(b"\n"):
        return conteudo[:-1]

    return conteudo


def dividir_multipart(
    body: bytes,
) -> list[tuple[dict[str, str], bytes]]:
    primeira_linha = body.splitlines()[0].strip()

    if not primeira_linha.startswith(b"--"):
        raise ValueError(
            "Não foi possível localizar o boundary "
            "na primeira linha do pacote."
        )

    marcador_boundary = primeira_linha

    blocos = body.split(
        marcador_boundary
    )

    partes: list[
        tuple[dict[str, str], bytes]
    ] = []

    for bloco in blocos:
        bloco = bloco.lstrip(
            b"\r\n"
        )

        if not bloco:
            continue

        if bloco.startswith(b"--"):
            continue

        if bloco.endswith(b"--\r\n"):
            bloco = bloco[:-4]
        elif bloco.endswith(b"--"):
            bloco = bloco[:-2]

        headers_bytes, conteudo = (
            separar_headers_conteudo(
                bloco
            )
        )

        conteudo = remover_quebra_final(
            conteudo
        )

        headers = interpretar_headers(
            headers_bytes
        )

        if not headers and not conteudo:
            continue

        partes.append(
            (
                headers,
                conteudo,
            )
        )

    return partes


def tentar_extrair_json(
    conteudo: bytes,
) -> Any | None:
    texto = conteudo.decode(
        "utf-8",
        errors="replace",
    ).strip()

    inicio_objeto = texto.find("{")
    fim_objeto = texto.rfind("}")

    if (
        inicio_objeto >= 0
        and fim_objeto > inicio_objeto
    ):
        candidato = texto[
            inicio_objeto:
            fim_objeto + 1
        ]

        try:
            return json.loads(
                candidato
            )
        except json.JSONDecodeError:
            pass

    inicio_lista = texto.find("[")
    fim_lista = texto.rfind("]")

    if (
        inicio_lista >= 0
        and fim_lista > inicio_lista
    ):
        candidato = texto[
            inicio_lista:
            fim_lista + 1
        ]

        try:
            return json.loads(
                candidato
            )
        except json.JSONDecodeError:
            pass

    return None


def formatar_valor(
    valor: Any,
) -> str:
    if isinstance(
        valor,
        (dict, list),
    ):
        texto = json.dumps(
            valor,
            ensure_ascii=False,
        )
    else:
        texto = str(valor)

    if len(texto) > 300:
        return texto[:300] + "..."

    return texto


def mostrar_campos_interessantes(
    valor: Any,
    caminho: str = "$",
) -> None:
    if isinstance(valor, dict):
        for chave, item in valor.items():
            caminho_atual = (
                f"{caminho}.{chave}"
            )

            chave_minuscula = (
                str(chave).lower()
            )

            if any(
                termo in chave_minuscula
                for termo
                in CHAVES_INTERESSANTES
            ):
                print(
                    f"  {caminho_atual} = "
                    f"{formatar_valor(item)}"
                )

            mostrar_campos_interessantes(
                item,
                caminho_atual,
            )

    elif isinstance(valor, list):
        for indice, item in enumerate(
            valor
        ):
            mostrar_campos_interessantes(
                item,
                f"{caminho}[{indice}]",
            )


def escolher_extensao(
    content_type: str,
    conteudo: bytes,
    possui_json: bool,
) -> str:
    tipo = content_type.lower()

    if (
        "image/jpeg" in tipo
        or conteudo.startswith(
            b"\xff\xd8\xff"
        )
    ):
        return ".jpg"

    if (
        "image/png" in tipo
        or conteudo.startswith(
            b"\x89PNG"
        )
    ):
        return ".png"

    if possui_json:
        return ".json"

    if (
        tipo.startswith("text/")
        or "xml" in tipo
    ):
        return ".txt"

    return ".bin"


def analisar_pacote(
    caminho_pacote: Path,
) -> None:
    body = caminho_pacote.read_bytes()

    pasta_saida = (
        PASTA_CAPTURAS
        / f"analise_{caminho_pacote.stem}"
    )

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n===== PACOTE DAHUA ====="
    )
    print(
        f"Arquivo: {caminho_pacote}"
    )
    print(
        f"Tamanho: {len(body)} bytes"
    )
    print(
        f"Saída: {pasta_saida}"
    )

    partes = dividir_multipart(
        body
    )

    print(
        f"Quantidade de partes: "
        f"{len(partes)}"
    )

    jsons_encontrados = 0
    imagens_encontradas = 0

    for indice, (
        headers,
        conteudo,
    ) in enumerate(
        partes,
        start=1,
    ):
        content_type = headers.get(
            "content-type",
            "desconhecido",
        )

        content_length = headers.get(
            "content-length",
            "não informado",
        )

        dados_json = tentar_extrair_json(
            conteudo
        )

        extensao = escolher_extensao(
            content_type=content_type,
            conteudo=conteudo,
            possui_json=(
                dados_json is not None
            ),
        )

        caminho_parte = (
            pasta_saida
            / f"parte_{indice:02d}"
            f"{extensao}"
        )

        if dados_json is not None:
            jsons_encontrados += 1

            caminho_parte.write_text(
                json.dumps(
                    dados_json,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            caminho_parte.write_bytes(
                conteudo
            )

        if extensao in {
            ".jpg",
            ".png",
        }:
            imagens_encontradas += 1

        print(
            "\n--------------------------"
        )
        print(
            f"Parte {indice}"
        )
        print(
            f"Content-Type: {content_type}"
        )
        print(
            f"Content-Length informado: "
            f"{content_length}"
        )
        print(
            f"Tamanho extraído: "
            f"{len(conteudo)} bytes"
        )
        print(
            f"Arquivo salvo: "
            f"{caminho_parte}"
        )

        if dados_json is not None:
            print(
                "\nChaves principais do JSON:"
            )

            if isinstance(
                dados_json,
                dict,
            ):
                for chave in (
                    dados_json.keys()
                ):
                    print(
                        f"  - {chave}"
                    )

            print(
                "\nCampos relacionados "
                "ao evento:"
            )

            mostrar_campos_interessantes(
                dados_json
            )

    print(
        "\n===== RESUMO ====="
    )
    print(
        f"Partes encontradas: "
        f"{len(partes)}"
    )
    print(
        f"JSONs encontrados: "
        f"{jsons_encontrados}"
    )
    print(
        f"Imagens encontradas: "
        f"{imagens_encontradas}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa um pacote multipart "
            "recebido da câmera Dahua."
        )
    )

    parser.add_argument(
        "arquivo",
        nargs="?",
        help=(
            "Caminho do pacote. "
            "Quando omitido, usa o mais recente."
        ),
    )

    argumentos = parser.parse_args()

    if argumentos.arquivo:
        caminho = Path(
            argumentos.arquivo
        )
    else:
        caminho = (
            localizar_pacote_mais_recente()
        )

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    analisar_pacote(
        caminho
    )


if __name__ == "__main__":
    main()
