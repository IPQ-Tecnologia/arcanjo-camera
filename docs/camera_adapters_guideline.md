# Guideline de adapters de câmera

## Objetivo

Cada fabricante deve converter seu formato proprietário para
`CameraEvent`, sem adicionar regras específicas ao pipeline.

## Contrato obrigatório

Todo adapter deve herdar de `CameraAdapter` e implementar:

- `consegue_processar(content_type, body)`
- `normalizar(pacote, body)`

## Responsabilidades do adapter

- reconhecer somente pacotes do próprio fabricante;
- extrair metadados, imagem e bounding boxes;
- converter coordenadas para pixels;
- salvar imagem original e marcada;
- devolver um `CameraEvent`;
- preservar informações proprietárias em `dados_extras`.

## Responsabilidades do pipeline

- escolher o adapter pela factory;
- ignorar eventos sem imagem ou sem boxes válidas;
- rastrear pessoas;
- enviar eventos ao painel;
- publicar eventos normalizados no Kafka.

## Regras de implementação

- usar type hints;
- usar nomes claros;
- evitar lógica de fabricante dentro do pipeline;
- não registrar adapters fora da factory;
- não alterar o schema sem versionamento;
- testar JSON direto e pacote multipart;
- manter compatibilidade com rotas existentes;
- registrar contexto suficiente nos logs;
- não armazenar senhas ou credenciais no código.

## Checklist para novo fabricante

1. Criar `<fabricante>.py`.
2. Implementar `CameraAdapter`.
3. Registrar na `CameraAdapterFactory`.
4. Testar reconhecimento.
5. Testar normalização sem imagem.
6. Testar normalização com imagem.
7. Validar bounding box em pixels.
8. Validar execução no pipeline.
9. Validar publicação no Kafka.
