import argparse
import hashlib
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

# quantidade de itens a ser enviada ao banco por vez
TAMANHO_LOTE = 50  

class ErroIngestao(Exception):
    """Erro previsto durante a criação/atualização do banco."""


def carregar_documentos() -> list[Document]:
    """Lê todos os PDFs da pasta base (um Document por página)."""
    if not Path(config.PASTA_BASE).exists():
        raise ErroIngestao(f"Pasta '{config.PASTA_BASE}' não encontrada.")

    loader = PyPDFDirectoryLoader(config.PASTA_BASE)
    documentos = loader.load()
    logger.info("%d páginas carregadas de '%s'", len(documentos), config.PASTA_BASE)
    return documentos


def dividir_chunks(documentos: list[Document]) -> list[Document]:
    """Quebra os documentos em pedaços menores, com sobreposição."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.TAMANHO_CHUNK,
        chunk_overlap=config.SOBREPOSICAO_CHUNK,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documentos)
    logger.info("%d chunks gerados", len(chunks))
    return chunks


def gerar_id(chunk: Document) -> str:
    """Geração de id's determinísticos para garantir a consitência do banco"""
    partes = [
        str(chunk.metadata.get("source", "")),
        str(chunk.metadata.get("page", "")),
        str(chunk.metadata.get("start_index", "")),
        chunk.page_content,
    ]
    # hashlib calcula a impressão digital
    # encode converte o texto para bytes
    # hexdigest mostra o resultado como um texto de 64 caracteres
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()


def mapear_chunks(chunks: list[Document]) -> dict[str, Document]:
    """Chama a função gerar_id para dar a cada chunk um id único"""
    return {gerar_id(chunk): chunk for chunk in chunks}


def em_lotes(itens: list, tamanho: int):
    # (min, tamanho total, valor do passo)
    for inicio in range(0, len(itens), tamanho):
        # entrega uma parte por vez
        yield itens[inicio : inicio + tamanho]

def abrir_banco() -> Chroma:
    return Chroma(
        persist_directory=config.CAMINHO_DB,
        embedding_function= GoogleGenerativeAIEmbeddings(model=config.MODELO_EMBEDDING)
    )


def recriar_banco() -> None:
    """Apaga o banco inteiro (use ao trocar o modelo de embedding)."""
    pasta = Path(config.CAMINHO_DB)
    if pasta.exists():
        shutil.rmtree(pasta)
        logger.warning("Banco '%s' apagado.", pasta)


def sincronizar(db: Chroma, chunks_por_id: dict[str, Document]) -> None:
    """Deixa o banco igual aos documentos atuais: adiciona o novo, remove o obsoleto."""
    # pega os id's do pdf que está sendo lido agora
    ids_atuais = set(chunks_por_id)
    # pega os id's dos documentos que estão no banco
    ids_no_banco = set(db.get(include=[])["ids"])

    # calcula os id's que devem ser adicionados ao banco
    ids_novos = [i for i in chunks_por_id if i not in ids_no_banco]
    # calcula os id's que devem ser retirados do banco
    ids_obsoletos = list(ids_no_banco - ids_atuais)


    for lote in em_lotes(ids_novos, TAMANHO_LOTE):
        db.add_documents([chunks_por_id[i] for i in lote], ids=lote)

    for lote in em_lotes(ids_obsoletos, TAMANHO_LOTE):
        db.delete(ids=lote)


def criar_db(recriar: bool = False) -> None:
    # lê os pdf's da pasta base
    documentos = carregar_documentos()
    # quebra o texto em pedaços
    chunks = dividir_chunks(documentos)

    # Proteção: sem nenhum texto, a sincronização apagaria o banco inteiro.
    if not chunks:
        raise ErroIngestao(
            f"Nenhum texto encontrado em '{config.PASTA_BASE}'. "
            "PDFs escaneados (só imagem) não têm texto extraível. "
            "O banco não foi alterado."
        )

    if recriar:
        # apaga a pasta do banco
        recriar_banco()

    # conecta ao chroma
    db = abrir_banco()
    # sincroniza o arquivo pdf que foi lido com o banco que está aberto
    sincronizar(db, mapear_chunks(chunks))


def main() -> None:
    # argparse permite que quem rode o programa mude seu comportamento através do terminal
    parser = argparse.ArgumentParser(description="Cria/atualiza o banco vetorial.")
    # permite que ao digitar python criar_db.py a opção de recriar receba true
    parser.add_argument(
        "--recriar",
        action="store_true",
        help="apaga o banco e recria do zero (necessário ao trocar o modelo de embedding)",
    )
    # lê e devolve o resultado do terminal, cada palavra após '--' virá um atributo
    args = parser.parse_args()

    config.configurar_logging()
    try:
        criar_db(recriar=args.recriar)
    except ErroIngestao as erro:
        print(f"Erro: {erro}")


if __name__ == "__main__":
    main()