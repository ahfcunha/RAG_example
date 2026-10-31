import logging
import os
from dotenv import load_dotenv
 
load_dotenv()

PASTA_BASE = os.getenv("PASTA_BASE", "base")   
CAMINHO_DB = os.getenv("CAMINHO_DB", "db")     
 

MODELO_EMBEDDING = os.getenv("MODELO_EMBEDDING", "gemini-embedding-001")
MODELO_LLM = os.getenv("MODELO_LLM", "gemini-3.1-flash-lite")
TEMPERATURA = float(os.getenv("TEMPERATURA", "0"))
 

NUM_RESULTADOS = int(os.getenv("NUM_RESULTADOS", "3"))
LIMIAR_RELEVANCIA = float(os.getenv("LIMIAR_RELEVANCIA", "0.5"))

TAMANHO_CHUNK = 1000
SOBREPOSICAO_CHUNK = 200
 
NIVEL_LOG = os.getenv("NIVEL_LOG", "INFO")
 
 
def configurar_logging() -> None:
    """Define o formato dos logs. Chame uma vez, no início do programa."""
    # define o nível do log, caso venha um valor inválido, utiliza o valor padrão (logging.info)
    nivel = getattr(logging, NIVEL_LOG.upper(), logging.INFO)
    # handler que escreve na tela
    # %(asctime)s: hora
    # %(levelname)-8s: nível
    # %(name)s: nome do logger (de qual módulo veio)
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # alterando o nível dos canais que já vem automaticamente do sistema
    for biblioteca in ("httpx", "httpcore", "openai", "chromadb"):
        logging.getLogger(biblioteca).setLevel(logging.WARNING)