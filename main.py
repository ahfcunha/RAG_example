import logging
import os
from typing import Iterator, Optional

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dataclasses import dataclass
from pathlib import Path

import config
import erro

# pega o nome do módulo para o log
# o nível de log é o mesmo do sistema
logger = logging.getLogger(__name__)


SEPARADOR_TRECHOS = "\n\n----\n\n"


PROMPT_SISTEMA = (
    "Você é um assistente que responde perguntas usando EXCLUSIVAMENTE o "
    "contexto fornecido.\n"
    "Regras:\n"
    "- Use apenas as informações presentes no contexto.\n"
    "- Se a resposta não estiver no contexto, diga que não encontrou essa "
    "informação nos documentos. Não invente nada.\n"
    "- Responda em português, de forma clara e objetiva."
)
PROMPT_USUARIO = "Contexto:\n{contexto}\n\nPergunta: {pergunta}"

MENSAGEM_SEM_RESULTADO = "Não consegui encontrar informação relevante na base."


@dataclass
class Contexto:
    texto: str
    fontes: list[str]


@dataclass
class Resposta:
    fontes: list[str]
    pedacos: Iterator[str]


def verificar_ambiente() -> None:
    "Confere se o que é necessário para rodar o sistema esta configurado"
    # verifica a presença da api key no .env
    if not os.getenv("GOOGLE_API_KEY"):
        # raise é responsável por lançar a exceção
        #  que será capturada por um try/catch mais para frente
        raise erro.ChaveApiAusente(
            "GOOGLE_API_KEY não encontrada"
        )

    # verifica se a pasta db esta criada na raiz do projeto
    if not Path(config.CAMINHO_DB).exists():
        raise erro.BancoNaoEncontrado(
            f"Banco '{config.CAMINHO_DB}' não encontrado"
        )


def carregar_banco() -> Chroma:
    """Abre o banco vetorial que foi criado pelo criar_db.py"""
    funcao_embedding = GoogleGenerativeAIEmbeddings(model=config.MODELO_EMBEDDING)
    return Chroma(persist_directory=config.CAMINHO_DB, embedding_function=funcao_embedding)


def buscar_trechos(db: Chroma, pergunta: str) -> list:
    """Responsável por retornar os chunks que contém maior proximidade com a pergunta feita pelo usuário (Retrieval).
    Devovolve uma lista de tuplas (documento, score)"""
    # k refere-se ao número de chunks que vai ser retornado pela função
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=config.NUM_RESULTADOS)
    for documento, score in resultados:
        logger.debug("score=%.3f | %r", score, documento.page_content[:60])
    return resultados


def verificar_relevancia(resultados: list) -> list:
    """Confere se o score é maior ou igual ao que foi determinado por LIMIAR_RELEVANCIA."""
    relevantes = [documento for documento, score in resultados if score >= config.LIMIAR_RELEVANCIA]
    
    logger.info(
        "%d de %d trechos passaram do limiar %.2f",
        len(relevantes), len(resultados), config.LIMIAR_RELEVANCIA,
    )
    return relevantes


def montar_contexto(resultados: list) -> str:
    """Gera uma string com os trechos/chunks que foram retornados"""
    # page_content retorna o texto puro do documento
    textos = [documento.page_content for documento in resultados]
    return SEPARADOR_TRECHOS.join(textos)


def formatar_fonte(documento) -> str:
    "Transforma os metadados do trecho em fontes que indicam a posição no arquivo"
    metadados = documento.metadata
    # Path transforma texto em objeto de caminho de arquivo
    # .name retorna so o final desse caminho
    arquivo = Path(metadados.get("source", "desconhecido")).name
    pagina = metadados.get("page")
    if pagina is None:
        return arquivo
    # páginas começam em 0
    return f"{arquivo} (página {pagina + 1})"


def listar_fontes(resultados: list) -> list[str]:
    "Lista os trechos da onde foram tiradas as respostas"
    fontes = [formatar_fonte(documento) for documento in resultados]
    # truque que transforma os itens da lista em chave de um dicionário, com valor None
    # e depois faz um list() que vai conter apenas as chaves do dicionário
    # impedindo que hajam duplicatas
    return list(dict.fromkeys(fontes))


def recuperar(pergunta: str) -> Optional[Contexto]:
    """Busca contextos relevantes para a pergunta"""
    db = carregar_banco()
    resultados = verificar_relevancia(buscar_trechos(db, pergunta))
    if not resultados:
        return None
    return Contexto(
        texto=montar_contexto(resultados),
        fontes=listar_fontes(resultados)
    )


def criar_cadeia():
    """prompt -> modelo -> texto, tudo encadeado com o operador |"""
    # modelos recebem uma lista de mensagens, cada uma com um papel
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SISTEMA),
        ("human", PROMPT_USUARIO)
    ])
    modelo = ChatGoogleGenerativeAI(model=config.MODELO_LLM, temperature=config.TEMPERATURA)
    # o retorno de um é diretamente enviado para o outro
    # StrOutputParser pega o texto referente a resposta do modelo
    return prompt | modelo | StrOutputParser()


def responder(pergunta: str) -> Resposta:
    verificar_ambiente()

    contexto = recuperar(pergunta)
    if contexto is None:
        return Resposta(fontes=[], pedacos=iter([MENSAGEM_SEM_RESULTADO]))

    logger.info("Chamando o modelo %s", config.MODELO_LLM)
    
    pedacos = criar_cadeia().stream(
        {"pergunta": pergunta, "contexto": contexto.texto}
    )
    return Resposta(fontes=contexto.fontes, pedacos=pedacos)


def main():
    
    config.configurar_logging()

    pergunta = input("Escreva sua pergunta: ").strip()
    if not pergunta:
        print("Você não digitou nada")
        return

    try:
        resposta = responder(pergunta)
        for pedaco in resposta.pedacos:
            print(pedaco, end="", flush=True)
        print()
        if resposta.fontes:
            print("\nFontes:")
            for fonte in resposta.fontes:
                print(f" - {fonte}")
    except erro.ErroRAG as falha:
        print(f"Erro: {falha}")
    except Exception:
        # erros imprevistos
        print()
        logger.exception("Falha")
        print("Ocorreu um erro inesperado, tente novamente")


if __name__ == "__main__":
    main()