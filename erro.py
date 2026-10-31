class ErroRAG(Exception):
    "Classe que configura os erros previstos do sistema"

class ChaveApiAusente(ErroRAG):
    "Exceção lançada quando a chave de api não esta configurada, herda de ErroRAG"
    pass

class BancoNaoEncontrado(ErroRAG):
    "Exceção lançada quando o banco não foi criado, herda de ErroRAG"
    pass