# RAG de Políticas Internas

Sistema de perguntas e respostas sobre documentos PDF usando **RAG** (*Retrieval-Augmented Generation*), construído com **LangChain**, **ChromaDB** e **Google Gemini**.

Em vez de pedir ao modelo que responda "de cabeça", o sistema busca os trechos mais relevantes dos seus documentos e os entrega ao modelo junto com a pergunta. A resposta vem acompanhada das fontes (arquivo e página).

> Projeto de estudo. O documento de exemplo (`base/politica_rh.pdf`) é um manual de RH fictício da empresa "Nexus Tecnologia".

## Como funciona

```
INGESTÃO  (criar_db.py)
PDFs -> texto por página -> chunks (1000 caracteres, sobreposição de 200)
     -> ID por hash do conteúdo -> embeddings (Gemini) -> ChromaDB

CONSULTA  (main.py)
Pergunta -> embedding -> busca dos trechos mais parecidos no ChromaDB
         -> filtro por score de relevância
         -> prompt (regras + contexto + pergunta) -> Gemini (streaming)
         -> resposta + fontes
```

## Funcionalidades

- **Ingestão incremental:** cada chunk recebe um ID calculado a partir do seu conteúdo (hash SHA-256). Rodar a ingestão de novo não gera duplicatas: só o que mudou é adicionado ou removido.
- **Respostas com fontes:** mostra de qual arquivo e página veio a informação.
- **Filtro de relevância:** só trechos acima de um score mínimo chegam ao modelo. Se nada for relevante, o sistema avisa em vez de inventar.
- **Prompt restritivo:** o modelo é instruído a responder apenas com base no contexto e a admitir quando não souber.
- **Streaming:** a resposta aparece aos poucos no terminal.
- **Configuração por `.env`:** modelos, limiar, número de trechos e nível de log podem ser alterados sem editar código.
- **Logging e tratamento de erros:** mensagens claras para problemas previstos (chave ausente, banco inexistente) e log detalhado para os imprevistos.

## Tecnologias

Python, LangChain, ChromaDB, Google Gemini (LLM e embeddings), PyPDF e python-dotenv.

## Estrutura do projeto

```
.
├── base/             # PDFs que serão consultados
├── db/               # banco vetorial (gerado; não versionado)
├── config.py         # configurações centrais e logging
├── criar_db.py       # ingestão: lê PDFs, divide em chunks, gera embeddings
├── erro.py           # exceções do sistema
├── main.py           # consulta: busca, filtra, gera a resposta e mostra as fontes
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior
- Uma chave de API do Gemini, gerada no [Google AI Studio](https://aistudio.google.com/)

## Instalação

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

## Configuração

Copie o arquivo de exemplo e preencha a sua chave:

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/macOS
```

| Variável | Obrigatória | Descrição |
|---|---|---|
| `GOOGLE_API_KEY` | Sim | Chave da API do Gemini |
| `MODELO_LLM` | Não | Modelo que gera as respostas (ex.: `gemini-3.1-flash-lite`) |
| `MODELO_EMBEDDING` | Não | Modelo que transforma texto em vetores |
| `TEMPERATURA` | Não | Aleatoriedade do modelo (0 = respostas mais estáveis) |
| `NUM_RESULTADOS` | Não | Quantos trechos buscar por pergunta |
| `LIMIAR_RELEVANCIA` | Não | Score mínimo para um trecho ser usado |
| `NIVEL_LOG` | Não | `DEBUG`, `INFO`, `WARNING`... |
| `CAMINHO_DB` / `PASTA_BASE` | Não | Pasta do banco vetorial e pasta dos PDFs |

## Uso

**1. Coloque seus PDFs na pasta `base/`.**

**2. Crie ou atualize o banco vetorial:**

```bash
python criar_db.py
```

Pode rodar de novo sempre que os documentos mudarem: o script adiciona o que é novo, remove o que sumiu e ignora o que já está no banco.

Para apagar tudo e recriar do zero (obrigatório ao trocar o modelo de embedding):

```bash
python criar_db.py --recriar
```

**3. Faça perguntas:**

```bash
python main.py
```

Exemplo (saída ilustrativa):

```
Escreva sua pergunta: Posso trabalhar home office?

Sim. A empresa adota o modelo híbrido: é possível trabalhar remotamente até
3 dias por semana, com presença obrigatória no escritório às terças e quintas.
É preciso ter mais de 90 dias de empresa e não ter advertências nos últimos
6 meses.

Fontes:
 - politica_rh.pdf (página 1)
```

## Ajustando a qualidade das respostas

O score de relevância **depende do modelo de embedding**: um limiar que funciona com um modelo pode barrar tudo com outro. Para calibrar:

1. Coloque `NIVEL_LOG=DEBUG` no `.env` para ver o score de cada trecho encontrado.
2. Faça uma pergunta cuja resposta está nos documentos e outra totalmente fora do assunto.
3. Escolha um `LIMIAR_RELEVANCIA` entre os scores das duas.

## Solução de problemas

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| "Não consegui encontrar informação relevante" para perguntas que deveriam funcionar | Limiar alto demais para o seu modelo de embedding | Calibre o `LIMIAR_RELEVANCIA` (seção acima) |
| `404 NOT_FOUND ... modelo não disponível` | O Google aposentou ou renomeou o modelo | Atualize `MODELO_LLM` com um nome atual do Google AI Studio |
| `WinError 10053` / `ReadError` durante a resposta | Conexão cortada na sua máquina ou rede (antivírus, firewall, VPN, proxy) | Teste sem VPN, em outra rede, ou troque `.stream` por `.invoke` em `main.py` |
| Resultados estranhos depois de trocar o modelo de embedding | Vetores de modelos diferentes não são comparáveis | `python criar_db.py --recriar` |
| `GOOGLE_API_KEY não encontrada` | Falta o `.env` ou a variável | Crie o `.env` a partir do `.env.example` |
| `Banco 'db' não encontrado` | A ingestão ainda não foi executada | Rode `python criar_db.py` |
