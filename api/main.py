import os
import re
import json
import secrets
import logging
import sqlparse
import threading
from contextlib import asynccontextmanager
from typing import Optional

import requests as http_requests
from fastapi import FastAPI, Header, HTTPException, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis as redis_lib
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# B2: nível de log configurável via variável de ambiente
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIGURAÇÕES (via .env)
# -------------------------------
OLLAMA_BASE_URL   = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen:7b")
REDIS_HOST        = os.getenv("REDIS_HOST", "redis")
REDIS_PORT        = int(os.getenv("REDIS_PORT", "6379"))
CORS_ORIGINS      = os.getenv("CORS_ORIGINS", "*")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "postgres-replica")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "ai_database")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "ai_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
API_KEY           = os.getenv("API_KEY")
RATE_LIMIT        = int(os.getenv("RATE_LIMIT", "5"))
CACHE_TTL         = int(os.getenv("CACHE_TTL", "3600"))
MAX_ROWS          = int(os.getenv("MAX_ROWS", "100"))
MAX_HISTORY       = int(os.getenv("MAX_HISTORY", "5"))
NL_TIMEOUT        = int(os.getenv("NL_TIMEOUT", "60"))

if not POSTGRES_PASSWORD or not API_KEY:
    raise RuntimeError("POSTGRES_PASSWORD e API_KEY são obrigatórios. Configure o arquivo .env.")

DB_URI = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Objetos globais inicializados no startup
cache: redis_lib.Redis = None
engine = None
sql_chain = None
nl_chain = None
db_instance: SQLDatabase = None
llm_lock = threading.Lock()  # garante 1 chamada ao Ollama por vez


# -------------------------------
# STARTUP / SHUTDOWN
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, engine, sql_chain, nl_chain, db_instance

    logger.info("Iniciando conexões...")

    # M8: timeouts nas conexões Redis — falha silenciosa se indisponível
    try:
        cache = redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5,
        )
        cache.ping()
        logger.info("Redis conectado.")
    except Exception as e:
        logger.warning(f"Redis indisponível: {e}. Cache e rate limiting desativados.")
        cache = None

    # M8: timeout na conexão PostgreSQL
    engine = create_engine(
        DB_URI,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 5},
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL conectado.")

    db_instance = SQLDatabase.from_uri(DB_URI)
    tables = db_instance.get_usable_table_names()

    # Item 9: alerta no startup se banco não tiver tabelas
    if not tables:
        logger.warning("Nenhuma tabela encontrada no banco. O modelo pode gerar SQL incorreto.")
    else:
        logger.info(f"Tabelas disponíveis: {tables}")

    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_predict=200,
        temperature=0,
        num_ctx=2048,
        timeout=240,
    )

    sql_prompt = PromptTemplate.from_template(
        "PostgreSQL expert. Respond with ONE SELECT only, no explanation, no markdown, no semicolon.\n\n"
        "Schema:\n{table_info}\n\n"
        "Column ownership (use ONLY these):\n"
        "- clientes: id, nome, email, cidade, estado, criado_em\n"
        "- produtos: id, nome, preco, estoque, categoria_id, ativo, criado_em\n"
        "- categorias: id, nome\n"
        "- pedidos: id, cliente_id, status, total, criado_em\n"
        "- itens_pedido: id, pedido_id, produto_id, quantidade, preco_unitario\n\n"
        "Relations:\n"
        "- produtos.categoria_id = categorias.id\n"
        "- pedidos.cliente_id = clientes.id\n"
        "- itens_pedido.pedido_id = pedidos.id\n"
        "- itens_pedido.produto_id = produtos.id\n\n"
        "Examples:\n"
        "Q: total de vendas por estado\n"
        "A: SELECT c.estado, SUM(p.total) AS total_vendas FROM clientes c JOIN pedidos p ON c.id = p.cliente_id GROUP BY c.estado ORDER BY total_vendas DESC LIMIT 10\n\n"
        "Q: 5 produtos mais vendidos\n"
        "A: SELECT pr.nome, SUM(ip.quantidade) AS total FROM produtos pr JOIN itens_pedido ip ON pr.id = ip.produto_id GROUP BY pr.nome ORDER BY total DESC LIMIT 5\n\n"
        "Q: clientes com mais pedidos\n"
        "A: SELECT c.nome, COUNT(p.id) AS qtd FROM clientes c JOIN pedidos p ON c.id = p.cliente_id GROUP BY c.nome ORDER BY qtd DESC LIMIT 10\n\n"
        "Q: faturamento por categoria\n"
        "A: SELECT cat.nome, SUM(ip.quantidade * ip.preco_unitario) AS faturamento FROM categorias cat JOIN produtos pr ON cat.id = pr.categoria_id JOIN itens_pedido ip ON pr.id = ip.produto_id GROUP BY cat.nome ORDER BY faturamento DESC LIMIT 10\n\n"
        "Rules: simple direct SELECT, no subqueries, no CTEs, no WITH clauses, no quoted identifiers, lowercase aliases only, LIMIT {top_k}.\n"
        "Question: {input}\nSQL:"
    )
    sql_chain = create_sql_query_chain(llm, db_instance, prompt=sql_prompt)

    nl_llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_predict=80,
        temperature=0,
        num_ctx=512,
        timeout=60,
    )
    nl_prompt = PromptTemplate.from_template(
        "Dados: {result} ({row_count} linhas)\n"
        "Responda em 1-2 frases em português: {question}"
    )
    nl_chain = nl_prompt | nl_llm | StrOutputParser()

    logger.info("Aplicação pronta.")
    yield
    logger.info("Aplicação encerrada.")


app = FastAPI(title="AI SQL Engine", version="3.0", lifespan=lifespan)

# CORS — origens configuráveis via CORS_ORIGINS (separe por vírgula em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# MODELOS
# -------------------------------
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None


# -------------------------------
# AUTENTICAÇÃO
# -------------------------------
def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------------
# RATE LIMITING (via Redis)
# -------------------------------
def check_rate_limit(ip: str):
    if cache is None:
        return
    key = f"rate:{ip}"
    count = cache.incr(key)
    if count == 1:
        cache.expire(key, 60)
    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# -------------------------------
# VALIDAÇÃO E LIMPEZA DE SQL
# -------------------------------
FORBIDDEN = {"delete", "update", "insert", "drop", "alter", "truncate", "create", "grant", "revoke"}


# B3: validação com sqlparse — detecta múltiplos statements e comentários
def validate_sql(sql: str):
    statements = [s for s in sqlparse.parse(sql) if s.get_type() is not None]

    if not statements:
        raise HTTPException(status_code=400, detail="SQL inválido ou não reconhecido")

    if len(statements) > 1:
        raise HTTPException(status_code=400, detail="Múltiplos statements não são permitidos")

    stmt_type = statements[0].get_type()
    if stmt_type != "SELECT":
        raise HTTPException(status_code=400, detail="Apenas SELECT é permitido")

    # Verifica tokens individuais (detecta variantes com comentários)
    tokens = {str(t).lower().strip() for t in statements[0].flatten() if str(t).strip()}
    blocked = tokens & FORBIDDEN
    if blocked:
        raise HTTPException(status_code=400, detail=f"SQL bloqueado: contém {blocked}")


def clean_sql(raw: str) -> str:
    """Remove markdown e texto extra que o LLM pode gerar ao redor do SQL."""
    raw = raw.strip()

    # Remove blocos de código markdown
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("sql"):
                part = part[3:].strip()
            if part.lower().startswith("select"):
                raw = part.strip()
                break

    # Extrai apenas o primeiro SELECT, ignorando tudo após o primeiro ponto-e-vírgula
    lines = raw.splitlines()
    sql_lines = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if not collecting and stripped.lower().startswith("select"):
            collecting = True
        if collecting:
            # Para ao encontrar linha vazia após ter coletado SQL, ou linha que parece comentário/texto
            if sql_lines and not stripped:
                break
            if stripped.startswith("--") and sql_lines:
                break
            sql_lines.append(line)

    result = "\n".join(sql_lines).strip() if sql_lines else raw

    # Remove ponto-e-vírgula e tudo que vier depois (segundo statement)
    semi = result.find(";")
    if semi != -1:
        result = result[:semi].strip()

    # Remove aspas duplas de aliases (causam problemas no ORDER BY)
    result = re.sub(r'AS\s+"([^"]+)"', lambda m: f'AS {m.group(1).lower()}', result)

    return result


def inject_limit(sql: str, limit: int) -> str:
    """Injeta LIMIT no SQL se não houver um já."""
    if not re.search(r'\blimit\b', sql, re.IGNORECASE):
        sql = sql.rstrip(';').rstrip()
        return f"{sql} LIMIT {limit}"
    return sql


# -------------------------------
# HISTÓRICO DE CONVERSA
# -------------------------------
def get_history(session_id: str) -> list:
    if cache is None:
        return []
    raw = cache.lrange(f"history:{session_id}", 0, MAX_HISTORY - 1)
    return [json.loads(item) for item in raw]


def save_history(session_id: str, question: str, row_count: int):
    if cache is None:
        return
    key = f"history:{session_id}"
    # C4: não persiste o SQL no histórico para evitar prompt injection
    entry = json.dumps({"q": question, "rows": row_count})
    cache.lpush(key, entry)
    cache.ltrim(key, 0, MAX_HISTORY - 1)
    cache.expire(key, 3600)


def build_question_with_history(question: str, session_id: Optional[str]) -> str:
    if not session_id:
        return question
    history = get_history(session_id)
    if not history:
        return question
    lines = ["Histórico recente da conversa:"]
    for item in reversed(history):
        # C4: apenas pergunta e contagem de linhas — sem SQL para evitar prompt injection
        lines.append(f'- Pergunta: "{item["q"]}" → {item["rows"]} linhas retornadas')
    lines.append(f"\nPergunta atual: {question}")
    return "\n".join(lines)


# -------------------------------
# RETRY NO OLLAMA
# -------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_sql_chain(question_with_context: str) -> str:
    return sql_chain.invoke({"question": question_with_context})


# -------------------------------
# ENDPOINTS
# -------------------------------

# M7: health check real com HTTP 503 quando degradado
@app.get("/health", summary="Health check das dependências")
def health(response: Response):
    status = {
        "version": "3.0",
        "redis": False,
        "postgres": False,
        "ollama": False,
    }
    try:
        if cache is not None:
            cache.ping()
            status["redis"] = True
    except Exception:
        pass
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = True
    except Exception:
        pass
    try:
        r = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        status["ollama"] = r.status_code == 200
    except Exception:
        pass

    all_ok = all([status["redis"], status["postgres"], status["ollama"]])
    status["status"] = "ok" if all_ok else "degraded"

    # M7: retorna 503 quando alguma dependência está indisponível
    if not all_ok:
        response.status_code = 503

    return status


@app.get("/schema", summary="Tabelas e colunas disponíveis no banco")
def get_schema(api_key: str = Depends(verify_api_key)):
    tables = db_instance.get_usable_table_names()
    schema = {table: db_instance.get_table_info([table]) for table in tables}
    return {"tables": tables, "schema": schema}


@app.post("/ask-db", summary="Pergunta para o banco de dados via IA")
def ask_db(data: Question, request: Request, api_key: str = Depends(verify_api_key)):
    ip = request.client.host
    check_rate_limit(ip)

    question = data.question.strip()

    # B4: cache key inclui session_id para evitar colisão entre sessões diferentes
    session_suffix = f":{data.session_id}" if data.session_id else ""
    cache_key = f"q:{question.lower()}{session_suffix}"
    logger.info(f"[{ip}] Pergunta: {question}")

    # 1. Verifica cache
    if cache is not None:
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"[{ip}] Cache hit.")
            return {"response": json.loads(cached), "source": "cache"}

    # 2. Monta pergunta com histórico de conversa
    question_with_context = build_question_with_history(question, data.session_id)

    # 3. Gera SQL com LangChain (lock garante 1 chamada ao Ollama por vez)
    with llm_lock:
        try:
            raw_sql = invoke_sql_chain(question_with_context)
            sql_query = clean_sql(raw_sql)
            logger.debug(f"[{ip}] SQL gerado: {sql_query}")
        except Exception as e:
            logger.error(f"[{ip}] Erro ao gerar SQL após retries: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao gerar SQL: {e}")

    # 4. Validação de segurança com sqlparse
    validate_sql(sql_query)

    # 5. Injeta LIMIT se necessário
    sql_query = inject_limit(sql_query, MAX_ROWS)

    # 6. Executa com transação read-only real
    try:
        with engine.begin() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            result = conn.execute(text(sql_query))
            rows = [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error(f"[{ip}] Erro ao executar SQL: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar SQL: {e}")

    # 7. Gera resposta em linguagem natural (NL_TIMEOUT=0 desabilita)
    nl_answer = None
    if NL_TIMEOUT > 0:
        try:
            nl_answer = nl_chain.invoke({
                "question": question,
                "sql": sql_query,
                "row_count": len(rows),
                "result": str(rows[:5]),
            })
        except Exception as e:
            logger.warning(f"[{ip}] Falha na resposta em linguagem natural: {e}")

    # 8. Salva histórico da sessão (sem SQL — C4)
    if data.session_id:
        save_history(data.session_id, question, len(rows))

    # 9. Formata e salva no cache
    formatted = {
        "answer": nl_answer,
        "sql": sql_query,
        "row_count": len(rows),
        "rows": rows,
    }
    if cache is not None:
        cache.setex(cache_key, CACHE_TTL, json.dumps(formatted, default=str))

    return {"response": formatted, "source": "model"}
