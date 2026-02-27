import os
import re
import json
import asyncio
import secrets
import logging
import sqlparse
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
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
OLLAMA_BASE_URL   = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen:7b")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
REDIS_HOST        = os.getenv("REDIS_HOST", "redis")
REDIS_PORT        = int(os.getenv("REDIS_PORT", "6379"))
CORS_ORIGINS      = os.getenv("CORS_ORIGINS", "*")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "ai_database")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "ai_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
API_KEY           = os.getenv("API_KEY")
RATE_LIMIT        = int(os.getenv("RATE_LIMIT", "5"))
CACHE_TTL         = int(os.getenv("CACHE_TTL", "3600"))
MAX_ROWS          = int(os.getenv("MAX_ROWS", "100"))
MAX_HISTORY       = int(os.getenv("MAX_HISTORY", "5"))
NL_TIMEOUT        = int(os.getenv("NL_TIMEOUT", "0"))

if not POSTGRES_PASSWORD or not API_KEY:
    raise RuntimeError("POSTGRES_PASSWORD e API_KEY são obrigatórios.")

DB_URI = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Schema compacto — injeta ~80 tokens no prompt vs ~500 do LangChain SQLDatabase
COMPACT_SCHEMA = (
    "categorias(id,nome) | "
    "clientes(id,nome,email,cidade,estado,criado_em) | "
    "produtos(id,nome,preco,estoque,categoria_id,ativo,criado_em) | "
    "pedidos(id,cliente_id,status,total,criado_em) | "
    "itens_pedido(id,pedido_id,produto_id,quantidade,preco_unitario) | "
    "FK: produtos.categoria_id=categorias.id, pedidos.cliente_id=clientes.id, "
    "itens_pedido.pedido_id=pedidos.id, itens_pedido.produto_id=produtos.id"
)

# Globals
cache: redis_lib.Redis = None
engine = None
llm = None
db_instance: SQLDatabase = None


# -------------------------------
# STARTUP / SHUTDOWN
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, engine, llm, db_instance

    logger.info("Iniciando conexões...")

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
    if not tables:
        logger.warning("Nenhuma tabela encontrada no banco.")
    else:
        logger.info(f"Tabelas disponíveis: {tables}")

    if GROQ_API_KEY:
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm = ChatGroq(
            model=groq_model,
            temperature=0,
            max_tokens=200,
            groq_api_key=GROQ_API_KEY,
        )
        logger.info(f"LLM: Groq {groq_model}")
    else:
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_predict=80,   # SQL raramente passa de 60 tokens
            temperature=0,
            num_ctx=384,      # prompt compacto + margem para precisão
            timeout=120,
        )
        logger.info(f"LLM: Ollama {OLLAMA_MODEL} (num_ctx=384, num_predict=80)")

    logger.info("Aplicação pronta.")
    yield
    logger.info("Aplicação encerrada.")


app = FastAPI(title="AI SQL Engine", version="3.1", lifespan=lifespan)

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
# RATE LIMITING
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


def validate_sql(sql: str):
    statements = [s for s in sqlparse.parse(sql) if s.get_type() is not None]
    if not statements:
        raise HTTPException(status_code=400, detail="SQL inválido ou não reconhecido")
    if len(statements) > 1:
        raise HTTPException(status_code=400, detail="Múltiplos statements não são permitidos")
    if statements[0].get_type() != "SELECT":
        raise HTTPException(status_code=400, detail="Apenas SELECT é permitido")
    tokens = {str(t).lower().strip() for t in statements[0].flatten() if str(t).strip()}
    blocked = tokens & FORBIDDEN
    if blocked:
        raise HTTPException(status_code=400, detail=f"SQL bloqueado: contém {blocked}")


def clean_sql(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("sql"):
                part = part[3:].strip()
            if part.lower().startswith("select"):
                raw = part.strip()
                break
    lines = raw.splitlines()
    sql_lines = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if not collecting and stripped.lower().startswith("select"):
            collecting = True
        if collecting:
            if sql_lines and not stripped:
                break
            if stripped.startswith("--") and sql_lines:
                break
            sql_lines.append(line)
    result = "\n".join(sql_lines).strip() if sql_lines else raw
    semi = result.find(";")
    if semi != -1:
        result = result[:semi].strip()
    result = re.sub(r'AS\s+"([^"]+)"', lambda m: f'AS {m.group(1).lower()}', result)
    # Normaliza duplos underscores em nomes de colunas (alucinação comum de LLMs)
    result = re.sub(r'(?<=[a-z])__(?=[a-z])', '_', result)
    # Converte TOP N (SQL Server) para LIMIT N (PostgreSQL)
    result = re.sub(r'\bSELECT\s+TOP\s+(\d+)\s+', r'SELECT ', result, flags=re.IGNORECASE)
    top_match = re.search(r'\bSELECT\s+TOP\s+(\d+)\b', result, re.IGNORECASE)
    if top_match:
        n = top_match.group(1)
        result = re.sub(r'\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', result, flags=re.IGNORECASE)
        if not re.search(r'\blimit\b', result, re.IGNORECASE):
            result = f"{result.rstrip()} LIMIT {n}"
    return result


def inject_limit(sql: str, limit: int) -> str:
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
    lines = ["Histórico:"]
    for item in reversed(history):
        lines.append(f'- "{item["q"]}" → {item["rows"]} linhas')
    lines.append(f"Pergunta: {question}")
    return "\n".join(lines)


# -------------------------------
# CHAMADA AO LLM
# Prompt compacto: ~120 tokens (vs ~500 com SQLDatabase)
# -------------------------------
def _build_prompt(question: str) -> str:
    return (
        f"PostgreSQL expert. ONE SELECT only. No explanation, no markdown, no semicolon.\n"
        f"Tables: {COMPACT_SCHEMA}\n"
        f"Rules: use LIMIT (not TOP), lowercase aliases, no subqueries, no CTEs, LIMIT {MAX_ROWS}.\n"
        f"Question: {question}\nSQL:"
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_llm(question: str) -> str:
    prompt = _build_prompt(question)
    result = llm.invoke(prompt)
    # ChatGroq retorna AIMessage; OllamaLLM retorna str
    return result.content if hasattr(result, "content") else result


# -------------------------------
# ENDPOINTS
# -------------------------------
@app.get("/health", summary="Health check das dependências")
def health(response: Response):
    status = {"version": "3.1", "redis": False, "postgres": False, "ollama": False}
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
    if not all_ok:
        response.status_code = 503
    return status


@app.get("/schema", summary="Tabelas e colunas disponíveis no banco")
def get_schema(api_key: str = Depends(verify_api_key)):
    tables = db_instance.get_usable_table_names()
    schema = {table: db_instance.get_table_info([table]) for table in tables}
    return {"tables": tables, "schema": schema}


@app.post("/ask-db", summary="Pergunta para o banco de dados via IA")
async def ask_db(data: Question, request: Request, api_key: str = Depends(verify_api_key)):
    ip = request.client.host
    await asyncio.to_thread(check_rate_limit, ip)

    question = data.question.strip()
    session_suffix = f":{data.session_id}" if data.session_id else ""
    cache_key = f"q:{question.lower()}{session_suffix}"
    logger.info(f"[{ip}] Pergunta: {question}")

    # 1. Cache
    if cache is not None:
        cached = await asyncio.to_thread(cache.get, cache_key)
        if cached:
            logger.info(f"[{ip}] Cache hit.")
            return {"response": json.loads(cached), "source": "cache"}

    # 2. Histórico
    question_with_context = await asyncio.to_thread(
        build_question_with_history, question, data.session_id
    )

    # 3. Gera SQL (LLM com prompt compacto)
    try:
        raw_sql = await asyncio.to_thread(invoke_llm, question_with_context)
        sql_query = clean_sql(raw_sql)
        logger.info(f"[{ip}] SQL: {sql_query}")
    except Exception as e:
        logger.error(f"[{ip}] Erro LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar SQL: {e}")

    # 4. Validação
    validate_sql(sql_query)

    # 5. LIMIT
    sql_query = inject_limit(sql_query, MAX_ROWS)

    # 6. Executa read-only
    try:
        def run_query():
            with engine.begin() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                result = conn.execute(text(sql_query))
                return [dict(row._mapping) for row in result]

        rows = await asyncio.to_thread(run_query)
    except Exception as e:
        logger.error(f"[{ip}] Erro SQL: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar SQL: {e}")

    # 7. Salva histórico
    if data.session_id:
        await asyncio.to_thread(save_history, data.session_id, question, len(rows))

    # 8. Cache e resposta
    formatted = {
        "sql": sql_query,
        "row_count": len(rows),
        "rows": rows,
    }
    if cache is not None:
        await asyncio.to_thread(
            cache.setex, cache_key, CACHE_TTL, json.dumps(formatted, default=str)
        )

    return {"response": formatted, "source": "model"}
