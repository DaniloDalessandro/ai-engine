
API que recebe perguntas em linguagem natural, gera SQL via LLM local (Ollama) e executa no PostgreSQL — tudo via Docker, sem dependência de serviços externos.

---

## Como funciona

```
Pergunta → FastAPI → Redis (cache?)
                         ↓ miss
                    LangChain + Ollama
                         ↓
                    SQL gerado (com schema real)
                         ↓
                    Validação de segurança
                         ↓
                    PostgreSQL (read-only)
                         ↓
                    Resposta em linguagem natural
```

---

## Pré-requisitos

- Docker e Docker Compose v2+
- ~5GB livres em disco (modelo qwen:7b)

---

## Uso local

### 1. Configure o ambiente

```bash
cp .env.example .env
```

Abra o `.env` e preencha `API_KEY` e `POSTGRES_PASSWORD` com chaves geradas:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Suba os containers

```bash
docker compose up -d --build
```

### 3. Baixe o modelo de linguagem

```bash
docker exec ai-ollama ollama pull qwen:7b
```

> Aguarde o download (~4GB) antes de fazer requisições.

### 4. Verifique se tudo está saudável

```bash
curl http://localhost:8080/health
```

Resposta esperada:
```json
{
  "status": "ok",
  "redis": true,
  "postgres": true,
  "ollama": true,
  "version": "3.0"
}
```

### 5. Faça uma consulta

```bash
curl -X POST http://localhost:8080/ask-db \
  -H "Content-Type: application/json" \
  -H "x-api-key: sua_chave_aqui" \
  -d '{"question": "Quais são os 10 produtos mais vendidos?"}'
```

Resposta:
```json
{
  "response": {
    "answer": "Os 10 produtos mais vendidos são: Produto A (320 unidades)...",
    "sql": "SELECT nome, SUM(quantidade) AS total FROM vendas GROUP BY nome ORDER BY total DESC LIMIT 10",
    "row_count": 10,
    "rows": [...]
  },
  "source": "model"
}
```

---

## Deploy em produção (EasyPanel, VPS, etc.)

O `docker-compose.yml` funciona em qualquer ambiente com Docker.

### 1. Publique a imagem

```bash
docker build -t danilodalessandro/ai-sql-engine:latest ./api
docker push danilodalessandro/ai-sql-engine:latest
```

### 2. Configure as variáveis de ambiente no servidor

No painel do servidor (EasyPanel, Portainer, etc.), adicione as mesmas variáveis do `.env`:

```
POSTGRES_HOST=postgres-replica
POSTGRES_PORT=5432
POSTGRES_DB=ai_database
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=sua_senha
REDIS_HOST=redis
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen:7b
API_KEY=sua_chave
RATE_LIMIT=5
CACHE_TTL=3600
MAX_ROWS=100
MAX_HISTORY=5
LOG_LEVEL=INFO
```

### 3. Cole o `docker-compose.yml` no painel e faça o deploy

### 4. Baixe o modelo no container do Ollama

```bash
ollama pull qwen:7b
```

### 5. Configure o domínio

Aponte seu domínio para a porta `8000` do serviço `api`.

---

## Endpoints

| Método | Rota      | Auth | Descrição                               |
|--------|-----------|------|-----------------------------------------|
| GET    | `/health` | Não  | Status real de Redis, Postgres e Ollama |
| GET    | `/schema` | Sim  | Tabelas e colunas disponíveis no banco  |
| POST   | `/ask-db` | Sim  | Faz uma pergunta em linguagem natural   |

### POST `/ask-db` — body

```json
{
  "question": "Sua pergunta aqui",
  "session_id": "usuario-123"
}
```

- `question` — obrigatório
- `session_id` — opcional, mantém histórico de conversa entre perguntas encadeadas

---

## Autenticação

Todas as rotas com `Auth: Sim` exigem o header `x-api-key`.

**Gere uma chave segura:**

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Coloque no `.env`:**

```env
API_KEY=valor_gerado_aqui
```

**Use nas requisições:**

```bash
curl -X POST http://localhost:8080/ask-db \
  -H "x-api-key: valor_gerado_aqui" \
  -H "Content-Type: application/json" \
  -d '{"question": "liste os produtos"}'
```

Sem o header ou com chave errada → `401 Unauthorized`.

---

## Variáveis de ambiente

| Variável            | Padrão             | Descrição                              |
|---------------------|--------------------|----------------------------------------|
| `API_KEY`           | —                  | Chave de autenticação (**obrigatório**)|
| `POSTGRES_PASSWORD` | —                  | Senha do banco (**obrigatório**)       |
| `POSTGRES_HOST`     | `postgres-replica` | Host do banco                          |
| `POSTGRES_PORT`     | `5432`             | Porta do banco                         |
| `POSTGRES_DB`       | `ai_database`      | Nome do banco                          |
| `POSTGRES_USER`     | `ai_user`          | Usuário do banco                       |
| `REDIS_HOST`        | `redis`            | Host do Redis                          |
| `OLLAMA_URL`        | `http://ollama:11434` | URL base do Ollama                  |
| `OLLAMA_MODEL`      | `qwen:7b`          | Modelo a ser usado                     |
| `RATE_LIMIT`        | `5`                | Requisições por minuto por IP          |
| `CACHE_TTL`         | `3600`             | TTL do cache em segundos               |
| `MAX_ROWS`          | `100`              | Limite de linhas retornadas por query  |
| `MAX_HISTORY`       | `5`                | Trocas mantidas no histórico de sessão |
| `LOG_LEVEL`         | `INFO`             | Nível de log (DEBUG, INFO, WARNING)    |

---

## Serviços

| Serviço            | Porta         | Descrição                                          |
|--------------------|---------------|----------------------------------------------------|
| `api`              | 8080          | API FastAPI                                        |
| `ollama`           | 11434 (local) | LLM local (acessível apenas no próprio servidor)   |
| `postgres-replica` | —             | PostgreSQL read-only (rede interna)                |
| `redis`            | —             | Cache (rede interna)                               |

---

## Segurança de SQL

O LLM é instruído a gerar apenas `SELECT`. A API valida com `sqlparse` e bloqueia qualquer query com:
`DELETE` `UPDATE` `INSERT` `DROP` `ALTER` `TRUNCATE` `CREATE` `GRANT` `REVOKE`

Toda execução roda em transação `READ ONLY` real no PostgreSQL.

---

## Parar os serviços

```bash
# Para os containers
docker compose down

# Para os containers e apaga os volumes (banco + modelos)
docker compose down -v
```

---

## Replicação PostgreSQL (opcional)

Se você já tem um PostgreSQL em produção e quer que a IA leia dele via streaming replication:

### No servidor PRIMARY

**`postgresql.conf`:**
```conf
wal_level = replica
max_wal_senders = 3
wal_keep_size = 256
listen_addresses = '*'
```

**Criar usuário de replicação:**
```sql
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'senha_forte';
```

**`pg_hba.conf`** — libere apenas o IP da VPS:
```conf
host replication replicator <IP_DA_VPS>/32 scram-sha-256
```

```bash
sudo systemctl restart postgresql
```

### Na VPS (REPLICA)

**1. Suba e pare o container:**
```bash
docker compose up -d postgres-replica
docker compose stop postgres-replica
```

**2. Limpe os dados:**
```bash
docker run --rm \
  -v ai-sql_enginer_postgres_data:/var/lib/postgresql/data \
  postgres:16 bash -c "rm -rf /var/lib/postgresql/data/*"
```

**3. Execute o basebackup:**
```bash
docker run --rm \
  -v ai-sql_enginer_postgres_data:/var/lib/postgresql/data \
  -e PGPASSWORD=senha_forte \
  postgres:16 \
  pg_basebackup -h <IP_DO_PRIMARY> -p 5432 -U replicator \
    -D /var/lib/postgresql/data -Fp -Xs -P -R
```

**4. Suba novamente e verifique no PRIMARY:**
```bash
docker compose up -d postgres-replica
```
```sql
SELECT client_addr, state FROM pg_stat_replication;
-- state deve ser "streaming"
```

---

## Stack

- **FastAPI** + **LangChain** — API e orquestração do LLM
- **Ollama** (`qwen:7b`) — modelo de linguagem local
- **PostgreSQL 16** — banco de dados
- **Redis 7** — cache de respostas
- **Docker Compose** — orquestração
