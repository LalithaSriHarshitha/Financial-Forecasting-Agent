# Setup Guide: TCS Financial Forecasting Agent

| | |
|---|---|
| **Document** | Installation, configuration and operation guide |
| **Applies to** | Release 3.1.0 and later |
| **Audience** | Anyone setting up the project from the GitHub repository: reviewers, evaluators, contributors |
| **Time required** | About 30–45 minutes for a first-time setup (most of it is installing prerequisites) |
| **Repository** | <https://github.com/LalithaSriHarshitha/Financial-Forecasting-Agent> |

---

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [System overview](#2-system-overview)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Verifying the installation](#6-verifying-the-installation)
7. [Running the application](#7-running-the-application)
8. [Evaluation and tests](#8-evaluation-and-tests)
9. [Optional configurations](#9-optional-configurations)
10. [Updating an existing installation](#10-updating-an-existing-installation)
11. [Troubleshooting](#11-troubleshooting)
12. [Security guidelines](#12-security-guidelines)
13. [Appendix A: environment variable reference](#appendix-a-environment-variable-reference)
14. [Appendix B: project structure](#appendix-b-project-structure)

---

## 1. Purpose and scope

This guide explains how to install, configure, verify and run the TCS Financial Forecasting
Agent on your own computer, using **your own API key and your own database**. Nothing in this
guide uses the original author's credentials or quotas.

At the end you will have:

- the FastAPI service running locally at `http://127.0.0.1:8000`;
- a working forecast produced by the AI agent from TCS financial reports and earnings-call transcripts;
- every request and response logged in your own MySQL database.

For the design and reasoning behind the system, see [README.md](README.md) and
[AI_USAGE.md](AI_USAGE.md). For public cloud deployment, see [DEPLOY.md](DEPLOY.md).

---

## 2. System overview

| Component | Technology | Role |
|---|---|---|
| API service | Python 3.11, FastAPI, Uvicorn | Receives requests at `POST /forecast` |
| AI agent | LangChain + Google Gemini (default) or Ollama | Plans the analysis and calls three tools |
| FinancialDataExtractorTool | PyMuPDF + rule-based parser | Reads validated figures from the quarterly fact sheets |
| QualitativeAnalysisTool | Chroma vector store + BM25 (hybrid search) | Analyses the earnings-call transcripts |
| MarketDataTool | yfinance | Fetches the live TCS share price (optional) |
| Database | MySQL 8.0 | Logs requests, responses, tool calls and extracted metrics |

**Included in the repository** (you do not need to download or build them):

- the six source documents in `data/raw/`: three quarterly fact sheets and three earnings-call transcripts;
- a ready-built transcript index for Gemini in `data/chroma/gemini_models_gemini-embedding-001_v2/`.

---

## 3. Requirements

### 3.1 Hardware

| Resource | Minimum | Notes |
|---|---|---|
| RAM | 4 GB free | The service uses about 200–400 MB while running |
| Disk | 2 GB free | Python packages take most of it |
| Internet | Required | For the Gemini API, package installation and (optional) market data |

Running models locally with Ollama instead of Gemini needs much more: see [section 9.1](#91-local-models-with-ollama-instead-of-gemini).

### 3.2 Software

| Software | Version | Download | Verify with |
|---|---|---|---|
| Python | 3.10, 3.11 or 3.12 (3.11 recommended) | <https://www.python.org/downloads/> | `python --version` |
| Git | Any recent version | <https://git-scm.com/downloads> | `git --version` |
| MySQL Server | 8.0 | <https://dev.mysql.com/downloads/installer/> (Windows) or your OS package manager | `mysql --version` |

> **Windows:** when installing Python, tick **"Add Python to PATH"**. During MySQL setup, note
> the **root password** and keep the default **port 3306**. To use the `mysql` command, add
> `C:\Program Files\MySQL\MySQL Server 8.0\bin` to your PATH.

### 3.3 Accounts and keys

| Item | Required | Where to get it | Cost |
|---|---|---|---|
| Google Gemini API key | Yes (default provider) | <https://aistudio.google.com/apikey> | Free tier available |
| GitHub account | Only if you want to fork or contribute | <https://github.com> | Free |

No key is needed for market data.

---

## 4. Installation

Run every command **one line at a time**. Do not add `# comments` after commands: Windows
Command Prompt treats them as extra arguments.

### 4.1 Clone the repository

```bash
git clone https://github.com/LalithaSriHarshitha/Financial-Forecasting-Agent.git
cd Financial-Forecasting-Agent
```

> **Windows Command Prompt:** to change to a folder on another drive, use `cd /d D:\path\to\folder`.

### 4.2 Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it with the command for your terminal:

| Terminal | Prompt looks like | Activation command |
|---|---|---|
| Windows Command Prompt | `C:\...>` | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `PS C:\...>` | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `$` | `source .venv/bin/activate` |

When activation works, the prompt starts with `(.venv)`. Activate the environment in **every new
terminal window** you use for this project.

> If PowerShell refuses to run the script, run this once, then activate again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 4.3 Install the Python packages

```bash
pip install -r requirements.txt
```

This takes 3–10 minutes the first time. Exact versions are pinned in `requirements.txt`.

### 4.4 Prepare MySQL

The application **creates its database and tables automatically** on first start, so you only
need a MySQL user it can connect with. Choose one option.

**Option A: use the root account (quickest).** Use `root` and its password in the configuration
(section 5). Nothing else to do.

**Option B: create a dedicated user (recommended).** Log in as root:

```bash
mysql -u root -p
```

At the `mysql>` prompt, run (choose your own password):

```sql
CREATE DATABASE IF NOT EXISTS tcs_forecast CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'forecast_user'@'localhost' IDENTIFIED BY 'ChooseAStrongPassword1!';
GRANT ALL PRIVILEGES ON tcs_forecast.* TO 'forecast_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 4.5 Create a Gemini API key

1. Open <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key**. For a fresh quota, choose **Create API key in new project**.
3. Copy the key; you will paste it into the configuration in the next section.

> **Quota note:** free-tier limits apply **per Google Cloud project, not per key**, and per model.
> A forecast uses about 5 requests. If a forecast returns HTTP 429 "quota exhausted", wait for
> the daily reset (midnight Pacific Time) or use a key from another project.

---

## 5. Configuration

All settings live in a `.env` file in the project root. The repository contains a template,
`.env.example`, with every setting documented.

### 5.1 Create the configuration file

| Terminal | Command |
|---|---|
| Windows (Command Prompt or PowerShell) | `copy .env.example .env` |
| macOS / Linux | `cp .env.example .env` |

Open it in a text editor (Windows: `notepad .env`).

### 5.2 Required settings

Change these values; you can leave everything else at its default for a local setup.

| Variable | What to enter | Example |
|---|---|---|
| `GOOGLE_API_KEY` | Your Gemini API key from section 4.5 | `AIza...` |
| `MYSQL_USER` | `root` (option A) or `forecast_user` (option B) | `forecast_user` |
| `MYSQL_PASSWORD` | That user's password | `ChooseAStrongPassword1!` |

Check that these defaults suit your machine:

| Variable | Default | Change it if |
|---|---|---|
| `MYSQL_HOST` / `MYSQL_PORT` | `localhost` / `3306` | MySQL runs elsewhere or on another port |
| `MYSQL_DATABASE` | `tcs_forecast` | You want another database name |
| `LLM_PROVIDER` | `gemini` | You use Ollama (section 9.1) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Your key does not offer this model (section 6.1 lists the ones it does) |

Save the file. **Never commit `.env`**; it is excluded by `.gitignore`.

### 5.3 Documents and transcript index

No action is needed: the documents and the Gemini transcript index are included in the repository.

Rebuild the index only if you change `LLM_PROVIDER` or `GEMINI_EMBEDDING_MODEL`, or replace the
transcripts. The command is in section 10.

---

## 6. Verifying the installation

### 6.1 Run the setup check

With the virtual environment active, in the project folder:

```bash
python -m app.check
```

This makes **one** Gemini request. Expected output:

```text
LLM provider: gemini  (change with LLM_PROVIDER in .env)

1) Gemini API key ... set
   model 'gemini-3.6-flash': available
   Generating a test reply ... ready
2) MySQL connection ... OK (MySQL 8.0.x, tables ready)
3) Documents in data/raw/:
   [x] Q4 FY26  financial_report          TCS_Q4FY26_FactSheet.pdf
   [x] Q4 FY26  earnings_call_transcript  TCS_Q4FY26_Transcript.pdf
   [x] Q1 FY27  financial_report          TCS_Q1FY27_FactSheet.pdf
   [x] Q1 FY27  earnings_call_transcript  TCS_Q1FY27_Transcript.pdf
   [x] Q2 FY27  financial_report          TCS_Q2FY27_FactSheet.pdf
   [x] Q2 FY27  earnings_call_transcript  TCS_Q2FY27_Transcript.pdf
4) Vector index (models/gemini-embedding-001): built

All checks passed.
```

If a check fails, the line explains what to fix; [section 11](#11-troubleshooting) covers common cases.
If the model shows `NOT FOUND`, the output lists the models your key can use; put one of them in
`GEMINI_MODEL`.

### 6.2 Run the automated tests (optional)

```bash
pytest -q
```

All tests should pass. They use a temporary SQLite database and a scripted fake LLM, so they
need neither MySQL, network access nor your API key.

---

## 7. Running the application

### 7.1 One-command end-to-end run

```bash
python -m app.demo
```

This sends the assignment's example task to the real `POST /forecast` endpoint, confirms the
MySQL log entry, prints a summary and saves the full response to `sample_output.json`. It takes
about 1–2 minutes and uses about 5 Gemini requests. Progress lines show each LLM and tool call
as it happens.

### 7.2 Start the API service

```bash
uvicorn app.main:app
```

Wait for `Application startup complete.` Keep this window open while you use the API; press
**Ctrl+C** to stop it. If Ctrl+C does not respond on Windows, close the window.

Open **<http://127.0.0.1:8000>** in a browser. It redirects to the interactive API documentation,
where you can call every endpoint with **Try it out → Execute**.

### 7.3 Endpoints

| Method | Path | Purpose | Uses Gemini quota |
|---|---|---|---|
| `GET` | `/health` | Database, documents and index status | No |
| `GET` | `/metrics?quarter=Q2 FY27` | Validated financial figures and accounting checks | No |
| `POST` | `/forecast` | Runs the agent; body `{}` uses the assignment's example task, or `{"task": "..."}` | Yes (~5 requests) |
| `GET` | `/forecast/{request_id}` | Returns a logged request, its response and tool calls from MySQL | No |

### 7.4 Calling the API from a terminal

Use a **second** terminal window while the server runs in the first.

**Windows PowerShell:**

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/forecast -ContentType "application/json" -Body '{"task": "What are the main margin risks for TCS next quarter?"}' -TimeoutSec 600 | ConvertTo-Json -Depth 8
```

> In PowerShell, `curl` is an alias for a different command. Use `Invoke-RestMethod` as above, or `curl.exe`.

**Windows Command Prompt:**

```bat
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d "{\"task\": \"What are the main margin risks for TCS next quarter?\"}"
```

**macOS / Linux:**

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" \
  -d '{"task": "What are the main margin risks for TCS next quarter?"}'
```

### 7.5 Response and errors

A successful forecast returns JSON with `request_id`, `tool_calls`, `grounding` (results of the
automatic checks), `usage` (LLM calls and tokens) and `result`. `result` contains the trends,
management outlook with verified quotes, risks, opportunities, structured forecast directions,
validated per-quarter metrics and market data. `sample_output.json` shows a complete example.

| HTTP status | Meaning | Action |
|---|---|---|
| 401 | Missing or invalid `X-API-Key` | Only when `API_KEYS` is set: send the header |
| 422 | Invalid request body | Send `{}` or `{"task": "at least 10 characters"}` |
| 429 | Hourly rate limit, or the Gemini quota is exhausted | Read the message; wait or use another key |
| 500 | Unexpected failure | See the server window; the error is logged with the `request_id` |
| 504 | Forecast exceeded `FORECAST_TIMEOUT_S` | Retry, or raise the limit |

### 7.6 Inspecting the database

```bash
mysql -u root -p
```

```sql
USE tcs_forecast;
SELECT request_id, status, latency_ms, created_at FROM request_logs ORDER BY id DESC LIMIT 10;
SELECT request_id, sequence, tool_name, duration_ms, error FROM tool_calls ORDER BY id DESC LIMIT 10;
SELECT quarter, revenue_cr, operating_margin_pct, net_profit_cr, verified FROM financial_metrics ORDER BY quarter;
```

Timestamps are stored in UTC.

---

## 8. Evaluation and tests

| Command | What it does | Gemini chat requests |
|---|---|---|
| `python -m eval.extraction` | Compares 33 extracted figures with `eval/ground_truth.json` | 0 |
| `python -m eval.run` | Extraction accuracy, retrieval quality, and faithfulness checks on `sample_output.json`; writes `eval/report.md` | 0 (uses embeddings) |
| `python -m eval.run --consistency 3` | Also runs the agent 3 times and measures agreement of the forecasts | ~15 |
| `pytest -q` | Unit and integration tests | 0 |

---

## 9. Optional configurations

### 9.1 Local models with Ollama instead of Gemini

Recommended only on a machine with 16 GB RAM and preferably a GPU; on a laptop CPU a forecast
can take 15 minutes or more.

1. Install Ollama from <https://ollama.com/download> and start it.
2. Download the models:
   ```bash
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```
3. In `.env`, set `LLM_PROVIDER=ollama`. With 8 GB RAM, also set `OLLAMA_MODEL=qwen3:4b` and `OLLAMA_NUM_CTX=8192`.
4. Build the transcript index for the local embedding model (one time):
   ```bash
   python -m app.ingestion --skip-download
   ```
5. Run `python -m app.check`, then continue with section 7.

### 9.2 Disable live market data

Set `MARKET_DATA_ENABLED=false` in `.env`. Forecasts still work; the market section is omitted.

### 9.3 Protect the API when others can reach it

If you expose the service beyond your own machine, set in `.env`:

```text
API_KEYS=choose-a-long-random-key
FORECAST_RATE_LIMIT_PER_HOUR=10
```

Callers must then send the header `X-API-Key: choose-a-long-random-key` on `POST /forecast` and
`GET /forecast/{request_id}`. In the browser docs, use the **Authorize** button. To reach the
service from other devices on your network, start it with
`uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### 9.4 Spread the Gemini free-tier quota

Quotas are counted per model. Set `GEMINI_TOOL_MODEL` to a second model your key offers (for
example a "flash-lite" model listed by `python -m app.check`), so the tools' internal calls use
that model's allowance.

### 9.5 Cloud deployment

Follow [DEPLOY.md](DEPLOY.md) for a free deployment on Render with a hosted MySQL database.

---

## 10. Updating an existing installation

```bash
git pull
pip install -r requirements.txt
python -m app.check
```

Rebuild the transcript index only if you changed `LLM_PROVIDER` or the embedding model, or if
`python -m app.check` reports it as not built:

```bash
python -m app.ingestion --skip-download
```

On Gemini's free tier the build pauses briefly to stay within limits and takes about 4 minutes.

---

## 11. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `'python' is not recognized` | Python not on PATH | Reinstall Python with "Add Python to PATH" ticked, or use `py` instead of `python` on Windows |
| Prompt does not show `(.venv)` | Virtual environment not active | Run the activation command for your terminal (section 4.2) |
| `The syntax of the command is incorrect` | A `# comment` was typed after a command | Type commands without comments |
| `'mysql' is not recognized` | MySQL `bin` folder not on PATH | Add `C:\Program Files\MySQL\MySQL Server 8.0\bin` to PATH, or use "MySQL 8.0 Command Line Client" from the Start menu |
| `Access denied for user ... (using password: NO)` | `.env` missing or password empty | Create `.env` (section 5.1) and set `MYSQL_PASSWORD` |
| `Can't connect to MySQL server on 'localhost'` | MySQL service stopped | Windows (admin terminal): `net start MySQL80`; macOS: `brew services start mysql`; Linux: `sudo systemctl start mysql` |
| `GOOGLE_API_KEY is not set` | Key missing in `.env` | Set `GOOGLE_API_KEY` and save the file |
| `model '...' NOT FOUND` | Model not offered to your key | Set `GEMINI_MODEL` to a model listed by `python -m app.check` |
| HTTP 429 "quota exhausted" | Gemini free-tier daily limit reached | Wait for the reset, use a key from a new project, or see section 9.4 |
| Frequent HTTP 503 in the logs | Gemini under high demand | Handled by automatic retries; just wait |
| Changed `.env` but nothing changed | Settings are read at startup | Stop the server (Ctrl+C) and start it again |
| `Could not connect to server` from curl | API server not running | Start it in another window (section 7.2) and keep that window open |
| PowerShell shows a "Script Execution Risk" prompt for `curl` | PowerShell's `curl` alias | Use `Invoke-RestMethod` or `curl.exe` |
| `Vector index: not built` | Index missing for your provider | Run `python -m app.ingestion --skip-download` |
| `Live market data unavailable` in `data_gaps` | Yahoo Finance unreachable | Forecast is still complete; retry later or disable (section 9.2) |
| A forecast never finishes with Ollama | CPU too slow for the model | Use Gemini, or a smaller model (section 9.1) |

If a problem persists, run `python -m app.check` and include its output, and the error shown in
the server window, when asking for help or opening an issue.

---

## 12. Security guidelines

- **Never commit `.env`** or paste API keys into issues, screenshots or chat. `.gitignore` already excludes `.env`.
- **Keep keys separate:** your Gemini key (`GOOGLE_API_KEY`) is a secret for the server. The keys in `API_KEYS` are the ones you give to callers.
- **Rotate a key that has been exposed:** create a new one in Google AI Studio, update `.env`, and delete the old key.
- **Use a dedicated MySQL user** (section 4.4, option B) rather than `root` for anything beyond local testing.
- **Set `API_KEYS`** before making the service reachable from other machines (section 9.3).

---

## Appendix A: environment variable reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `GOOGLE_API_KEY` | — | Gemini API key (required for Gemini) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Chat model for the agent and tools |
| `GEMINI_TOOL_MODEL` | empty | Optional separate model for the tools' internal calls |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for transcript search (the included index uses it) |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature where the model supports it |
| `LLM_REQUESTS_PER_MINUTE` | `8` | Pacing for Gemini chat calls |
| `EMBED_REQUESTS_PER_MINUTE` | `90` | Pacing for Gemini embedding calls during indexing |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` / `OLLAMA_TOOL_MODEL` | `qwen3:8b` / empty | Ollama chat models |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_NUM_CTX` | `16384` | Ollama context window (tokens) |
| `OLLAMA_NUM_PREDICT` / `OLLAMA_TIMEOUT_S` | `3072` / `900` | Ollama output cap and per-call timeout |
| `RAG_CHUNKS_PER_QUARTER` | `2` | Transcript excerpts retrieved per topic per quarter |
| `MARKET_DATA_ENABLED` / `MARKET_SYMBOL` | `true` / `TCS.NS` | Live market data |
| `FORECAST_TIMEOUT_S` | `600` | Maximum seconds per forecast (HTTP 504 beyond) |
| `API_KEYS` | empty | Comma-separated caller keys; empty means no key required |
| `FORECAST_RATE_LIMIT_PER_HOUR` | `10` | Forecasts per client per hour (0 = unlimited) |
| `MYSQL_HOST` / `MYSQL_PORT` | `localhost` / `3306` | MySQL server |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `forecast_user` / — | MySQL credentials |
| `MYSQL_DATABASE` | `tcs_forecast` | Database name (created automatically) |
| `MYSQL_SSL` / `MYSQL_SSL_CA_CERT` | `false` / empty | TLS for hosted MySQL |
| `DATABASE_URL` | empty | Advanced: a full SQLAlchemy URL that overrides all `MYSQL_*` values |
| `DATA_DIR` / `CHROMA_DIR` | `data` / `data/chroma` | Data and index locations |
| `INDEX_AUTOBUILD` | `true` | Build a missing index in the background when the API starts |
| `LOG_LEVEL` | `INFO` | Logging level |

## Appendix B: project structure

```text
app/            API, configuration, agent, tools, ingestion, schemas, database
data/raw/       the six source PDFs (included)
data/chroma/    transcript index for Gemini (included)
data/manifest.json   document list with source URLs
eval/           ground truth, retrieval questions, evaluation suite and report
tests/          automated tests (with real fact-sheet text fixtures)
README.md       overview, design, agent and tool details
AI_USAGE.md     AI stack, guardrails, evaluation, limits
DEPLOY.md       cloud deployment on Render
SETUP.md        this guide
```
