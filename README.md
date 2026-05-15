# Bug classifier (LangGraph)

Multi-agent bug classification pipeline: type, severity (P0–P3), component, and summary. Notion logging will plug in under `integrations/`.

## Prerequisites

- Python 3.10 or newer (3.11+ recommended)
- An [XAI](https://platform.openai.com/) API key
- A [Notion](https://www.notion.so/) integration and database (for future logging)
- Optional: [LangSmith](https://smith.langchain.com/) for tracing (`LANGCHAIN_*` variables)

## Clone the repository

```bash
git clone <your-repo-url>
cd BugClassification
```

## Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r bug_classifier/requirements.txt
```

## Configure environment variables

Copy the example file to the **repository root** (recommended) or into `bug_classifier/`:

```bash
cp bug_classifier/.env.example .env
```

Edit `.env` and set:

| Variable               | Purpose |
|------------------------|---------|
| `xAI_API_KEY`          | Required. Used by LangChain OpenAI chat models. |
| `NOTION_API_KEY`       | Required for config validation. Used by `notion_logger` when implemented. |
| `NOTION_DATABASE_ID`   | Required. Target database for Notion rows. |
| `LANGCHAIN_TRACING_V2` | Set to `true` to send traces to LangSmith. |
| `LANGCHAIN_API_KEY`    | Required **if** `LANGCHAIN_TRACING_V2=true`. |
| `LANGCHAIN_PROJECT`    | LangSmith project name (default in example: `bug-classifier`). |

If tracing is enabled without `LANGCHAIN_API_KEY`, the app raises a clear error at import time.

## Notion database setup

1. In Notion, create a new **integration** at [My integrations](https://www.notion.so/my-integrations) and copy the **Internal Integration Secret** into `NOTION_API_KEY`.
2. Create a **database** (or use an existing one) that will store classified bugs. Share the database with your integration (**⋯** on the database page → **Connections** → add your integration).
3. Copy the database ID from the database URL:  
   `https://www.notion.so/{workspace}/{NOTION_DATABASE_ID}?v=...`  
   The `NOTION_DATABASE_ID` is the 32-character hex segment (with or without hyphens; the client accepts both).

The `integrations/notion_logger.py` module is a placeholder; wire `notion-client` there when you implement logging.

## Run the system

From the **repository root** (`BugClassification/`), with your virtual environment activated:

```bash
# Single report
python -m bug_classifier.main --report "App crashes on login..."

# Live demo (five example reports)
python -m bug_classifier.main --demo --dry-run
```

### Make targets

```bash
make install   # install dependencies
make test      # unit + integration tests (mocked LLM, NOTION_DRY_RUN)
make eval      # 20-example accuracy evaluation suite
make demo      # run the interactive demo (dry-run)
```

## Testing

Tests live under `bug_classifier/tests/` and use **pytest** with mocked LLM calls for speed and determinism. Set `NOTION_DRY_RUN=true` (default in `make test`) to avoid Notion API writes.

```bash
make test
make eval
```

After changing any agent prompt, run `make eval` to check type/severity/component accuracy and review confusion matrices for regressions.

## Project layout

- `main.py` — Entry point
- `state.py` — `BugState` TypedDict and `create_initial_state`
- `agents/` — One node per concern (placeholders)
- `graph/workflow.py` — LangGraph `StateGraph` wiring
- `integrations/notion_logger.py` — Notion hook (placeholder)
- `config.py` — Environment loading and validation
