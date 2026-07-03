# Bug classifier (LangGraph)

Multi-agent bug classification system: type, severity (P0–P3), component, and summary, with Notion ticket logging under `integrations/`.

## Architecture

```
              ┌─→ Type Classifier ───────┐
   report ────┼─→ Severity Assessor ─────┼─→ Supervisor / Critic ─┬─(revise)──→ flagged specialists ─→ Supervisor …
              └─→ Component Identifier ──┘                        ├─(P0)──────→ Emergency Handler → END
                                                                  └─(approve)─→ Summary Agent → Notion Logger → END
```

- **Parallel specialists** — the type, severity, and component agents run
  concurrently; reducer-annotated fields in `state.py` merge their
  simultaneous writes (confidences, agent notes).
- **Tool-using agents** — each specialist can call investigation tools
  (`agents/tools.py`): stack-trace parsing, blast-radius signal scanning,
  technology→component keyword matching, and similar-bug history lookup in
  Notion. The LLM decides which tools to call in a bounded ReAct loop
  (`AGENT_TOOL_ROUNDS`).
- **Supervisor / critic** (`agents/supervisor.py`) — reviews the joined
  result. Complete, confident, consistent classifications are approved on a
  deterministic fast path (no LLM cost). Missing/low-confidence/inconsistent
  dimensions get targeted LLM critique fed back to the flagged specialists
  for a revision round, bounded by `MAX_REVISION_ROUNDS`; if the budget runs
  out, the result is approved with `needs_review=True` for human triage.
- **Emergency path** — an approved P0 skips the summary pipeline and pages
  on-call. With `HITL_EMERGENCY=true` (and a checkpointer passed to
  `build_graph`), the graph pauses on a LangGraph interrupt so an operator
  can approve the escalation or downgrade the severity before anything fires.
- **Inter-agent communication** — agents append to a shared `agent_notes`
  channel (reasoning, tools used, critiques), giving downstream agents and
  operators an audit trail of the conversation.

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
| `CONFIDENCE_REVIEW_THRESHOLD` | Confidence below this triggers a supervisor revision (default `0.7`). |
| `MAX_REVISION_ROUNDS` | Supervisor revision-round budget (default `1`). |
| `AGENT_TOOL_ROUNDS`   | Max tool-calling iterations per specialist run (default `3`). |
| `HITL_EMERGENCY`      | `true` pauses P0 escalation on a human-approval interrupt (needs a checkpointer). |

If tracing is enabled without `LANGCHAIN_API_KEY`, the app raises a clear error at import time.

## Notion database setup

1. In Notion, create a new **integration** at [My integrations](https://www.notion.so/my-integrations) and copy the **Internal Integration Secret** into `NOTION_API_KEY`.
2. Create a **database** (or use an existing one) that will store classified bugs. Share the database with your integration (**⋯** on the database page → **Connections** → add your integration).
3. Copy the database ID from the database URL:  
   `https://www.notion.so/{workspace}/{NOTION_DATABASE_ID}?v=...`  
   The `NOTION_DATABASE_ID` is the 32-character hex segment (with or without hyphens; the client accepts both).

The `integrations/notion_logger.py` module is a placeholder; wire `notion-client` there when you implement logging.

## Web UI — watch the agents work

```bash
make ui            # http://127.0.0.1:8765 (dry-run: no Notion writes)
# or:
python -m bug_classifier.ui --port 8765 --dry-run
```

The control-room UI animates the whole process live: the report fanning out
to the three specialists in parallel, each agent's request/response pulsing
along the graph edges, tool calls, the supervisor's verdict (including
revision feedback flowing back to flagged specialists), the P0 emergency
branch, confidence meters with the review threshold, and the final ticket.
From the UI you can classify your own bug reports, run the built-in demo
examples, or run the pytest suite in a live console.

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

- `Makefile` — `install` / `test` / `eval` / `demo` targets (run from the repo root)
- `bug_classifier/main.py` — Entry point (`python -m bug_classifier.main`)
- `bug_classifier/state.py` — `BugState` TypedDict with reducers for parallel-agent writes
- `bug_classifier/agents/` — Specialist agents (type, severity, component, summary), the supervisor/critic, and their investigation tools (`tools.py`)
- `bug_classifier/graph/workflow.py` — LangGraph `StateGraph` wiring: parallel fan-out, supervisor routing, revision loop, emergency path
- `bug_classifier/integrations/notion_logger.py` — Notion ticket creation
- `bug_classifier/config.py` — Environment loading, validation, and orchestration knobs
