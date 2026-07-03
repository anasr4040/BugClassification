import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_PKG_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and set {name}."
        )
    return value.strip()


def _optional(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()



_xai = (_optional("XAI_API_KEY") or _optional("GROK_API_KEY") or "").strip()
if not _xai:
    raise RuntimeError(
        "Missing required environment variable 'XAI_API_KEY' (or 'GROK_API_KEY'). "
        "Copy .env.example to .env and set XAI_API_KEY."
    )
XAI_API_KEY = _xai
NOTION_API_KEY = _require("NOTION_API_KEY")
NOTION_DATABASE_ID = _require("NOTION_DATABASE_ID")
NOTION_PARENT_PAGE_ID = _optional("NOTION_PARENT_PAGE_ID")
_dry_run_raw = (_optional("NOTION_DRY_RUN", "false") or "false").lower()
NOTION_DRY_RUN = _dry_run_raw in ("1", "true", "yes", "on")

# --- Multi-agent orchestration -------------------------------------------------
# Confidence below this threshold makes the supervisor request a revision.
CONFIDENCE_REVIEW_THRESHOLD = float(
    _optional("CONFIDENCE_REVIEW_THRESHOLD", "0.7") or "0.7"
)
# Maximum supervisor-directed revision rounds before approving with a
# needs_review flag (bounds the review/revise loop).
MAX_REVISION_ROUNDS = int(_optional("MAX_REVISION_ROUNDS", "1") or "1")
# Maximum tool-calling iterations each specialist agent may take per run.
AGENT_TOOL_ROUNDS = int(_optional("AGENT_TOOL_ROUNDS", "3") or "3")
# When true, the P0 emergency handler pauses for human approval via a
# LangGraph interrupt (requires compiling the graph with a checkpointer).
_hitl_raw = (_optional("HITL_EMERGENCY", "false") or "false").lower()
HITL_EMERGENCY = _hitl_raw in ("1", "true", "yes", "on")

# LangSmith / LangChain tracing
LANGCHAIN_TRACING_V2 = (_optional("LANGCHAIN_TRACING_V2", "false") or "false").strip()
LANGCHAIN_API_KEY = (_optional("LANGCHAIN_API_KEY") or "").strip()
LANGCHAIN_PROJECT = (_optional("LANGCHAIN_PROJECT", "bug-classifier") or "bug-classifier").strip()

_tracing_requested = LANGCHAIN_TRACING_V2.lower() in ("1", "true", "yes", "on")
LANGCHAIN_TRACING_ENABLED = bool(
    _tracing_requested and LANGCHAIN_API_KEY and LANGCHAIN_PROJECT
)

if LANGCHAIN_TRACING_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if _tracing_requested and not LANGCHAIN_API_KEY:
        print(
            "Warning: LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is missing. "
            "LangSmith tracing is disabled."
        )
    elif not _tracing_requested:
        print(
            "Warning: LangSmith tracing is disabled. Set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY to enable traces in the LangSmith UI."
        )

DEFAULT_COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "backend": "Server-side logic, business rules, data processing, background jobs",
    "frontend": "UI rendering, client-side JavaScript, CSS, React/Vue components",
    "api": "REST/GraphQL endpoints, request handling, serialization, rate limiting",
    "infrastructure": "Deployment, CI/CD, cloud services, networking, DNS, load balancers",
    "database": "Queries, migrations, schema issues, connection pooling, replication",
    "authentication": "Login, OAuth, SSO, tokens, permissions, RBAC, session management",
}


def _parse_bug_components() -> tuple[str, ...]:
    raw = _optional("BUG_COMPONENTS")
    if not raw:
        return tuple(DEFAULT_COMPONENT_DESCRIPTIONS.keys())
    components = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not components:
        raise RuntimeError(
            "BUG_COMPONENTS is set but empty. Provide a comma-separated list, e.g. "
            "'backend,frontend,api'."
        )
    return components


BUG_COMPONENTS = _parse_bug_components()
COMPONENT_DESCRIPTIONS: dict[str, str] = {
    name: DEFAULT_COMPONENT_DESCRIPTIONS.get(
        name, "Engineering team area (custom component)"
    )
    for name in BUG_COMPONENTS
}
DEFAULT_COMPONENT = BUG_COMPONENTS[0]
