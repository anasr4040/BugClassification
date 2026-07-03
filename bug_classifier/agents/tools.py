"""Investigation tools for the specialist agents.

Each specialist is a tool-using agent: instead of answering from the raw
report alone, its LLM may call these tools to gather evidence (stack-trace
facts, blast-radius signals, technology→component matches, similar past
tickets) before committing to a structured classification.

Tools are deterministic and offline-safe (the Notion lookup degrades
gracefully under ``NOTION_DRY_RUN`` or API failure), so agent behavior stays
testable and the pipeline never blocks on a tool.

``run_tool_loop`` implements the shared bounded ReAct-style loop: the bound
LLM decides which tools to call; observations are appended as
``ToolMessage``s; the loop ends when the model stops requesting tools or the
round budget (``config.AGENT_TOOL_ROUNDS``) is exhausted.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool

from bug_classifier import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal dictionaries
# ---------------------------------------------------------------------------
_EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error))\b")
_FILE_LINE_RE = re.compile(
    r"([\w./\\-]+\.(?:java|py|js|jsx|ts|tsx|c|cc|cpp|h|hpp|go|rb|rs|php|cs)):(\d+)"
)
_HTTP_CODE_RE = re.compile(r"\b(4\d{2}|5\d{2})\b")
_SIGNAL_RE = re.compile(r"\b(SIGSEGV|SIGABRT|SIGKILL|SIGBUS|segfault|core dump(?:ed)?|OOMKilled|OutOfMemory\w*)\b", re.IGNORECASE)

_BROAD_IMPACT_PHRASES = (
    "all users",
    "every user",
    "100%",
    "production down",
    "production is down",
    "complete outage",
    "outage",
    "no users can",
    "cannot access",
    "data loss",
    "actively exploited",
    "no authentication required",
)
_URGENCY_PHRASES = (
    "urgent",
    "critical",
    "immediately",
    "asap",
    "emergency",
    "right now",
    "escalate",
)
_MITIGATION_PHRASES = (
    "workaround",
    "cosmetic",
    "minor",
    "low priority",
    "no functional impact",
    "edge case",
    "rarely",
    "refresh",
)

_COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontend": (
        "react", "vue", "angular", "css", "flexbox", "layout", "button",
        "tooltip", "navbar", "dark mode", "browser", "chrome", "safari",
        "mobile view", "component", "render", "white screen",
    ),
    "backend": (
        "java", "python service", "service crashes", "business logic",
        "calculation", "background job", "worker", "nullpointerexception",
        "processor", "batch",
    ),
    "api": (
        "endpoint", "rest", "graphql", "http", "request", "response",
        "payload", "serialization", "rate limit", "mutation", "resolver",
        "/api/",
    ),
    "infrastructure": (
        "kubernetes", "k8s", "pod", "docker", "deploy", "ci/cd", "pipeline",
        "dns", "load balancer", "redis", "sentinel", "failover", "cluster",
        "oomkilled", "restart",
    ),
    "database": (
        "sql", "postgres", "postgresql", "mysql", "query", "index",
        "migration", "schema", "replication", "table", "row-level lock",
        "connection pool",
    ),
    "authentication": (
        "login", "oauth", "sso", "token", "session", "password", "rbac",
        "permission", "auth", "jwt", "sign in", "signed out",
    ),
}


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def parse_stack_trace(report: str) -> str:
    """Extract exception names, file:line references, crash signals, and HTTP
    status codes from a bug report. Useful for grounding type and component
    decisions in technical evidence."""
    exceptions = sorted(set(_EXCEPTION_RE.findall(report)))
    file_refs = [f"{path}:{line}" for path, line in _FILE_LINE_RE.findall(report)]
    http_codes = sorted(set(_HTTP_CODE_RE.findall(report)))
    crash_signals = sorted({match.group(0) for match in _SIGNAL_RE.finditer(report)})
    return _json(
        {
            "exceptions": exceptions,
            "file_references": file_refs,
            "http_status_codes": http_codes,
            "crash_signals": crash_signals,
            "has_technical_evidence": bool(
                exceptions or file_refs or http_codes or crash_signals
            ),
        }
    )


@tool
def scan_severity_signals(report: str) -> str:
    """Scan a bug report for blast-radius and urgency signals: outage phrases,
    affected-user scope, urgency language, and mitigating factors such as
    workarounds or cosmetic-only impact."""
    lowered = report.lower()
    broad = [phrase for phrase in _BROAD_IMPACT_PHRASES if phrase in lowered]
    urgency = [phrase for phrase in _URGENCY_PHRASES if phrase in lowered]
    mitigation = [phrase for phrase in _MITIGATION_PHRASES if phrase in lowered]
    percentages = re.findall(r"\b(\d{1,3})\s?%", report)

    if broad and not mitigation:
        suggestion = "signals point to critical/high severity (P0-P1)"
    elif mitigation and not broad:
        suggestion = "signals point to lower severity (P2-P3)"
    else:
        suggestion = "mixed or weak signals; weigh report details"

    return _json(
        {
            "broad_impact_signals": broad,
            "urgency_signals": urgency,
            "mitigation_signals": mitigation,
            "percentages_mentioned": percentages,
            "heuristic": suggestion,
        }
    )


@tool
def match_component_keywords(report: str) -> str:
    """Match technology keywords in a bug report against the configured
    engineering components; returns per-component keyword hits and scores."""
    lowered = report.lower()
    scores: dict[str, dict[str, Any]] = {}
    for component in config.BUG_COMPONENTS:
        keywords = _COMPONENT_KEYWORDS.get(component, ())
        hits = [keyword for keyword in keywords if keyword in lowered]
        if hits:
            scores[component] = {"score": len(hits), "matched_keywords": hits}
    best = max(scores, key=lambda name: scores[name]["score"]) if scores else None
    return _json(
        {
            "component_matches": scores,
            "best_match": best,
            "valid_components": list(config.BUG_COMPONENTS),
        }
    )


@tool
def search_similar_bugs(query: str) -> str:
    """Search the Notion bug database for previously classified tickets whose
    titles match the query, returning their type/severity/component labels.
    Returns an empty result set when history is unavailable (e.g. dry-run)."""
    if config.NOTION_DRY_RUN:
        return _json(
            {"results": [], "note": "bug history unavailable (NOTION_DRY_RUN)"}
        )
    try:
        from bug_classifier.integrations.notion_logger import (
            _get_client,
            _resolve_data_source_id,
        )

        client = _get_client()
        data_source_id = _resolve_data_source_id(client, config.NOTION_DATABASE_ID)
        response = client.data_sources.query(
            data_source_id,
            filter={
                "property": "title",
                "title": {"contains": query[:100]},
            },
            page_size=5,
        )
        results = []
        for page in response.get("results", []):
            properties = page.get("properties", {})

            def _select(name: str) -> str | None:
                option = (properties.get(name) or {}).get("select") or {}
                return option.get("name")

            title_parts = (properties.get("Title") or {}).get("title") or []
            title = "".join(
                part.get("plain_text", "") for part in title_parts
            )
            results.append(
                {
                    "title": title[:120],
                    "bug_type": _select("Bug Type"),
                    "severity": _select("Severity"),
                    "component": _select("Component"),
                }
            )
        return _json({"results": results, "note": f"{len(results)} similar tickets"})
    except Exception as exc:
        logger.warning("search_similar_bugs failed: %s", exc)
        return _json({"results": [], "note": f"bug history lookup failed: {exc}"})


# Tool assignments per specialist.
TYPE_TOOLS = [parse_stack_trace, scan_severity_signals]
SEVERITY_TOOLS = [scan_severity_signals, parse_stack_trace, search_similar_bugs]
COMPONENT_TOOLS = [match_component_keywords, parse_stack_trace]


# ---------------------------------------------------------------------------
# Shared bounded tool-calling loop
# ---------------------------------------------------------------------------
def run_tool_loop(
    llm: Any,
    tools: list[Any],
    messages: list[BaseMessage],
    *,
    agent_name: str,
    max_rounds: int | None = None,
) -> tuple[list[BaseMessage], list[str]]:
    """Let the LLM investigate with tools before its final structured answer.

    Args:
        llm: Chat model supporting ``bind_tools``.
        tools: Tools the agent may call.
        messages: Conversation so far (system + human context).
        agent_name: For logging.
        max_rounds: Max tool-calling iterations (default
            ``config.AGENT_TOOL_ROUNDS``).

    Returns:
        ``(transcript, tools_used)`` — the tool-call/observation messages to
        append before the final structured-output call, and the names of tools
        invoked. Both are empty if the model never requests a tool or the
        model does not support tool binding.
    """
    rounds = config.AGENT_TOOL_ROUNDS if max_rounds is None else max_rounds
    transcript: list[BaseMessage] = []
    tools_used: list[str] = []

    try:
        bound = llm.bind_tools(tools)
    except Exception:
        logger.debug("[%s] model does not support tool binding; skipping.", agent_name)
        return transcript, tools_used

    tool_map = {t.name: t for t in tools}
    conversation = list(messages)

    for _ in range(rounds):
        try:
            response = bound.invoke(conversation)
        except Exception:
            logger.warning("[%s] tool-loop LLM call failed.", agent_name, exc_info=True)
            break

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        if not isinstance(response, AIMessage):  # defensive: mocks/adapters
            response = AIMessage(content="", tool_calls=tool_calls)
        conversation.append(response)
        transcript.append(response)

        for call in tool_calls:
            name = call.get("name", "")
            selected = tool_map.get(name)
            if selected is None:
                output = f"Unknown tool: {name}"
            else:
                try:
                    output = selected.invoke(call.get("args") or {})
                except Exception as exc:
                    output = f"Tool {name} failed: {exc}"
            observation = ToolMessage(
                content=str(output), tool_call_id=call.get("id") or name
            )
            conversation.append(observation)
            transcript.append(observation)
            tools_used.append(name)
            logger.info("[%s] tool call: %s", agent_name, name)

    return transcript, tools_used
