import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.7
SCORE_KEYS = ("type", "severity", "component")

SUGGESTED_ACTIONS = {
    "P0": "Page on-call team immediately.",
    "P1": "Escalate to the owning team for same-day resolution.",
    "P2": "Schedule a fix within the next 1-2 days.",
    "P3": "Add to backlog for next sprint.",
}

SYSTEM_PROMPT = """You are a bug ticket summary writer. Synthesize classification results and the original report into a clean, structured summary for engineering ticketing (e.g., Notion).

Produce output in exactly this markdown format:

**Classification Result**
- Type: {bug_type}
- Severity: {severity}
- Component: {component}

**Summary**: A 2-3 sentence description of the bug suitable for an engineering ticket title and description.

**Suggested Action**: One sentence recommending next steps based on the severity (P0 = page on-call immediately; P1 = same-day response; P2 = fix within 1-2 days; P3 = add to backlog for next sprint).

**Confidence**: Overall confidence assessment based on individual agent scores. List each score (type, severity, component). If any score is below 0.7, explicitly recommend manual review for those dimensions.

Use the provided classification values as authoritative. Do not invent new types, severities, or components. Write clearly for engineers triaging the ticket."""


class BugTicketSummary(BaseModel):
    """Structured LLM output containing the full formatted ticket summary."""

    summary: str = Field(
        description="Complete markdown summary in the required ticket format."
    )


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def _confidence_scores(state: BugState) -> dict[str, float]:
    return dict(state.get("confidence_scores") or {})


def _low_confidence_dimensions(scores: dict[str, float]) -> list[str]:
    return [
        key
        for key in SCORE_KEYS
        if key in scores and scores[key] < HIGH_CONFIDENCE_THRESHOLD
    ]


def _overall_confidence(scores: dict[str, float]) -> float:
    values = [scores[key] for key in SCORE_KEYS if key in scores]
    return min(values) if values else 0.0


def _has_complete_classification(state: BugState) -> bool:
    return bool(
        state.get("bug_type")
        and state.get("severity")
        and state.get("component")
    )


def _has_high_confidence(scores: dict[str, float]) -> bool:
    return all(
        scores.get(key, 0.0) >= HIGH_CONFIDENCE_THRESHOLD for key in SCORE_KEYS
    )


def _can_format_directly(state: BugState) -> bool:
    scores = _confidence_scores(state)
    return _has_complete_classification(state) and _has_high_confidence(scores)


def _brief_description(raw_report: str, max_sentences: int = 3) -> str:
    text = " ".join(raw_report.split())
    if not text:
        return "No bug description provided."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    excerpt = " ".join(sentences[:max_sentences]).strip()
    return excerpt or text[:500]


def _suggested_action(severity: str) -> str:
    return SUGGESTED_ACTIONS.get(severity, "Review and prioritize with the owning team.")


def _confidence_block(scores: dict[str, float]) -> str:
    lines = [
        f"- {key}: {scores[key]:.0%}"
        for key in SCORE_KEYS
        if key in scores
    ]
    overall = _overall_confidence(scores)
    lines.append(f"- Overall: {overall:.0%}")

    block = "\n".join(lines)
    low = _low_confidence_dimensions(scores)
    if low:
        labels = ", ".join(low)
        block += (
            f"\n\n**Manual review recommended** — low confidence on: {labels}."
        )
    return block


def _supervisor_block(state: BugState) -> str:
    """Surface the supervisor's verdict in the ticket when it matters."""
    if state.get("needs_review") and state.get("supervisor_notes"):
        return (
            f"\n\n**Supervisor**: Manual review requested — "
            f"{state['supervisor_notes']}"
        )
    return ""


def _format_summary_direct(state: BugState) -> str:
    """Build the ticket summary without an LLM call."""
    scores = _confidence_scores(state)
    bug_type = state["bug_type"]  # type: ignore[typeddict-item]
    severity = state["severity"]  # type: ignore[typeddict-item]
    component = state["component"]  # type: ignore[typeddict-item]

    return (
        "**Classification Result**\n"
        f"- Type: {bug_type}\n"
        f"- Severity: {severity}\n"
        f"- Component: {component}\n\n"
        f"**Summary**: {_brief_description(state['raw_report'])}\n\n"
        f"**Suggested Action**: {_suggested_action(severity)}\n\n"
        f"**Confidence**:\n{_confidence_block(scores)}"
        f"{_supervisor_block(state)}"
    )


def _build_llm_context(state: BugState) -> str:
    scores = _confidence_scores(state)
    score_lines = "\n".join(
        f"- {key}: {value:.2f}" for key, value in scores.items()
    ) or "- (none)"
    supervisor_note = ""
    if state.get("supervisor_notes"):
        supervisor_note = f"\nSupervisor review: {state['supervisor_notes']}"
        if state.get("needs_review"):
            supervisor_note += " (manual review recommended — mention this)"
    return (
        f"Bug type: {state.get('bug_type') or 'unknown'}\n"
        f"Severity: {state.get('severity') or 'unknown'}\n"
        f"Component: {state.get('component') or 'unknown'}\n"
        f"{supervisor_note}\n"
        f"Confidence scores:\n{score_lines}\n\n"
        f"Original report:\n{state['raw_report']}"
    )


def _summarize_with_llm(state: BugState) -> str:
    llm = _build_llm().with_structured_output(BugTicketSummary)
    result = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_llm_context(state)),
        ]
    )
    if not isinstance(result, BugTicketSummary):
        result = BugTicketSummary.model_validate(result)
    return result.summary.strip()


@trace_agent("summary_agent")
def summarize(state: BugState) -> dict:
    """Produce a structured bug ticket summary from full classification state.

    Returns only the field owned by this agent: ``summary``.
    """
    if _can_format_directly(state):
        return {"summary": _format_summary_direct(state)}

    try:
        return {"summary": _summarize_with_llm(state)}
    except Exception:
        logger.warning(
            "LLM summary synthesis failed; falling back to direct formatting.",
            exc_info=True,
        )
        if _has_complete_classification(state):
            return {"summary": _format_summary_direct(state)}
        return {
            "summary": (
                "**Classification Result**\n"
                f"- Type: {state.get('bug_type') or 'unknown'}\n"
                f"- Severity: {state.get('severity') or 'unknown'}\n"
                f"- Component: {state.get('component') or 'unknown'}\n\n"
                f"**Summary**: {_brief_description(state['raw_report'])}\n\n"
                "**Suggested Action**: Review classification outputs and assign manually.\n\n"
                "**Confidence**:\n"
                f"{_confidence_block(_confidence_scores(state))}"
                f"{_supervisor_block(state)}"
            )
        }


def summary_agent_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`summarize`."""
    return summarize(state)
