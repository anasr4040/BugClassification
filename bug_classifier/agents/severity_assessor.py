import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.agents.tools import SEVERITY_TOOLS, run_tool_loop
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

AGENT_NAME = "severity_assessor"

VALID_SEVERITIES = ("P0", "P1", "P2", "P3")
SeverityLevel = Literal["P0", "P1", "P2", "P3"]

SYSTEM_PROMPT = """You are a specialized bug severity assessor agent in a multi-agent triage system. Your ONLY task is to assess the severity/priority of bug reports.

Severity definitions:
- P0 (Critical): System-wide outage, data loss, security breach actively exploited, complete feature failure affecting all users. Requires immediate response.
- P1 (High): Major feature broken, no workaround available, significant user impact, data integrity at risk. Requires same-day response.
- P2 (Medium): Feature degraded but workaround exists, affects subset of users, non-critical data issues. Requires response within 1-2 days.
- P3 (Low): Cosmetic issues, minor inconveniences, edge cases, documentation errors. Can be scheduled for next sprint.

You may call the provided tools (blast-radius signal scanning, stack-trace parsing, similar-bug history lookup) to gather evidence about impact scope before deciding.

If a bug type from a parallel classifier is provided, consider it: security bugs generally warrant a higher baseline severity. A crash affecting all users is typically P0, but a crash on an obscure device or rare edge case may be P2. Performance issues range from P1 (severe latency for everyone) to P3 (minor slowness in a rarely used view). UI issues are often P3 unless they block critical workflows.

If a supervisor reviewer provided revision feedback, take it seriously and re-examine the evidence rather than repeating your previous answer.

You are ONLY assessing severity. Do NOT reclassify the bug type. Do NOT identify components. Do NOT write a summary. Your final answer MUST be the structured assessment.

Required JSON shape:
{"severity": "<P0|P1|P2|P3>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}"""


class SeverityAssessment(BaseModel):
    """Structured LLM output for severity assessment."""

    severity: SeverityLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def _default_assessment(existing_scores: dict[str, float]) -> dict:
    existing_scores["severity"] = 0.5
    return {
        "severity": "P2",
        "confidence_scores": existing_scores,
        "agent_notes": [
            {
                "agent": AGENT_NAME,
                "note": (
                    "LLM assessment failed; defaulted to 'P2' with confidence "
                    "0.5 — treat as unverified."
                ),
            }
        ],
    }


def _build_human_message(state: BugState) -> str:
    bug_type = state.get("bug_type") or "unknown"
    feedback = (state.get("revision_feedback") or {}).get("severity")
    feedback_block = (
        f"Supervisor revision feedback on your previous assessment: {feedback}\n\n"
        if feedback
        else ""
    )
    return (
        f"Bug type (from parallel classifier, may be unknown): {bug_type}\n\n"
        f"{feedback_block}"
        f"Bug report:\n{state['raw_report']}"
    )


@trace_agent(AGENT_NAME)
def assess_severity(state: BugState) -> dict:
    """Assess bug severity from ``raw_report`` (and ``bug_type`` when present).

    Tool-using agent: may call investigation tools before committing to its
    structured answer. Returns only fields owned by this agent: ``severity``,
    ``confidence_scores``, and its ``agent_notes`` entry.
    """
    existing_scores = dict(state.get("confidence_scores") or {})

    try:
        llm = _build_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_human_message(state)),
        ]
        transcript, tools_used = run_tool_loop(
            llm, SEVERITY_TOOLS, messages, agent_name=AGENT_NAME
        )
        result = llm.with_structured_output(SeverityAssessment).invoke(
            messages + transcript
        )

        if not isinstance(result, SeverityAssessment):
            result = SeverityAssessment.model_validate(result)

        severity = result.severity
        if severity not in VALID_SEVERITIES:
            logger.warning(
                "Severity assessor returned unexpected severity %r; defaulting to 'P2'.",
                severity,
            )
            return _default_assessment(existing_scores)

        confidence = result.confidence
        if not 0.0 <= confidence <= 1.0:
            logger.warning(
                "Severity assessor returned out-of-range confidence %r; defaulting to 0.5.",
                confidence,
            )
            confidence = 0.5

        existing_scores["severity"] = confidence
        return {
            "severity": severity,
            "confidence_scores": existing_scores,
            "agent_notes": [
                {
                    "agent": AGENT_NAME,
                    "note": result.reasoning,
                    "tools_used": tools_used,
                }
            ],
        }

    except Exception:
        logger.warning(
            "Severity assessment failed; defaulting to 'P2' with confidence 0.5.",
            exc_info=True,
        )
        return _default_assessment(existing_scores)


def severity_assessor_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`assess_severity`."""
    return assess_severity(state)
