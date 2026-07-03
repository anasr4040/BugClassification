"""Supervisor/critic agent: reviews joined specialist output and routes.

After the parallel specialists (type, severity, component) complete, the
supervisor reviews the combined classification:

1. **Deterministic checks** find issues: missing dimensions, confidences
   below ``config.CONFIDENCE_REVIEW_THRESHOLD``, and cross-dimension
   consistency rules (e.g. a security bug rated P3).
2. **Fast-path approve**: nothing wrong → approve without an LLM call.
3. **LLM critique**: issues found and revision budget remains → an LLM critic
   writes targeted, per-dimension feedback which is sent back to the flagged
   specialists for another pass (bounded by ``config.MAX_REVISION_ROUNDS``).
   If the critic LLM fails, deterministic rule-based feedback is used instead
   so the loop never blocks.
4. **Approve with flag**: issues remain but the budget is exhausted → approve
   and set ``needs_review`` so a human triages the ticket.

Routing itself lives in ``graph/workflow.py`` (``route_after_supervisor``),
which reads the fields this agent writes.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

AGENT_NAME = "supervisor"

DIMENSIONS = ("type", "severity", "component")
DIMENSION_FIELDS = {
    "type": "bug_type",
    "severity": "severity",
    "component": "component",
}

SYSTEM_PROMPT = """You are the supervising critic in a multi-agent bug triage system. Specialist agents have produced a classification (type, severity, component) with confidence scores, and automated checks flagged one or more dimensions as questionable.

For EACH flagged dimension, write one sentence of concrete, actionable feedback telling that specialist what evidence in the report to re-examine (stack traces, blast radius, technology keywords, mitigating factors). Do not classify the bug yourself; critique so the specialist can do better on its second pass.

Only critique the flagged dimensions listed in the request."""


class RevisionDirective(BaseModel):
    """One targeted critique for a specialist to act on."""

    dimension: Literal["type", "severity", "component"]
    feedback: str = Field(description="One-sentence actionable critique.")


class SupervisorCritique(BaseModel):
    """Structured LLM output for the supervisor's critique."""

    revisions: list[RevisionDirective]
    reasoning: str


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def find_issues(state: BugState) -> dict[str, str]:
    """Deterministic review: return {dimension: rule-based feedback} for
    anything missing, low-confidence, or internally inconsistent."""
    issues: dict[str, str] = {}
    scores = state.get("confidence_scores") or {}

    for dimension in DIMENSIONS:
        value = state.get(DIMENSION_FIELDS[dimension])
        confidence = scores.get(dimension, 0.0)
        if not value:
            issues[dimension] = (
                f"No {dimension} was produced; run the classification again "
                "using the investigation tools."
            )
        elif confidence < config.CONFIDENCE_REVIEW_THRESHOLD:
            issues[dimension] = (
                f"Confidence {confidence:.2f} is below the "
                f"{config.CONFIDENCE_REVIEW_THRESHOLD:.2f} review threshold; "
                "gather more evidence with your tools and re-examine."
            )

    # Cross-dimension consistency rules. The supervisor re-checks the raw
    # evidence itself (via the deterministic signal scanner) and recalls the
    # severity assessor when its rating contradicts what the report says.
    bug_type = state.get("bug_type")
    severity = state.get("severity")

    if "severity" not in issues and bug_type == "security" and severity == "P3":
        issues["severity"] = (
            "A security bug was rated P3; security issues warrant a higher "
            "baseline severity unless clearly trivial — re-assess the exposure."
        )

    if "severity" not in issues and bug_type == "crash" and severity in ("P2", "P3"):
        signals = _severity_signals(state["raw_report"])
        broad = signals.get("broad_impact_signals") or []
        mitigation = signals.get("mitigation_signals") or []
        if broad and not mitigation:
            issues["severity"] = (
                f"A crash was rated {severity}, but the report contains "
                f"broad-impact signals ({', '.join(broad[:3])}) and no "
                "mitigating factors — re-assess whether this is P0/P1."
            )

    if "severity" not in issues and bug_type == "ui" and severity == "P0":
        issues["severity"] = (
            "A UI bug was rated P0; verify the visual issue truly blocks a "
            "critical workflow for all users, otherwise lower the severity."
        )

    return issues


def _severity_signals(report: str) -> dict:
    """Run the deterministic blast-radius scanner the assessor's tool uses."""
    from bug_classifier.agents.tools import scan_severity_signals

    try:
        return json.loads(scan_severity_signals.func(report))  # type: ignore[misc]
    except Exception:
        logger.warning("Supervisor signal scan failed.", exc_info=True)
        return {}


def _format_note(scores: dict) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in sorted(scores.items()))


def _llm_feedback(state: BugState, issues: dict[str, str]) -> dict[str, str]:
    """Ask the LLM critic for richer per-dimension feedback; fall back to the
    deterministic rule feedback on any failure."""
    feedback = dict(issues)
    try:
        scores = state.get("confidence_scores") or {}
        context = (
            f"Flagged dimensions: {', '.join(sorted(issues))}\n"
            f"Automated check findings:\n"
            + "\n".join(f"- {dim}: {msg}" for dim, msg in sorted(issues.items()))
            + "\n\nCurrent classification:\n"
            f"- type: {state.get('bug_type')}\n"
            f"- severity: {state.get('severity')}\n"
            f"- component: {state.get('component')}\n"
            f"- confidences: {_format_note(scores) or '(none)'}\n\n"
            f"Original bug report:\n{state['raw_report']}"
        )
        llm = _build_llm().with_structured_output(SupervisorCritique)
        critique = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]
        )
        if not isinstance(critique, SupervisorCritique):
            critique = SupervisorCritique.model_validate(critique)
        for directive in critique.revisions:
            if directive.dimension in issues and directive.feedback.strip():
                feedback[directive.dimension] = directive.feedback.strip()
    except Exception:
        logger.warning(
            "Supervisor LLM critique failed; using rule-based feedback.",
            exc_info=True,
        )
    return feedback


@trace_agent(AGENT_NAME)
def supervise(state: BugState) -> dict:
    """Review joined specialist output; approve or direct a revision round."""
    issues = find_issues(state)
    current_round = state.get("revision_round") or 0

    if not issues:
        logger.info("Supervisor: classification approved (round %d).", current_round)
        return {
            "supervisor_verdict": "approve",
            "revision_targets": None,
            "needs_review": False,
            "agent_notes": [
                {
                    "agent": AGENT_NAME,
                    "note": (
                        "Approved: all dimensions present, confident, and "
                        "consistent."
                    ),
                }
            ],
        }

    if current_round >= config.MAX_REVISION_ROUNDS:
        flagged = ", ".join(sorted(issues))
        logger.warning(
            "Supervisor: revision budget exhausted; approving with "
            "needs_review for: %s",
            flagged,
        )
        return {
            "supervisor_verdict": "approve",
            "revision_targets": None,
            "needs_review": True,
            "agent_notes": [
                {
                    "agent": AGENT_NAME,
                    "note": (
                        f"Revision budget exhausted after {current_round} "
                        f"round(s); flagged for manual review: {flagged}."
                    ),
                }
            ],
        }

    feedback = _llm_feedback(state, issues)
    targets = sorted(feedback)
    logger.info(
        "Supervisor: requesting revision round %d for: %s",
        current_round + 1,
        ", ".join(targets),
    )
    return {
        "supervisor_verdict": "revise",
        "revision_feedback": feedback,
        "revision_targets": targets,
        "revision_round": current_round + 1,
        "agent_notes": [
            {
                "agent": AGENT_NAME,
                "note": (
                    f"Revision round {current_round + 1} requested for: "
                    f"{', '.join(targets)}."
                ),
            }
        ],
    }


def supervisor_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`supervise`."""
    return supervise(state)
