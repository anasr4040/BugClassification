"""Supervisor agent: reviews the combined classification before ticket creation.

This node is what makes the pipeline a multi-agent *system* rather than a fixed
workflow: after the specialist agents run, the supervisor inspects their
combined output and decides — at runtime — whether to accept it or to send a
specific dimension back for reclassification with targeted feedback.

Design:

1. **Rule-based pre-checks (cheap, deterministic).** Low per-dimension
   confidence and suspicious cross-dimension combinations (e.g. a P3 security
   bug) flag the classification for deeper review. When nothing is flagged the
   supervisor approves without an LLM call, so the happy path costs nothing.

2. **LLM review (only when flagged).** A reviewer LLM sees the original report,
   the classifications, and the flagged issues, and returns a structured
   verdict: approve, or re-run one dimension with a hint.

3. **Bounded feedback loops.** Each dimension may be reclassified at most
   ``MAX_RETRIES_PER_DIMENSION`` times. When the budget is exhausted (or the
   reviewer LLM fails) the supervisor approves gracefully and sets
   ``needs_review`` so a human double-checks the ticket.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES_PER_DIMENSION = 1
DIMENSIONS = ("type", "severity", "component")

VERDICT_APPROVED = "approved"
VERDICT_RECLASSIFY = "reclassify"

# (bug_type, severity) pairs that rarely make sense together. These do not
# force a reclassification by themselves — they trigger the LLM review, which
# decides whether the combination is justified by the report.
SUSPICIOUS_COMBINATIONS: dict[tuple[str, str], str] = {
    ("security", "P3"): (
        "Security bugs are rarely low priority; P3 may understate the risk."
    ),
    ("security", "P2"): (
        "Security bugs usually warrant P0/P1; verify P2 is justified."
    ),
    ("ui", "P0"): (
        "UI bugs rarely justify paging on-call; P0 may overstate the impact."
    ),
    ("crash", "P3"): (
        "A crash marked P3 is unusual; verify it truly affects only an edge case."
    ),
}

ReviewDimension = Literal["type", "severity", "component", "none"]

SYSTEM_PROMPT = """You are a bug classification supervisor. Specialist agents have classified a bug report by type, severity, and component. Automated checks flagged potential issues with their combined output. Your ONLY task is to decide whether the classification is acceptable or whether exactly ONE dimension should be reclassified.

Rules:
- Base your judgment on evidence in the original bug report.
- Approve when the classification is defensible, even if imperfect.
- Request reclassification ONLY when a dimension clearly contradicts the report or another dimension (e.g. a confirmed SQL injection classified as "logic", or an actively exploited breach marked P3).
- Pick the single dimension whose correction matters most; do not request multiple.
- When requesting reclassification, write a short actionable hint quoting the evidence the specialist likely missed.

Respond ONLY with valid JSON.

Required JSON shape:
{"approved": <true|false>, "problem_dimension": "<type|severity|component|none>", "hint": "<actionable feedback, empty if approved>", "reasoning": "<one sentence>"}"""


class SupervisorReview(BaseModel):
    """Structured LLM output for the supervisor's verdict."""

    approved: bool
    problem_dimension: ReviewDimension = "none"
    hint: str = ""
    reasoning: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def hint_for_dimension(state: BugState, dimension: str) -> str:
    """Return supervisor feedback for an agent re-running its classification.

    Empty string unless the supervisor targeted ``dimension`` on this pass.
    """
    if state.get("reclassify_target") == dimension and state.get("reclassify_hint"):
        return (
            "\n\nA supervisor reviewed a previous classification attempt and "
            f"asked you to reconsider: {state['reclassify_hint']}"
        )
    return ""


def _flagged_issues(state: BugState) -> list[str]:
    """Cheap deterministic checks that gate the LLM review."""
    issues: list[str] = []
    scores = dict(state.get("confidence_scores") or {})

    for dimension in DIMENSIONS:
        score = scores.get(dimension)
        if score is not None and score < CONFIDENCE_THRESHOLD:
            issues.append(
                f"Low confidence on {dimension}: {score:.0%} "
                f"(threshold {CONFIDENCE_THRESHOLD:.0%})."
            )

    combination = (state.get("bug_type") or "", state.get("severity") or "")
    if combination in SUSPICIOUS_COMBINATIONS:
        issues.append(
            f"Suspicious combination {combination[0]}/{combination[1]}: "
            f"{SUSPICIOUS_COMBINATIONS[combination]}"
        )

    return issues


def _approve(
    retry_counts: dict[str, int],
    *,
    notes: str,
    needs_review: bool,
) -> dict:
    return {
        "supervisor_verdict": VERDICT_APPROVED,
        "reclassify_target": None,
        "reclassify_hint": None,
        "retry_counts": retry_counts,
        "needs_review": needs_review,
        "supervisor_notes": notes,
    }


def _build_review_message(state: BugState, issues: list[str]) -> str:
    scores = dict(state.get("confidence_scores") or {})
    score_lines = "\n".join(
        f"- {key}: {value:.2f}" for key, value in scores.items()
    ) or "- (none)"
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        f"Classification under review:\n"
        f"- Type: {state.get('bug_type') or 'unknown'}\n"
        f"- Severity: {state.get('severity') or 'unknown'}\n"
        f"- Component: {state.get('component') or 'unknown'}\n\n"
        f"Confidence scores:\n{score_lines}\n\n"
        f"Flagged issues:\n{issue_lines}\n\n"
        f"Original bug report:\n{state['raw_report']}"
    )


def _review_with_llm(state: BugState, issues: list[str]) -> SupervisorReview:
    llm = _build_llm().with_structured_output(SupervisorReview)
    result = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_review_message(state, issues)),
        ]
    )
    if not isinstance(result, SupervisorReview):
        result = SupervisorReview.model_validate(result)
    return result


@trace_agent("supervisor")
def supervise(state: BugState) -> dict:
    """Review combined classification; approve or request one reclassification.

    Returns only fields owned by this agent: ``supervisor_verdict``,
    ``reclassify_target``, ``reclassify_hint``, ``retry_counts``,
    ``needs_review``, and ``supervisor_notes``.
    """
    retry_counts = dict(state.get("retry_counts") or {})
    issues = _flagged_issues(state)

    if not issues:
        logger.info("Supervisor: no issues flagged; approving classification.")
        return _approve(
            retry_counts,
            notes="All confidence and consistency checks passed.",
            needs_review=False,
        )

    logger.info("Supervisor: %d issue(s) flagged; consulting reviewer LLM.", len(issues))

    try:
        review = _review_with_llm(state, issues)
    except Exception:
        logger.warning(
            "Supervisor LLM review failed; approving with needs_review=True.",
            exc_info=True,
        )
        return _approve(
            retry_counts,
            notes="Supervisor review unavailable; flagged issues: " + " ".join(issues),
            needs_review=True,
        )

    target = review.problem_dimension
    if review.approved or target == "none" or target not in DIMENSIONS:
        return _approve(
            retry_counts,
            notes=review.reasoning or "Approved by supervisor review.",
            needs_review=True,  # flags existed — keep a human in the loop
        )

    if retry_counts.get(target, 0) >= MAX_RETRIES_PER_DIMENSION:
        logger.warning(
            "Supervisor wanted to reclassify %r but the retry budget is exhausted; "
            "approving with needs_review=True.",
            target,
        )
        return _approve(
            retry_counts,
            notes=(
                f"Reclassification budget exhausted for '{target}'. "
                f"Unresolved concern: {review.reasoning}"
            ),
            needs_review=True,
        )

    retry_counts[target] = retry_counts.get(target, 0) + 1
    logger.info(
        "Supervisor: requesting reclassification of %r (attempt %d). Hint: %s",
        target,
        retry_counts[target],
        review.hint,
    )
    return {
        "supervisor_verdict": VERDICT_RECLASSIFY,
        "reclassify_target": target,
        "reclassify_hint": review.hint or review.reasoning,
        "retry_counts": retry_counts,
        "needs_review": None,
        "supervisor_notes": review.reasoning,
    }


def supervisor_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`supervise`."""
    return supervise(state)
