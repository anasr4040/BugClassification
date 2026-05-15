import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

VALID_BUG_TYPES = ("crash", "performance", "logic", "security", "ui")
BugType = Literal["crash", "performance", "logic", "security", "ui"]

SYSTEM_PROMPT = """You are a specialized bug type classifier. Your ONLY task is to classify bug reports.

Valid categories:
- crash: Application terminates unexpectedly, segfaults, unhandled exceptions, OOM kills
- performance: Slowness, high latency, memory leaks, CPU spikes, timeouts
- logic: Incorrect behavior, wrong calculations, data corruption, unexpected outputs
- security: Vulnerabilities, auth bypass, injection, data exposure, permission issues
- ui: Visual glitches, layout broken, accessibility issues, wrong text, styling bugs

Do NOT assess severity. Do NOT identify components. Do NOT write a summary. Respond ONLY with valid JSON.

Required JSON shape:
{"type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}"""


class TypeClassification(BaseModel):
    """Structured LLM output for bug type classification."""

    type: BugType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def _default_classification(existing_scores: dict[str, float]) -> dict:
    existing_scores["type"] = 0.5
    return {
        "bug_type": "logic",
        "confidence_scores": existing_scores,
    }


@trace_agent("type_classifier")
def classify_type(state: BugState) -> dict:
    """Classify ``raw_report`` into exactly one bug type.

    Returns only fields owned by this agent: ``bug_type`` and ``confidence_scores``.
    """
    raw_report = state["raw_report"]
    existing_scores = dict(state.get("confidence_scores") or {})

    try:
        llm = _build_llm().with_structured_output(TypeClassification)
        result = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=raw_report),
            ]
        )

        if not isinstance(result, TypeClassification):
            result = TypeClassification.model_validate(result)

        bug_type = result.type
        if bug_type not in VALID_BUG_TYPES:
            logger.warning(
                "Type classifier returned unexpected type %r; defaulting to 'logic'.",
                bug_type,
            )
            return _default_classification(existing_scores)

        confidence = result.confidence
        if not 0.0 <= confidence <= 1.0:
            logger.warning(
                "Type classifier returned out-of-range confidence %r; defaulting to 0.5.",
                confidence,
            )
            confidence = 0.5

        existing_scores["type"] = confidence
        return {"bug_type": bug_type, "confidence_scores": existing_scores}

    except Exception:
        logger.warning(
            "Type classification failed; defaulting to 'logic' with confidence 0.5.",
            exc_info=True,
        )
        return _default_classification(existing_scores)


def type_classifier_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`classify_type`."""
    return classify_type(state)
