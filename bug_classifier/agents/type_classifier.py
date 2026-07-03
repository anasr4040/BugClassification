import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field

from bug_classifier import config
from bug_classifier.agents.tools import TYPE_TOOLS, run_tool_loop
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

AGENT_NAME = "type_classifier"

VALID_BUG_TYPES = ("crash", "performance", "logic", "security", "ui")
BugType = Literal["crash", "performance", "logic", "security", "ui"]

SYSTEM_PROMPT = """You are a specialized bug type classifier agent in a multi-agent triage system. Your ONLY task is to classify bug reports.

Valid categories:
- crash: Application terminates unexpectedly, segfaults, unhandled exceptions, OOM kills
- performance: Slowness, high latency, memory leaks, CPU spikes, timeouts
- logic: Incorrect behavior, wrong calculations, data corruption, unexpected outputs
- security: Vulnerabilities, auth bypass, injection, data exposure, permission issues
- ui: Visual glitches, layout broken, accessibility issues, wrong text, styling bugs

You may call the provided tools (stack-trace parsing, severity-signal scanning) to gather technical evidence before deciding. Use tool observations to ground your classification and confidence.

If a supervisor reviewer provided revision feedback, take it seriously and re-examine the evidence rather than repeating your previous answer.

Do NOT assess severity. Do NOT identify components. Do NOT write a summary. Your final answer MUST be the structured classification.

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
        "agent_notes": [
            {
                "agent": AGENT_NAME,
                "note": (
                    "LLM classification failed; defaulted to 'logic' with "
                    "confidence 0.5 — treat as unverified."
                ),
            }
        ],
    }


def _build_human_message(state: BugState) -> str:
    feedback = (state.get("revision_feedback") or {}).get("type")
    if not feedback:
        return state["raw_report"]
    return (
        f"{state['raw_report']}\n\n"
        f"Supervisor revision feedback on your previous classification: {feedback}"
    )


@trace_agent(AGENT_NAME)
def classify_type(state: BugState) -> dict:
    """Classify ``raw_report`` into exactly one bug type.

    Tool-using agent: may call investigation tools before committing to its
    structured answer. Returns only fields owned by this agent: ``bug_type``,
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
            llm, TYPE_TOOLS, messages, agent_name=AGENT_NAME
        )
        result = llm.with_structured_output(TypeClassification).invoke(
            messages + transcript
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
        return {
            "bug_type": bug_type,
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
            "Type classification failed; defaulting to 'logic' with confidence 0.5.",
            exc_info=True,
        )
        return _default_classification(existing_scores)


def type_classifier_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`classify_type`."""
    return classify_type(state)
