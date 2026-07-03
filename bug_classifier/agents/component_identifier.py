import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field, field_validator

from bug_classifier import config
from bug_classifier.agents.tools import COMPONENT_TOOLS, run_tool_loop
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

AGENT_NAME = "component_identifier"

_VALID_COMPONENTS = frozenset(config.BUG_COMPONENTS)


def _build_system_prompt() -> str:
    component_lines = "\n".join(
        f"- {name}: {description}"
        for name, description in config.COMPONENT_DESCRIPTIONS.items()
    )
    valid_list = ", ".join(config.BUG_COMPONENTS)
    return f"""You are a specialized component identifier agent in a multi-agent triage system. Your ONLY task is to identify which engineering component or team area is affected by a bug report.

Valid components (engineering team boundaries):
{component_lines}

You may call the provided tools (technology-keyword matching, stack-trace parsing) to gather technical evidence before deciding.

Look for technical signals in the bug report:
- Stack traces mentioning specific files or services
- Keywords like "endpoint", "query", "login", "deploy"
- Error codes or HTTP status codes
- References to specific technologies (e.g., "React" → frontend, "PostgreSQL" → database)

Use bug_type and severity from parallel classifiers as supporting context when available, but base your decision on technical evidence in the report.

If a supervisor reviewer provided revision feedback, take it seriously and re-examine the evidence rather than repeating your previous answer.

You are ONLY identifying the affected component. Do NOT reassess severity or reclassify the bug type. Do NOT write a summary. Your final answer MUST be the structured identification.

Required JSON shape:
{{"component": "<one of: {valid_list}>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


class ComponentIdentification(BaseModel):
    """Structured LLM output for component identification."""

    component: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("component")
    @classmethod
    def validate_component(cls, value: str) -> str:
        if value not in _VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {', '.join(config.BUG_COMPONENTS)}"
            )
        return value


def _build_llm() -> ChatXAI:
    return ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
        api_key=config.XAI_API_KEY,
    )


def _default_identification(existing_scores: dict[str, float]) -> dict:
    existing_scores["component"] = 0.5
    return {
        "component": config.DEFAULT_COMPONENT,
        "confidence_scores": existing_scores,
        "agent_notes": [
            {
                "agent": AGENT_NAME,
                "note": (
                    f"LLM identification failed; defaulted to "
                    f"{config.DEFAULT_COMPONENT!r} with confidence 0.5 — "
                    "treat as unverified."
                ),
            }
        ],
    }


def _build_human_message(state: BugState) -> str:
    bug_type = state.get("bug_type") or "unknown"
    severity = state.get("severity") or "unknown"
    feedback = (state.get("revision_feedback") or {}).get("component")
    feedback_block = (
        f"Supervisor revision feedback on your previous identification: {feedback}\n\n"
        if feedback
        else ""
    )
    return (
        f"Bug type: {bug_type}\n"
        f"Severity: {severity}\n\n"
        f"{feedback_block}"
        f"Bug report:\n{state['raw_report']}"
    )


@trace_agent(AGENT_NAME)
def identify_component(state: BugState) -> dict:
    """Identify the affected engineering component from state context.

    Tool-using agent: may call investigation tools before committing to its
    structured answer. Returns only fields owned by this agent: ``component``,
    ``confidence_scores``, and its ``agent_notes`` entry.
    """
    existing_scores = dict(state.get("confidence_scores") or {})

    try:
        llm = _build_llm()
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_human_message(state)),
        ]
        transcript, tools_used = run_tool_loop(
            llm, COMPONENT_TOOLS, messages, agent_name=AGENT_NAME
        )
        result = llm.with_structured_output(ComponentIdentification).invoke(
            messages + transcript
        )

        if not isinstance(result, ComponentIdentification):
            result = ComponentIdentification.model_validate(result)

        component = result.component
        if component not in _VALID_COMPONENTS:
            logger.warning(
                "Component identifier returned unexpected component %r; defaulting to %r.",
                component,
                config.DEFAULT_COMPONENT,
            )
            return _default_identification(existing_scores)

        confidence = result.confidence
        if not 0.0 <= confidence <= 1.0:
            logger.warning(
                "Component identifier returned out-of-range confidence %r; defaulting to 0.5.",
                confidence,
            )
            confidence = 0.5

        existing_scores["component"] = confidence
        return {
            "component": component,
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
            "Component identification failed; defaulting to %r with confidence 0.5.",
            config.DEFAULT_COMPONENT,
            exc_info=True,
        )
        return _default_identification(existing_scores)


def component_identifier_node(state: BugState) -> dict:
    """LangGraph node wrapper for :func:`identify_component`."""
    return identify_component(state)
