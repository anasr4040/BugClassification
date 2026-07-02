import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field, field_validator

from bug_classifier import config
from bug_classifier.agents.supervisor import hint_for_dimension
from bug_classifier.observability import trace_agent
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

_VALID_COMPONENTS = frozenset(config.BUG_COMPONENTS)


def _build_system_prompt() -> str:
    component_lines = "\n".join(
        f"- {name}: {description}"
        for name, description in config.COMPONENT_DESCRIPTIONS.items()
    )
    valid_list = ", ".join(config.BUG_COMPONENTS)
    return f"""You are a specialized component identifier. Your ONLY task is to identify which engineering component or team area is affected by a bug report.

Valid components (engineering team boundaries):
{component_lines}

Look for technical signals in the bug report:
- Stack traces mentioning specific files or services
- Keywords like "endpoint", "query", "login", "deploy"
- Error codes or HTTP status codes
- References to specific technologies (e.g., "React" → frontend, "PostgreSQL" → database)

Use bug_type and severity from prior classification as supporting context, but base your decision on technical evidence in the report.

You are ONLY identifying the affected component. Do NOT reassess severity or reclassify the bug type. Do NOT write a summary. Respond ONLY with valid JSON.

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
    }


def _build_human_message(state: BugState) -> str:
    bug_type = state.get("bug_type") or "unknown"
    severity = state.get("severity") or "unknown"
    return (
        f"Bug type: {bug_type}\n"
        f"Severity: {severity}\n\n"
        f"Bug report:\n{state['raw_report']}"
        f"{hint_for_dimension(state, 'component')}"
    )


@trace_agent("component_identifier")
def identify_component(state: BugState) -> dict:
    """Identify the affected engineering component from state context.

    Returns only fields owned by this agent: ``component`` and ``confidence_scores``.
    """
    existing_scores = dict(state.get("confidence_scores") or {})

    try:
        llm = _build_llm().with_structured_output(ComponentIdentification)
        result = llm.invoke(
            [
                SystemMessage(content=_build_system_prompt()),
                HumanMessage(content=_build_human_message(state)),
            ]
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
        return {"component": component, "confidence_scores": existing_scores}

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
