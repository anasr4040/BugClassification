"""Shared LangGraph state for the multi-agent bug classification pipeline.

Every agent reads from and writes to a single ``BugState`` instance (as dict
merges between steps). This object is the communication backbone of the graph.
"""

from typing import NotRequired, Required, TypedDict


class BugState(TypedDict, total=False):
    """State carried through the bug classification graph.

    Agents receive the accumulated state and return partial updates (dicts)
    that LangGraph merges into this structure.

    Fields:
        raw_report:
            Original bug report text from a user or developer. Set once at
            graph entry (via :func:`create_initial_state`); agents should not
            overwrite it unless correcting input.

        bug_type:
            One of ``\"crash\"``, ``\"performance\"``, ``\"logic\"``,
            ``\"security\"``, ``\"ui\"``. Set by the **type classifier** agent
            (``agents/type_classifier.py``).

        severity:
            One of ``\"P0\"``, ``\"P1\"``, ``\"P2\"``, ``\"P3\"``. Set by the
            **severity assessor** agent (``agents/severity_assessor.py``).

        component:
            One of ``\"backend\"``, ``\"frontend\"``, ``\"api\"``,
            ``\"infrastructure\"``, ``\"database\"``, ``\"authentication\"``.
            Set by the **component identifier** agent
            (``agents/component_identifier.py``).

        summary:
            Structured summary combining classification outputs. Set by the
            **summary** agent (``agents/summary_agent.py``).

        ticket_url:
            URL of the Notion page created for this classification. Set by the
            **Notion logger** (``integrations/notion_logger.py``) after a page
            is created.

        confidence_scores:
            Optional per-head confidences, e.g.
            ``{\"type\": 0.92, \"severity\": 0.85, \"component\": 0.78}``.
            May be written by any agent that produces calibrated scores.

        error:
            Human-readable processing or integration failure message. Any agent
            or integration may set this when aborting or degrading gracefully.
    """

    raw_report: Required[str]
    bug_type: NotRequired[str | None]
    severity: NotRequired[str | None]
    component: NotRequired[str | None]
    summary: NotRequired[str | None]
    ticket_url: NotRequired[str | None]
    confidence_scores: NotRequired[dict[str, float] | None]
    error: NotRequired[str | None]


def create_initial_state(raw_report: str) -> BugState:
    """Build starting graph state with only ``raw_report`` populated.

    All classification and downstream fields are explicitly set to ``None``
    so later nodes can distinguish \"not yet run\" from legitimate values.

    Args:
        raw_report: Verbatim bug report text submitted for classification.

    Returns:
        A ``BugState`` ready to pass to the compiled LangGraph ``invoke`` /
        ``ainvoke`` entrypoint.
    """
    return {
        "raw_report": raw_report,
        "bug_type": None,
        "severity": None,
        "component": None,
        "summary": None,
        "ticket_url": None,
        "confidence_scores": None,
        "error": None,
    }
