import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from bug_classifier.agents.component_identifier import identify_component
from bug_classifier.agents.severity_assessor import assess_severity
from bug_classifier.agents.summary_agent import summarize
from bug_classifier.agents.supervisor import supervise
from bug_classifier.agents.type_classifier import classify_type
from bug_classifier.integrations.notion_logger import log_to_notion
from bug_classifier.observability import log_classification_metrics
from bug_classifier.state import BugState, create_initial_state

logger = logging.getLogger(__name__)

_NODE_LOG_FORMAT = "%(levelname)s | %(message)s"


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_NODE_LOG_FORMAT)


def _logged_node(
    node_name: str, fn: Callable[[BugState], dict[str, Any]]
) -> Callable[[BugState], dict[str, Any]]:
    """Wrap a node function with console transition logging."""

    def wrapper(state: BugState) -> dict[str, Any]:
        logger.info("→ Entering node: %s", node_name)
        try:
            result = fn(state)
            logger.info("← Exiting node: %s", node_name)
            return result
        except Exception:
            logger.exception("✗ Node failed: %s", node_name)
            raise

    return wrapper


def emergency_handler(state: BugState) -> dict[str, Any]:
    """P0 escalation path: alert operators and log to Notion urgently."""
    logger.warning("P0 EMERGENCY — skipping standard pipeline for immediate escalation.")
    print(
        "WARNING: P0 critical bug detected. "
        "Paging on-call and logging to Notion with urgent flag."
    )
    return log_to_notion(state, urgent=True)


def _route_after_severity(state: BugState) -> str:
    if state.get("severity") == "P0":
        logger.info("Branch: P0 severity → emergency_handler")
        return "emergency_handler"
    logger.info("Branch: non-P0 severity → identify_component")
    return "identify_component"


_RECLASSIFY_NODES = {
    "type": "classify_type",
    "severity": "assess_severity",
    "component": "identify_component",
}


def _route_after_supervisor(state: BugState) -> str:
    """Route on the supervisor's verdict: proceed, or loop back to a specialist.

    A reclassified dimension cascades through its downstream agents (a new
    bug_type re-informs severity and component) and returns to the supervisor,
    whose retry budget bounds the loop.
    """
    if state.get("supervisor_verdict") == "reclassify":
        target = state.get("reclassify_target")
        node = _RECLASSIFY_NODES.get(target or "")
        if node:
            logger.info("Branch: supervisor requested reclassification → %s", node)
            return node
        logger.warning(
            "Supervisor requested reclassification of unknown target %r; "
            "proceeding to summarize.",
            target,
        )
    logger.info("Branch: supervisor approved → summarize")
    return "summarize"


def _build_graph() -> StateGraph:
    graph = StateGraph(BugState)

    graph.add_node("classify_type", _logged_node("classify_type", classify_type))
    graph.add_node("assess_severity", _logged_node("assess_severity", assess_severity))
    graph.add_node(
        "identify_component", _logged_node("identify_component", identify_component)
    )
    graph.add_node("supervisor", _logged_node("supervisor", supervise))
    graph.add_node("summarize", _logged_node("summarize", summarize))
    graph.add_node(
        "log_to_notion",
        _logged_node("log_to_notion", lambda state: log_to_notion(state)),
    )
    graph.add_node(
        "emergency_handler", _logged_node("emergency_handler", emergency_handler)
    )

    graph.set_entry_point("classify_type")

    graph.add_edge("classify_type", "assess_severity")
    graph.add_conditional_edges(
        "assess_severity",
        _route_after_severity,
        {
            "emergency_handler": "emergency_handler",
            "identify_component": "identify_component",
        },
    )
    graph.add_edge("identify_component", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "summarize": "summarize",
            "classify_type": "classify_type",
            "assess_severity": "assess_severity",
            "identify_component": "identify_component",
        },
    )
    graph.add_edge("summarize", "log_to_notion")
    graph.add_edge("log_to_notion", END)
    graph.add_edge("emergency_handler", END)

    return graph


def build_graph():
    """Compile and return the bug classification LangGraph application."""
    return _build_graph().compile()


app = build_graph()


def get_graph_visualization() -> str:
    """Return a Mermaid diagram string for the compiled workflow graph."""
    return app.get_graph().draw_mermaid()


def run_classification(bug_report: str) -> BugState:
    """Run the full classification pipeline for a bug report.

    Args:
        bug_report: Raw bug report text from a user or developer.

    Returns:
        Final ``BugState`` after graph execution, or initial state with
        ``error`` set if the pipeline fails.
    """
    _ensure_logging()
    initial = create_initial_state(bug_report)
    logger.info("Starting classification pipeline")

    try:
        final_state = app.invoke(initial)
        log_classification_metrics(final_state)
        logger.info("Classification pipeline completed successfully")
        return final_state
    except Exception as exc:
        logger.exception("Classification pipeline failed")
        return {**initial, "error": str(exc)}
