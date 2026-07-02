"""Multi-agent bug classification graph.

Topology::

              ┌─→ classify_type ──────┐
    START ────┼─→ assess_severity ────┼─→ supervisor ──┬─(revise)──→ flagged specialists ──→ supervisor …
              └─→ identify_component ─┘                ├─(P0)─────→ emergency_handler → END
                                                       └─(approve)→ summarize → log_to_notion → END

- The three specialists run **in parallel**; reducer-annotated state fields
  merge their concurrent writes.
- The **supervisor** reviews the joined result and either approves, routes a
  P0 to the emergency path, or sends targeted feedback back to the flagged
  specialists for a bounded revision round (``config.MAX_REVISION_ROUNDS``).
- The **emergency handler** optionally pauses for human approval via a
  LangGraph interrupt (``config.HITL_EMERGENCY``; requires compiling with a
  checkpointer).
"""

import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from bug_classifier import config
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

SPECIALIST_NODES = ("classify_type", "assess_severity", "identify_component")

# Supervisor revision dimension → specialist node.
DIMENSION_NODES = {
    "type": "classify_type",
    "severity": "assess_severity",
    "component": "identify_component",
}


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
    """P0 escalation path: alert operators and log to Notion urgently.

    With ``HITL_EMERGENCY`` enabled (and a checkpointer on the compiled
    graph), execution pauses on a LangGraph interrupt so a human can approve
    the page-out or downgrade the severity before anything is escalated.
    """
    if config.HITL_EMERGENCY:
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "reason": "P0 emergency escalation requires approval",
                "raw_report": state["raw_report"],
                "bug_type": state.get("bug_type"),
                "component": state.get("component"),
                "options": [
                    {"action": "escalate"},
                    {"action": "downgrade", "severity": "P1|P2|P3"},
                ],
            }
        )
        if isinstance(decision, dict) and decision.get("action") == "downgrade":
            severity = decision.get("severity") or "P1"
            logger.warning("Operator downgraded P0 to %s; logging normally.", severity)
            result = log_to_notion({**state, "severity": severity})
            return {
                "severity": severity,
                **result,
                "agent_notes": [
                    {
                        "agent": "emergency_handler",
                        "note": f"Operator downgraded P0 to {severity}.",
                    }
                ],
            }

    logger.warning("P0 EMERGENCY — skipping standard summary pipeline for immediate escalation.")
    print(
        "WARNING: P0 critical bug detected. "
        "Paging on-call and logging to Notion with urgent flag."
    )
    result = log_to_notion(state, urgent=True)
    return {
        **result,
        "agent_notes": [
            {
                "agent": "emergency_handler",
                "note": "P0 escalation: on-call paged, urgent Notion ticket created.",
            }
        ],
    }


def route_after_supervisor(state: BugState) -> list[str] | str:
    """Dynamic routing based on the supervisor's verdict.

    - ``revise`` → fan out (in parallel) to the specialists the supervisor
      flagged this round.
    - approved P0 → emergency handler.
    - approved otherwise → summary agent.
    """
    if state.get("supervisor_verdict") == "revise":
        targets = [
            DIMENSION_NODES[dimension]
            for dimension in (state.get("revision_targets") or [])
            if dimension in DIMENSION_NODES
        ]
        if targets:
            logger.info("Branch: supervisor revision → %s", ", ".join(targets))
            return targets

    if state.get("severity") == "P0":
        logger.info("Branch: P0 severity → emergency_handler")
        return "emergency_handler"

    logger.info("Branch: approved → summarize")
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
        "emergency_handler",
        _logged_node("emergency_handler", lambda state: emergency_handler(state)),
    )

    # Parallel specialist fan-out from entry.
    for node in SPECIALIST_NODES:
        graph.add_edge(START, node)

    # Join: all specialists report to the supervisor.
    for node in SPECIALIST_NODES:
        graph.add_edge(node, "supervisor")

    # Supervisor decides: revise (loop back), emergency, or summarize.
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        [*SPECIALIST_NODES, "emergency_handler", "summarize"],
    )

    graph.add_edge("summarize", "log_to_notion")
    graph.add_edge("log_to_notion", END)
    graph.add_edge("emergency_handler", END)

    return graph


def build_graph(checkpointer: Any | None = None):
    """Compile and return the bug classification LangGraph application.

    Args:
        checkpointer: Optional LangGraph checkpointer. Required for the
            human-in-the-loop emergency interrupt (``HITL_EMERGENCY``) and for
            resuming paused runs.
    """
    return _build_graph().compile(checkpointer=checkpointer)


app = build_graph()


def get_graph_visualization() -> str:
    """Return a Mermaid diagram string for the compiled workflow graph."""
    return app.get_graph().draw_mermaid()


def run_classification(bug_report: str) -> BugState:
    """Run the full multi-agent classification for a bug report.

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
