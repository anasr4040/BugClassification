"""LangSmith tracing helpers and multi-agent observability utilities.

Viewing traces in LangSmith
---------------------------
1. Enable tracing in ``.env`` (``LANGCHAIN_TRACING_V2=true``, ``LANGCHAIN_API_KEY``, ``LANGCHAIN_PROJECT``).
2. Run the pipeline (``python -m bug_classifier.main`` or ``run_classification(...)``).
3. Open https://smith.langchain.com → Projects → select ``bug-classifier`` (or your project name).
4. Each pipeline run appears as a trace; expand nested spans to see LLM calls inside agents.
5. Filter by tags, latency, or errors to debug slow or failing classifications.

Alerts for low-confidence classifications
---------------------------------------
In LangSmith → Project → **Automations** (or **Alerts**), create a rule on trace metadata
or run feedback, e.g. trigger when ``confidence_scores.type < 0.7`` or when the summary
contains ``Manual review recommended``. Route alerts to Slack/email. Locally,
``log_classification_metrics`` logs average confidence and flags dimensions below 0.7 —
pipe these logs to your monitoring stack (Datadog, CloudWatch) and alert on thresholds.

A/B testing prompts with comparison view
----------------------------------------
1. Create two prompt versions (e.g. different system prompts in ``type_classifier.py`` on branches).
2. Run ``run_evaluation`` on the same labeled dataset for each version with
   ``log_to_langsmith=True`` and distinct experiment names.
3. In LangSmith → **Datasets** → your eval set → **Compare** runs side-by-side:
   accuracy, latency, and per-example diffs.
4. Pick the prompt with higher type/severity/component accuracy before promoting to production.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from bug_classifier import config
from bug_classifier.state import BugState, create_initial_state

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.7
AGENT_OUTPUT_FIELDS = {
    "type_classifier": "bug_type",
    "severity_assessor": "severity",
    "component_identifier": "component",
    "supervisor": "supervisor_verdict",
    "summary_agent": "summary",
}


def trace_agent(agent_name: str) -> Callable[[Callable[[BugState], dict]], Callable[[BugState], dict]]:
    """Augment an agent with structured console logs (complements LangSmith native tracing).

    LangChain/LangGraph already emit spans when ``LANGCHAIN_TRACING_V2`` is enabled.
    This decorator adds lightweight local telemetry (timing, state keys) without
    duplicating LangSmith traces.
    """

    def decorator(fn: Callable[[BugState], dict]) -> Callable[[BugState], dict]:
        @functools.wraps(fn)
        def wrapper(state: BugState) -> dict:
            started_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            input_keys = sorted(state.keys())
            logger.info(
                "[%s] start | timestamp=%s | input_keys=%s",
                agent_name,
                started_at,
                input_keys,
            )
            try:
                result = fn(state)
                elapsed = time.perf_counter() - start
                output_keys = sorted(result.keys()) if isinstance(result, dict) else []
                logger.info(
                    "[%s] complete | duration=%.3fs | output_keys=%s",
                    agent_name,
                    elapsed,
                    output_keys,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.exception(
                    "[%s] failed | duration=%.3fs | error=%s",
                    agent_name,
                    elapsed,
                    exc,
                )
                raise

        return wrapper

    return decorator


def _infer_pipeline_branch(state: BugState) -> str:
    if state.get("severity") == "P0" and not state.get("component"):
        return "emergency_handler (P0 fast-path)"
    if state.get("component") and state.get("summary"):
        return "normal (classify → assess → component → summarize → notion)"
    if state.get("severity"):
        return "partial (incomplete downstream agents)"
    return "unknown"


def _agents_run_count(state: BugState) -> int:
    count = 0
    if state.get("bug_type"):
        count += 1
    if state.get("severity"):
        count += 1
    if state.get("component"):
        count += 1
    if state.get("summary"):
        count += 1
    return count


def _average_confidence(state: BugState) -> float | None:
    scores = state.get("confidence_scores") or {}
    if not scores:
        return None
    return round(sum(scores.values()) / len(scores), 3)


def _low_confidence_dimensions(state: BugState) -> list[str]:
    scores = state.get("confidence_scores") or {}
    return [
        name
        for name, value in scores.items()
        if value < LOW_CONFIDENCE_THRESHOLD
    ]


def log_classification_metrics(state: BugState) -> None:
    """Log end-of-pipeline metrics for trend analysis and ops dashboards."""
    branch = _infer_pipeline_branch(state)
    agents_run = _agents_run_count(state)
    avg_confidence = _average_confidence(state)
    low_conf = _low_confidence_dimensions(state)
    severity = state.get("severity")
    bug_type = state.get("bug_type")

    retry_counts = state.get("retry_counts") or {}
    total_reclassifications = sum(retry_counts.values())

    logger.info(
        "Pipeline metrics | branch=%s | agents_run=%d | bug_type=%s | severity=%s | "
        "avg_confidence=%s | supervisor=%s | reclassifications=%d | "
        "needs_review=%s | ticket_url=%s",
        branch,
        agents_run,
        bug_type,
        severity,
        avg_confidence,
        state.get("supervisor_verdict"),
        total_reclassifications,
        bool(state.get("needs_review")),
        state.get("ticket_url"),
    )

    if total_reclassifications:
        logger.info(
            "Supervisor triggered %d reclassification(s): %s. If this rate is "
            "high across runs, revisit specialist prompts.",
            total_reclassifications,
            retry_counts,
        )

    if state.get("needs_review"):
        logger.warning(
            "Supervisor flagged ticket for manual review: %s",
            state.get("supervisor_notes"),
        )

    if low_conf:
        logger.warning(
            "Low-confidence dimensions (<%s): %s — consider manual review.",
            LOW_CONFIDENCE_THRESHOLD,
            ", ".join(low_conf),
        )

    if state.get("error"):
        logger.error("Pipeline error recorded in state: %s", state["error"])

    if severity == "P0":
        logger.warning(
            "P0 classification detected. If P0 volume is unusually high over time, "
            "investigate upstream reporting noise or severity calibration."
        )


def create_evaluation_dataset(examples: list[dict]) -> list[dict]:
    """Format labeled bug reports for LangSmith evaluation runs.

    Each example should include:
    ``report``, ``expected_type``, ``expected_severity``, ``expected_component``.

    Returns a list of dicts with ``inputs``, ``reference_outputs``, and ``metadata``
    suitable for ``run_evaluation`` and LangSmith dataset upload.
    """
    dataset: list[dict] = []
    for index, example in enumerate(examples):
        dataset.append(
            {
                "inputs": {"bug_report": example["report"]},
                "reference_outputs": {
                    "bug_type": example["expected_type"],
                    "severity": example["expected_severity"],
                    "component": example["expected_component"],
                },
                "metadata": {"example_id": index},
            }
        )
    return dataset


def _matches(actual: str | None, expected: str | None) -> bool:
    if actual is None or expected is None:
        return False
    return actual.strip().lower() == expected.strip().lower()


def run_evaluation(
    app: Any,
    dataset: list[dict],
    *,
    log_to_langsmith: bool = False,
    experiment_name: str = "bug-classifier-eval",
) -> dict[str, Any]:
    """Run the pipeline on a labeled dataset and compute accuracy metrics.

    Args:
        app: Compiled LangGraph application (``workflow.app``).
        dataset: Output of :func:`create_evaluation_dataset`.
        log_to_langsmith: When True and tracing is enabled, upload results to LangSmith.
        experiment_name: LangSmith experiment prefix for comparison runs.

    Returns:
        Summary dict with per-agent and overall accuracy plus per-example details.
    """
    total = len(dataset)
    type_correct = severity_correct = component_correct = pipeline_correct = 0
    details: list[dict[str, Any]] = []

    for example in dataset:
        bug_report = example["inputs"]["bug_report"]
        expected = example["reference_outputs"]
        state = app.invoke(create_initial_state(bug_report))

        actual_type = state.get("bug_type")
        actual_severity = state.get("severity")
        actual_component = state.get("component")

        type_ok = _matches(actual_type, expected["bug_type"])
        severity_ok = _matches(actual_severity, expected["severity"])
        component_ok = _matches(actual_component, expected["component"])
        all_ok = type_ok and severity_ok and component_ok

        type_correct += int(type_ok)
        severity_correct += int(severity_ok)
        component_correct += int(component_ok)
        pipeline_correct += int(all_ok)

        details.append(
            {
                "report": bug_report[:120],
                "expected": expected,
                "actual": {
                    "bug_type": actual_type,
                    "severity": actual_severity,
                    "component": actual_component,
                },
                "correct": {
                    "bug_type": type_ok,
                    "severity": severity_ok,
                    "component": component_ok,
                    "pipeline": all_ok,
                },
            }
        )

    summary = {
        "total": total,
        "type_accuracy": round(type_correct / total, 3) if total else 0.0,
        "severity_accuracy": round(severity_correct / total, 3) if total else 0.0,
        "component_accuracy": round(component_correct / total, 3) if total else 0.0,
        "pipeline_accuracy": round(pipeline_correct / total, 3) if total else 0.0,
        "details": details,
    }

    _print_evaluation_report(summary)

    if log_to_langsmith and config.LANGCHAIN_TRACING_ENABLED:
        _log_evaluation_to_langsmith(summary, experiment_name=experiment_name)
    elif log_to_langsmith:
        logger.warning(
            "log_to_langsmith=True but LangSmith tracing is disabled; skipping upload."
        )

    return summary


def _print_evaluation_report(summary: dict[str, Any]) -> None:
    total = summary["total"]
    print("\n=== Bug Classifier Evaluation Report ===")
    print(f"Examples: {total}")
    print(f"Type accuracy:       {summary['type_accuracy']:.1%}")
    print(f"Severity accuracy:   {summary['severity_accuracy']:.1%}")
    print(f"Component accuracy:  {summary['component_accuracy']:.1%}")
    print(f"Pipeline accuracy:   {summary['pipeline_accuracy']:.1%}")
    print("========================================\n")


def _log_evaluation_to_langsmith(
    summary: dict[str, Any], *, experiment_name: str
) -> None:
    """Upload evaluation summary as feedback to LangSmith (best-effort)."""
    try:
        from langsmith import Client

        client = Client()
        client.create_project(
            project_name=f"{experiment_name}-{config.LANGCHAIN_PROJECT}",
            metadata={
                "type_accuracy": summary["type_accuracy"],
                "severity_accuracy": summary["severity_accuracy"],
                "component_accuracy": summary["component_accuracy"],
                "pipeline_accuracy": summary["pipeline_accuracy"],
                "total_examples": summary["total"],
            },
        )
        logger.info(
            "Logged evaluation metrics to LangSmith project %s",
            f"{experiment_name}-{config.LANGCHAIN_PROJECT}",
        )
    except Exception:
        logger.exception("Failed to log evaluation results to LangSmith")
