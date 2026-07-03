"""Accuracy evaluation suite with labeled bug reports (mocked LLM for CI)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from bug_classifier.agents.component_identifier import ComponentIdentification
from bug_classifier.agents.severity_assessor import SeverityAssessment
from bug_classifier.agents.type_classifier import TypeClassification
from bug_classifier.observability import create_evaluation_dataset, run_evaluation
from bug_classifier.tests.conftest import MockChatXAI

HIGH_CONFIDENCE_THRESHOLD = 0.7

# 20 labeled examples for regression testing after prompt changes.
# ``mock_*`` fields simulate LLM outputs; a few intentional mismatches test flagging.
LABELED_DATASET: list[dict[str, Any]] = [
    {
        "report": "App segfaults on launch for all users. Core dump attached.",
        "expected_type": "crash",
        "expected_severity": "P0",
        "expected_component": "backend",
        "mock_type": "crash",
        "mock_severity": "P0",
        "mock_component": "backend",
    },
    {
        "report": "Production login down. OAuth NPE in AuthService.java:247.",
        "expected_type": "crash",
        "expected_severity": "P0",
        "expected_component": "authentication",
        "mock_type": "crash",
        "mock_severity": "P0",
        "mock_component": "authentication",
    },
    {
        "report": "Dashboard query on analytics_events takes 20s; missing index.",
        "expected_type": "performance",
        "expected_severity": "P2",
        "expected_component": "database",
        "mock_type": "performance",
        "mock_severity": "P2",
        "mock_component": "database",
    },
    {
        "report": "Footer overlaps submit button on iPhone SE. Cosmetic only.",
        "expected_type": "ui",
        "expected_severity": "P3",
        "expected_component": "frontend",
        "mock_type": "ui",
        "mock_severity": "P3",
        "mock_component": "frontend",
    },
    {
        "report": "GET /api/v2/users exposes hashed_password to any user.",
        "expected_type": "security",
        "expected_severity": "P1",
        "expected_component": "api",
        "mock_type": "security",
        "mock_severity": "P1",
        "mock_component": "api",
    },
    {
        "report": "Coupon applied twice after cart remove/re-add. 40% not 20%.",
        "expected_type": "logic",
        "expected_severity": "P2",
        "expected_component": "backend",
        "mock_type": "logic",
        "mock_severity": "P2",
        "mock_component": "backend",
    },
    {
        "report": "Kubernetes pods OOMKilled after deploy. All API pods restarting.",
        "expected_type": "crash",
        "expected_severity": "P0",
        "expected_component": "infrastructure",
        "mock_type": "crash",
        "mock_severity": "P0",
        "mock_component": "infrastructure",
    },
    {
        "report": "React ContactForm CSS flexbox overlap in mobile Chrome.",
        "expected_type": "ui",
        "expected_severity": "P3",
        "expected_component": "frontend",
        "mock_type": "ui",
        "mock_severity": "P3",
        "mock_component": "frontend",
    },
    {
        "report": "PostgreSQL replication lag causes stale reads on replicas.",
        "expected_type": "performance",
        "expected_severity": "P2",
        "expected_component": "database",
        "mock_type": "performance",
        "mock_severity": "P2",
        "mock_component": "database",
    },
    {
        "report": "SSO token refresh returns 401 for all Google OAuth users.",
        "expected_type": "security",
        "expected_severity": "P1",
        "expected_component": "authentication",
        "mock_type": "security",
        "mock_severity": "P1",
        "mock_component": "authentication",
    },
    {
        "report": "CI pipeline fails on docker push step. Cannot deploy.",
        "expected_type": "logic",
        "expected_severity": "P1",
        "expected_component": "infrastructure",
        "mock_type": "logic",
        "mock_severity": "P1",
        "mock_component": "infrastructure",
    },
    {
        "report": "GraphQL resolver N+1 query causes 8s latency on orders list.",
        "expected_type": "performance",
        "expected_severity": "P2",
        "expected_component": "api",
        "mock_type": "performance",
        "mock_severity": "P2",
        "mock_component": "api",
    },
    {
        "report": "Typo in help tooltip text on billing page.",
        "expected_type": "ui",
        "expected_severity": "P3",
        "expected_component": "frontend",
        "mock_type": "ui",
        "mock_severity": "P3",
        "mock_component": "frontend",
    },
    {
        "report": "SQL migration 20240315_add_index locks table for 10 minutes.",
        "expected_type": "performance",
        "expected_severity": "P2",
        "expected_component": "database",
        "mock_type": "performance",
        "mock_severity": "P2",
        "mock_component": "database",
    },
    {
        "report": "RBAC allows interns to delete production datasets.",
        "expected_type": "security",
        "expected_severity": "P1",
        "expected_component": "authentication",
        "mock_type": "security",
        "mock_severity": "P1",
        "mock_component": "authentication",
    },
    {
        "report": "Tax calculation off by 1 cent on discounted orders.",
        "expected_type": "logic",
        "expected_severity": "P2",
        "expected_component": "backend",
        "mock_type": "logic",
        "mock_severity": "P2",
        "mock_component": "backend",
    },
    {
        "report": "Load balancer health check misconfigured; 50% traffic dropped.",
        "expected_type": "crash",
        "expected_severity": "P0",
        "expected_component": "infrastructure",
        "mock_type": "crash",
        "mock_severity": "P0",
        "mock_component": "infrastructure",
    },
    {
        "report": "REST endpoint /health returns 500 under moderate load.",
        "expected_type": "crash",
        "expected_severity": "P1",
        "expected_component": "api",
        "mock_type": "crash",
        "mock_severity": "P1",
        "mock_component": "api",
    },
    {
        "report": "Something feels slow sometimes but hard to reproduce.",
        "expected_type": "performance",
        "expected_severity": "P3",
        "expected_component": "frontend",
        "mock_type": "logic",  # intentional mismatch
        "mock_severity": "P3",
        "mock_component": "frontend",
    },
    {
        "report": "Dark mode toggle shows wrong icon color on settings tab.",
        "expected_type": "ui",
        "expected_severity": "P3",
        "expected_component": "frontend",
        "mock_type": "ui",
        "mock_severity": "P2",  # intentional mismatch
        "mock_component": "frontend",
    },
]


def _confusion_matrix(
    expected: list[str], actual: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {label: {l: 0 for l in labels} for label in labels}
    for exp, act in zip(expected, actual, strict=True):
        if exp in matrix and act in matrix[exp]:
            matrix[exp][act] += 1
    return matrix


def _print_confusion_matrix(name: str, matrix: dict[str, dict[str, int]], labels: list[str]) -> None:
    header = "expected \\ actual".ljust(18) + "".join(label.rjust(14) for label in labels)
    print(f"\n{name} confusion matrix:")
    print(header)
    for exp in labels:
        row = exp.ljust(18) + "".join(str(matrix[exp][act]).rjust(14) for act in labels)
        print(row)


def _flag_high_confidence_wrong(
    details: list[dict[str, Any]], dataset: list[dict[str, Any]]
) -> list[str]:
    flagged: list[str] = []
    mock_conf = 0.92
    for row, detail in zip(dataset, details, strict=True):
        if mock_conf < HIGH_CONFIDENCE_THRESHOLD:
            continue
        for dim, key in (
            ("type", "bug_type"),
            ("severity", "severity"),
            ("component", "component"),
        ):
            if key == "component" and detail["actual"].get("component") is None:
                continue
            if row[f"expected_{dim}"] != detail["actual"].get(key):
                flagged.append(
                    f"{dim}: expected={row[f'expected_{dim}']}, "
                    f"actual={detail['actual'].get(key)} (conf={mock_conf})"
                )
    return flagged


@pytest.fixture
def eval_app(monkeypatch: pytest.MonkeyPatch):
    """Route mocked LLM outputs from LABELED_DATASET mock_* fields."""
    lookup = {row["report"]: row for row in LABELED_DATASET}

    monkeypatch.setattr(
        "bug_classifier.integrations.notion_logger._print_dry_run", lambda *args, **kwargs: None
    )

    def _quiet_emergency(state: Any) -> dict:
        from bug_classifier.integrations.notion_logger import log_to_notion

        return log_to_notion(state, urgent=True)

    monkeypatch.setattr(
        "bug_classifier.graph.workflow.emergency_handler", _quiet_emergency
    )

    def _extract_report(content: str) -> str:
        if "Bug report:\n" in content:
            return content.split("Bug report:\n", 1)[1]
        return content

    def _type_llm() -> MockChatXAI:
        def invoke(messages: Any) -> TypeClassification:
            report = messages[1].content
            row = lookup[report]
            return TypeClassification(
                type=row["mock_type"], confidence=0.92, reasoning="eval"
            )

        llm = MockChatXAI(None)
        llm.with_structured_output = lambda schema: type(  # type: ignore[method-assign]
            "Runner", (), {"invoke": staticmethod(invoke)}
        )()
        return llm

    def _severity_llm() -> MockChatXAI:
        def invoke(messages: Any) -> SeverityAssessment:
            report = _extract_report(messages[1].content)
            row = lookup[report]
            return SeverityAssessment(
                severity=row["mock_severity"], confidence=0.92, reasoning="eval"
            )

        llm = MockChatXAI(None)
        llm.with_structured_output = lambda schema: type(  # type: ignore[method-assign]
            "Runner", (), {"invoke": staticmethod(invoke)}
        )()
        return llm

    def _component_llm() -> MockChatXAI:
        def invoke(messages: Any) -> ComponentIdentification:
            report = _extract_report(messages[1].content)
            row = lookup[report]
            return ComponentIdentification(
                component=row["mock_component"], confidence=0.92, reasoning="eval"
            )

        llm = MockChatXAI(None)
        llm.with_structured_output = lambda schema: type(  # type: ignore[method-assign]
            "Runner", (), {"invoke": staticmethod(invoke)}
        )()
        return llm

    monkeypatch.setattr("bug_classifier.agents.type_classifier._build_llm", _type_llm)
    monkeypatch.setattr(
        "bug_classifier.agents.severity_assessor._build_llm", _severity_llm
    )
    monkeypatch.setattr(
        "bug_classifier.agents.component_identifier._build_llm", _component_llm
    )

    from bug_classifier.graph.workflow import build_graph

    return build_graph()


class TestEvaluation:
    def test_labeled_dataset_size(self) -> None:
        assert len(LABELED_DATASET) == 20

    def test_create_evaluation_dataset_format(self) -> None:
        examples = [
            {
                "report": row["report"],
                "expected_type": row["expected_type"],
                "expected_severity": row["expected_severity"],
                "expected_component": row["expected_component"],
            }
            for row in LABELED_DATASET
        ]
        dataset = create_evaluation_dataset(examples)
        assert len(dataset) == 20
        assert "inputs" in dataset[0]
        assert "reference_outputs" in dataset[0]

    def test_evaluation_accuracy_and_confusion_matrices(self, eval_app, capsys) -> None:
        examples = [
            {
                "report": row["report"],
                "expected_type": row["expected_type"],
                "expected_severity": row["expected_severity"],
                "expected_component": row["expected_component"],
            }
            for row in LABELED_DATASET
        ]
        dataset = create_evaluation_dataset(examples)
        summary = run_evaluation(eval_app, dataset, log_to_langsmith=False)

        assert summary["total"] == 20
        assert 0.0 <= summary["type_accuracy"] <= 1.0
        assert 0.0 <= summary["severity_accuracy"] <= 1.0
        assert 0.0 <= summary["component_accuracy"] <= 1.0

        expected_types = [row["expected_type"] for row in LABELED_DATASET]
        expected_severities = [row["expected_severity"] for row in LABELED_DATASET]
        expected_components = [row["expected_component"] for row in LABELED_DATASET]

        actual_types = [d["actual"]["bug_type"] for d in summary["details"]]
        actual_severities = [d["actual"]["severity"] for d in summary["details"]]
        actual_components = [d["actual"]["component"] for d in summary["details"]]

        type_labels = sorted(set(expected_types + actual_types))
        sev_labels = sorted(set(expected_severities + actual_severities))
        comp_labels = sorted(
            {c for c in expected_components + actual_components if c is not None}
        )

        _print_confusion_matrix(
            "Type",
            _confusion_matrix(expected_types, actual_types, type_labels),
            type_labels,
        )
        _print_confusion_matrix(
            "Severity",
            _confusion_matrix(expected_severities, actual_severities, sev_labels),
            sev_labels,
        )
        _print_confusion_matrix(
            "Component",
            _confusion_matrix(
                [e for e, a in zip(expected_components, actual_components, strict=True) if a],
                [a for a in actual_components if a],
                comp_labels,
            ),
            comp_labels,
        )

        # P0 emergency path skips component identification — exclude from component accuracy.
        component_pairs = [
            (row["expected_component"], detail["actual"]["component"])
            for row, detail in zip(LABELED_DATASET, summary["details"], strict=True)
            if detail["actual"].get("component") is not None
        ]
        if component_pairs:
            comp_acc = sum(
                1 for exp, act in component_pairs if exp == act
            ) / len(component_pairs)
            print(f"\nComponent accuracy (non-P0 path only): {comp_acc:.1%}")

        captured = capsys.readouterr()
        assert "Evaluation Report" in captured.out

        high_conf_wrong = _flag_high_confidence_wrong(summary["details"], LABELED_DATASET)

        if high_conf_wrong:
            print("\nHigh-confidence WRONG classifications (prompt review needed):")
            for line in high_conf_wrong:
                print(f"  - {line}")

        assert summary["type_accuracy"] >= 0.85
        assert len(high_conf_wrong) >= 2  # intentional mismatches in dataset
