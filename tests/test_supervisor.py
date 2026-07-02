"""Unit and integration tests for the supervisor agent and its feedback loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from bug_classifier.agents.component_identifier import ComponentIdentification
from bug_classifier.agents.severity_assessor import SeverityAssessment
from bug_classifier.agents.supervisor import (
    SupervisorReview,
    VERDICT_APPROVED,
    VERDICT_RECLASSIFY,
    hint_for_dimension,
    supervise,
)
from bug_classifier.agents.type_classifier import TypeClassification, classify_type
from bug_classifier.graph.workflow import _route_after_supervisor
from bug_classifier.state import BugState, create_initial_state
from bug_classifier.tests.conftest import JAVA_BACKEND_REPORT, patch_llm


def _reviewed_state(
    *,
    bug_type: str = "logic",
    severity: str = "P2",
    component: str = "backend",
    scores: dict[str, float] | None = None,
    retry_counts: dict[str, int] | None = None,
) -> BugState:
    state = create_initial_state(JAVA_BACKEND_REPORT)
    state.update(
        {
            "bug_type": bug_type,
            "severity": severity,
            "component": component,
            "confidence_scores": scores
            or {"type": 0.9, "severity": 0.9, "component": 0.9},
            "retry_counts": retry_counts,
        }
    )
    return state


class SequencedLLM:
    """Mock LLM returning successive responses across separate invocations."""

    def __init__(self, responses: list[Any]) -> None:
        self._iterator = iter(responses)

    def with_structured_output(self, schema: Any) -> Any:
        mock = MagicMock()
        mock.invoke = MagicMock(side_effect=lambda messages: next(self._iterator))
        return mock


class CapturingLLM:
    """Mock LLM that records the messages passed to ``invoke``."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[Any] = []

    def with_structured_output(self, schema: Any) -> Any:
        outer = self

        class _Structured:
            def invoke(self, messages: Any) -> Any:
                outer.calls.append(messages)
                return outer.response

        return _Structured()


class TestSupervise:
    def test_approves_clean_classification_without_llm_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """High confidence + consistent dimensions → fast-path approval."""
        boom = MagicMock(side_effect=AssertionError("LLM must not be called"))
        monkeypatch.setattr("bug_classifier.agents.supervisor._build_llm", boom)

        result = supervise(_reviewed_state())

        assert result["supervisor_verdict"] == VERDICT_APPROVED
        assert result["needs_review"] is False
        assert result["reclassify_target"] is None
        boom.assert_not_called()

    def test_low_confidence_triggers_reclassification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.supervisor._build_llm",
            SupervisorReview(
                approved=False,
                problem_dimension="type",
                hint="Report mentions SQL injection — reconsider 'logic'.",
                reasoning="Security evidence contradicts the type.",
            ),
        )

        state = _reviewed_state(scores={"type": 0.4, "severity": 0.9, "component": 0.9})
        result = supervise(state)

        assert result["supervisor_verdict"] == VERDICT_RECLASSIFY
        assert result["reclassify_target"] == "type"
        assert "SQL injection" in result["reclassify_hint"]
        assert result["retry_counts"] == {"type": 1}

    def test_suspicious_combination_triggers_review(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """security/P3 is flagged even when every confidence is high."""
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.supervisor._build_llm",
            SupervisorReview(
                approved=False,
                problem_dimension="severity",
                hint="An exposed credential should not be P3.",
                reasoning="Severity understates the risk.",
            ),
        )

        state = _reviewed_state(bug_type="security", severity="P3")
        result = supervise(state)

        assert result["supervisor_verdict"] == VERDICT_RECLASSIFY
        assert result["reclassify_target"] == "severity"

    def test_retry_budget_exhausted_approves_with_needs_review(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.supervisor._build_llm",
            SupervisorReview(
                approved=False,
                problem_dimension="type",
                hint="Still looks wrong.",
                reasoning="Unresolved type concern.",
            ),
        )

        state = _reviewed_state(
            scores={"type": 0.4, "severity": 0.9, "component": 0.9},
            retry_counts={"type": 1},
        )
        result = supervise(state)

        assert result["supervisor_verdict"] == VERDICT_APPROVED
        assert result["needs_review"] is True
        assert result["retry_counts"] == {"type": 1}
        assert "budget exhausted" in result["supervisor_notes"]

    def test_llm_failure_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failing = MagicMock()
        failing.with_structured_output.side_effect = RuntimeError("LLM unavailable")
        monkeypatch.setattr(
            "bug_classifier.agents.supervisor._build_llm", lambda: failing
        )

        state = _reviewed_state(scores={"type": 0.4, "severity": 0.9, "component": 0.9})
        result = supervise(state)

        assert result["supervisor_verdict"] == VERDICT_APPROVED
        assert result["needs_review"] is True

    def test_approval_despite_flags_keeps_human_in_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.supervisor._build_llm",
            SupervisorReview(
                approved=True,
                problem_dimension="none",
                reasoning="Edge-case crash justifies P3.",
            ),
        )

        state = _reviewed_state(bug_type="crash", severity="P3")
        result = supervise(state)

        assert result["supervisor_verdict"] == VERDICT_APPROVED
        assert result["needs_review"] is True


class TestHintPropagation:
    def test_hint_only_for_targeted_dimension(self) -> None:
        state = _reviewed_state()
        state.update(
            {"reclassify_target": "type", "reclassify_hint": "Look at the stack trace."}
        )

        assert "Look at the stack trace." in hint_for_dimension(state, "type")
        assert hint_for_dimension(state, "severity") == ""
        assert hint_for_dimension(state, "component") == ""

    def test_type_classifier_receives_hint_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingLLM(
            TypeClassification(
                type="security", confidence=0.9, reasoning="Reconsidered."
            )
        )
        monkeypatch.setattr(
            "bug_classifier.agents.type_classifier._build_llm", lambda: capturing
        )

        state = _reviewed_state()
        state.update(
            {
                "reclassify_target": "type",
                "reclassify_hint": "Report mentions SQL injection.",
            }
        )
        result = classify_type(state)

        assert result["bug_type"] == "security"
        human_message = capturing.calls[0][1].content
        assert "SQL injection" in human_message


class TestSupervisorRouting:
    def test_routes_approved_to_summarize(self) -> None:
        assert _route_after_supervisor({"supervisor_verdict": "approved"}) == "summarize"

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("type", "classify_type"),
            ("severity", "assess_severity"),
            ("component", "identify_component"),
        ],
    )
    def test_routes_reclassification_to_specialist(
        self, target: str, expected: str
    ) -> None:
        state = {"supervisor_verdict": "reclassify", "reclassify_target": target}
        assert _route_after_supervisor(state) == expected

    def test_unknown_target_falls_through_to_summarize(self) -> None:
        state = {"supervisor_verdict": "reclassify", "reclassify_target": "bogus"}
        assert _route_after_supervisor(state) == "summarize"


class TestSupervisorLoopIntegration:
    def test_reclassification_loop_end_to_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patch_type_llm,
        patch_severity_llm,
    ) -> None:
        """Component starts low-confidence; the supervisor sends it back once
        with a hint, the second attempt succeeds, and the pipeline completes."""
        patch_type_llm(
            TypeClassification(type="logic", confidence=0.9, reasoning="Logic bug.")
        )
        patch_severity_llm(
            SeverityAssessment(severity="P2", confidence=0.9, reasoning="Medium.")
        )
        monkeypatch.setattr(
            "bug_classifier.agents.component_identifier._build_llm",
            lambda seq=SequencedLLM(
                [
                    ComponentIdentification(
                        component="backend", confidence=0.5, reasoning="Unsure."
                    ),
                    ComponentIdentification(
                        component="database", confidence=0.9, reasoning="Migration."
                    ),
                ]
            ): seq,
        )
        monkeypatch.setattr(
            "bug_classifier.agents.supervisor._build_llm",
            lambda seq=SequencedLLM(
                [
                    SupervisorReview(
                        approved=False,
                        problem_dimension="component",
                        hint="Report describes a schema migration failure.",
                        reasoning="Technical evidence points at the database.",
                    ),
                ]
            ): seq,
        )

        from bug_classifier.graph.workflow import build_graph

        report = (
            "Nightly schema migration intermittently fails, leaving the orders "
            "table locked. Retrying manually usually works."
        )
        final = build_graph().invoke(create_initial_state(report))

        assert final["component"] == "database"
        assert final["supervisor_verdict"] == VERDICT_APPROVED
        assert final["retry_counts"] == {"component": 1}
        assert final.get("needs_review") is False
        assert final.get("summary") is not None
        assert final.get("ticket_url") is not None
