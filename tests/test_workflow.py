"""Integration tests for the LangGraph classification pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bug_classifier.agents.component_identifier import ComponentIdentification
from bug_classifier.agents.severity_assessor import SeverityAssessment
from bug_classifier.agents.type_classifier import TypeClassification
from bug_classifier.state import create_initial_state
from bug_classifier.tests.conftest import (
    JAVA_BACKEND_REPORT,
    OUTAGE_REPORT,
    patch_llm,
)


SAMPLE_REPORT = (
    "Users cannot complete checkout. PaymentService.java throws NPE at line 88. "
    "Affects 30% of transactions."
)


class TestWorkflow:
    def test_full_pipeline_end_to_end(self, fresh_app) -> None:
        app = fresh_app
        final = app.invoke(create_initial_state(SAMPLE_REPORT))

        assert final.get("bug_type") is not None
        assert final.get("severity") is not None
        assert final.get("component") is not None
        assert final.get("summary") is not None
        assert final.get("ticket_url") is not None
        assert final.get("confidence_scores")

    def test_emergency_path_for_p0(
        self, monkeypatch: pytest.MonkeyPatch, patch_type_llm, patch_component_llm
    ) -> None:
        patch_type_llm(
            TypeClassification(type="crash", confidence=0.95, reasoning="Outage.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.9, reasoning="Should not run."
            )
        )
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.severity_assessor._build_llm",
            SeverityAssessment(severity="P0", confidence=0.99, reasoning="Critical."),
        )

        from bug_classifier.graph.workflow import build_graph

        app = build_graph()
        final = app.invoke(create_initial_state(OUTAGE_REPORT))

        assert final["severity"] == "P0"
        assert final.get("component") is None
        assert final.get("summary") is None
        assert final.get("ticket_url") is not None

    def test_all_state_fields_after_full_run(
        self, fresh_app, patch_type_llm, patch_severity_llm, patch_component_llm
    ) -> None:
        patch_type_llm(
            TypeClassification(type="logic", confidence=0.9, reasoning="Logic bug.")
        )
        patch_severity_llm(
            SeverityAssessment(severity="P2", confidence=0.88, reasoning="Medium.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.87, reasoning="Java service."
            )
        )

        final = fresh_app.invoke(create_initial_state(JAVA_BACKEND_REPORT))

        for field in (
            "raw_report",
            "bug_type",
            "severity",
            "component",
            "summary",
            "ticket_url",
            "confidence_scores",
        ):
            assert final.get(field) is not None, f"Missing field: {field}"

    def test_agent_error_does_not_crash_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, patch_severity_llm, patch_component_llm
    ) -> None:
        failing_llm = MagicMock()
        failing_llm.with_structured_output.side_effect = RuntimeError("LLM unavailable")
        monkeypatch.setattr(
            "bug_classifier.agents.type_classifier._build_llm", lambda: failing_llm
        )
        patch_severity_llm(
            SeverityAssessment(severity="P2", confidence=0.8, reasoning="Recovered.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.85, reasoning="Backend."
            )
        )

        from bug_classifier.graph.workflow import build_graph

        final = build_graph().invoke(create_initial_state(SAMPLE_REPORT))

        assert final.get("bug_type") == "logic"  # default fallback
        assert final.get("severity") == "P2"
        assert final.get("component") == "backend"
        assert final.get("error") is None
