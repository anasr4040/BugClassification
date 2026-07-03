"""Integration tests for the multi-agent LangGraph classification system."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bug_classifier.agents.component_identifier import ComponentIdentification
from bug_classifier.agents.severity_assessor import SeverityAssessment
from bug_classifier.agents.summary_agent import BugTicketSummary
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
        assert final.get("supervisor_verdict") == "approve"

    def test_parallel_specialists_merge_confidence_scores(self, fresh_app) -> None:
        """All three specialists run concurrently; reducers merge their writes."""
        final = fresh_app.invoke(create_initial_state(SAMPLE_REPORT))

        scores = final["confidence_scores"]
        assert set(scores) == {"type", "severity", "component"}
        note_agents = {note["agent"] for note in final["agent_notes"]}
        assert {
            "type_classifier",
            "severity_assessor",
            "component_identifier",
            "supervisor",
        } <= note_agents

    def test_emergency_path_for_p0(
        self, monkeypatch: pytest.MonkeyPatch, patch_type_llm, patch_component_llm
    ) -> None:
        patch_type_llm(
            TypeClassification(type="crash", confidence=0.95, reasoning="Outage.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.9, reasoning="Runs in parallel."
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
        # Specialists run in parallel, so component IS populated on P0 now…
        assert final.get("component") == "backend"
        # …but the standard summary pipeline is still bypassed.
        assert final.get("summary") is None
        assert final.get("ticket_url") is not None
        assert final.get("supervisor_verdict") == "approve"

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
            "supervisor_verdict",
        ):
            assert final.get(field) is not None, f"Missing field: {field}"

    def test_agent_error_does_not_crash_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patch_severity_llm,
        patch_component_llm,
        patch_summary_llm,
        fail_supervisor_llm,
    ) -> None:
        patch_summary_llm(
            BugTicketSummary(summary="**Classification Result**: fallback summary.")
        )
        failing_llm = MagicMock()
        failing_llm.with_structured_output.side_effect = RuntimeError("LLM unavailable")
        failing_llm.bind_tools.side_effect = RuntimeError("LLM unavailable")
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
        # Supervisor retried the failing specialist, then flagged the result.
        assert final.get("revision_round") == 1
        assert final.get("needs_review") is True
        assert final.get("summary") is not None


class TestSupervisorLoop:
    def test_low_confidence_triggers_revision_and_improves(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patch_severity_llm,
        patch_component_llm,
        fail_supervisor_llm,
    ) -> None:
        """First pass: low-confidence type → supervisor sends it back with
        feedback; second pass returns a confident answer that is approved."""
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.type_classifier._build_llm",
            [
                TypeClassification(
                    type="logic", confidence=0.4, reasoning="Unsure on first pass."
                ),
                TypeClassification(
                    type="crash", confidence=0.93, reasoning="Stack trace confirms crash."
                ),
            ],
        )
        patch_severity_llm(
            SeverityAssessment(severity="P1", confidence=0.9, reasoning="High impact.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.9, reasoning="Java service."
            )
        )

        from bug_classifier.graph.workflow import build_graph

        final = build_graph().invoke(create_initial_state(JAVA_BACKEND_REPORT))

        assert final["revision_round"] == 1
        assert final["bug_type"] == "crash"
        assert final["confidence_scores"]["type"] == 0.93
        assert final["supervisor_verdict"] == "approve"
        assert final["needs_review"] is False
        # Only the flagged dimension was revised.
        assert "type" in (final.get("revision_feedback") or {})
        assert "severity" not in (final.get("revision_feedback") or {})

    def test_revision_loop_is_bounded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patch_severity_llm,
        patch_component_llm,
        patch_summary_llm,
        fail_supervisor_llm,
    ) -> None:
        """A specialist that stays low-confidence cannot loop forever."""
        patch_summary_llm(
            BugTicketSummary(summary="**Classification Result**: low-confidence summary.")
        )
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.type_classifier._build_llm",
            TypeClassification(
                type="logic", confidence=0.3, reasoning="Perpetually unsure."
            ),
        )
        patch_severity_llm(
            SeverityAssessment(severity="P2", confidence=0.9, reasoning="Medium.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="backend", confidence=0.9, reasoning="Backend."
            )
        )

        from bug_classifier.graph.workflow import build_graph

        final = build_graph().invoke(create_initial_state(SAMPLE_REPORT))

        assert final["revision_round"] == 1  # config.MAX_REVISION_ROUNDS
        assert final["supervisor_verdict"] == "approve"
        assert final["needs_review"] is True
        assert final.get("summary") is not None
        assert final.get("ticket_url") is not None

    def test_security_p3_inconsistency_triggers_revision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patch_type_llm,
        patch_component_llm,
        fail_supervisor_llm,
    ) -> None:
        patch_type_llm(
            TypeClassification(type="security", confidence=0.95, reasoning="Injection.")
        )
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.severity_assessor._build_llm",
            [
                SeverityAssessment(
                    severity="P3", confidence=0.8, reasoning="Underestimated."
                ),
                SeverityAssessment(
                    severity="P1", confidence=0.92, reasoning="Security exposure."
                ),
            ],
        )
        patch_component_llm(
            ComponentIdentification(
                component="api", confidence=0.9, reasoning="Endpoint."
            )
        )

        from bug_classifier.graph.workflow import build_graph

        final = build_graph().invoke(
            create_initial_state("Search endpoint leaks other users' emails.")
        )

        assert final["severity"] == "P1"
        assert final["revision_round"] == 1
        assert final["needs_review"] is False


class TestHumanInTheLoop:
    def test_p0_interrupt_pause_and_downgrade(
        self, monkeypatch: pytest.MonkeyPatch, patch_type_llm, patch_component_llm
    ) -> None:
        """With HITL enabled, a P0 pauses for approval; an operator downgrade
        resumes the run and logs a non-urgent ticket."""
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        monkeypatch.setattr("bug_classifier.config.HITL_EMERGENCY", True)
        patch_type_llm(
            TypeClassification(type="crash", confidence=0.95, reasoning="Outage.")
        )
        patch_component_llm(
            ComponentIdentification(
                component="infrastructure", confidence=0.9, reasoning="Redis."
            )
        )
        patch_llm(
            monkeypatch,
            "bug_classifier.agents.severity_assessor._build_llm",
            SeverityAssessment(severity="P0", confidence=0.99, reasoning="Outage."),
        )

        from bug_classifier.graph.workflow import build_graph

        app = build_graph(checkpointer=InMemorySaver())
        thread = {"configurable": {"thread_id": "hitl-test"}}

        paused = app.invoke(create_initial_state(OUTAGE_REPORT), thread)
        assert "__interrupt__" in paused  # waiting for the operator

        final = app.invoke(
            Command(resume={"action": "downgrade", "severity": "P1"}), thread
        )
        assert final["severity"] == "P1"
        assert final.get("ticket_url") is not None


class TestRouting:
    def test_route_revise_returns_flagged_specialists(self) -> None:
        from bug_classifier.graph.workflow import route_after_supervisor

        state = create_initial_state("x")
        state.update(
            {
                "supervisor_verdict": "revise",
                "revision_targets": ["type", "component"],
            }
        )
        assert route_after_supervisor(state) == ["classify_type", "identify_component"]

    def test_route_p0_goes_to_emergency(self) -> None:
        from bug_classifier.graph.workflow import route_after_supervisor

        state = create_initial_state("x")
        state.update({"supervisor_verdict": "approve", "severity": "P0"})
        assert route_after_supervisor(state) == "emergency_handler"

    def test_route_approved_goes_to_summarize(self) -> None:
        from bug_classifier.graph.workflow import route_after_supervisor

        state = create_initial_state("x")
        state.update({"supervisor_verdict": "approve", "severity": "P2"})
        assert route_after_supervisor(state) == "summarize"
