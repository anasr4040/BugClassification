"""Unit tests for the supervisor/critic agent and the investigation tools."""

from __future__ import annotations

import json

from bug_classifier.agents.supervisor import (
    SupervisorCritique,
    RevisionDirective,
    find_issues,
    supervise,
)
from bug_classifier.agents.tools import (
    match_component_keywords,
    parse_stack_trace,
    scan_severity_signals,
    search_similar_bugs,
)
from bug_classifier.state import create_initial_state
from bug_classifier.tests.conftest import CRASH_REPORT, OUTAGE_REPORT, REACT_FRONTEND_REPORT


def _classified(scores: dict[str, float], **overrides):
    state = create_initial_state("Checkout crashes in PaymentService.java:88.")
    state.update(
        {
            "bug_type": "crash",
            "severity": "P1",
            "component": "backend",
            "confidence_scores": scores,
        }
    )
    state.update(overrides)
    return state


class TestFindIssues:
    def test_no_issues_when_complete_and_confident(self) -> None:
        state = _classified({"type": 0.9, "severity": 0.9, "component": 0.9})
        assert find_issues(state) == {}

    def test_missing_dimension_flagged(self) -> None:
        state = _classified({"type": 0.9, "severity": 0.9})
        state["component"] = None
        issues = find_issues(state)
        assert set(issues) == {"component"}

    def test_low_confidence_flagged(self) -> None:
        state = _classified({"type": 0.4, "severity": 0.9, "component": 0.9})
        issues = find_issues(state)
        assert set(issues) == {"type"}

    def test_security_p3_inconsistency_flagged(self) -> None:
        state = _classified(
            {"type": 0.9, "severity": 0.9, "component": 0.9},
            bug_type="security",
            severity="P3",
        )
        issues = find_issues(state)
        assert set(issues) == {"severity"}


class TestSupervise:
    def test_approves_clean_classification_without_llm(self) -> None:
        # No LLM patching: the fast path must not construct a model at all.
        state = _classified({"type": 0.9, "severity": 0.9, "component": 0.9})
        result = supervise(state)
        assert result["supervisor_verdict"] == "approve"
        assert result["needs_review"] is False

    def test_requests_revision_with_llm_feedback(self, patch_supervisor_llm) -> None:
        patch_supervisor_llm(
            SupervisorCritique(
                revisions=[
                    RevisionDirective(
                        dimension="type",
                        feedback="Re-read the stack trace; an NPE is a crash.",
                    )
                ],
                reasoning="Type looks wrong for a stack trace report.",
            )
        )
        state = _classified({"type": 0.4, "severity": 0.9, "component": 0.9})
        result = supervise(state)
        assert result["supervisor_verdict"] == "revise"
        assert result["revision_targets"] == ["type"]
        assert "stack trace" in result["revision_feedback"]["type"]
        assert result["revision_round"] == 1

    def test_llm_failure_falls_back_to_rule_feedback(self, fail_supervisor_llm) -> None:
        state = _classified({"type": 0.4, "severity": 0.9, "component": 0.9})
        result = supervise(state)
        assert result["supervisor_verdict"] == "revise"
        assert "below" in result["revision_feedback"]["type"]

    def test_budget_exhausted_approves_with_flag(self) -> None:
        state = _classified(
            {"type": 0.4, "severity": 0.9, "component": 0.9}, revision_round=1
        )
        result = supervise(state)
        assert result["supervisor_verdict"] == "approve"
        assert result["needs_review"] is True


class TestTools:
    def test_parse_stack_trace_extracts_evidence(self) -> None:
        report = (
            "NullPointerException in OrderService.java:247 after deploy; "
            "clients see 500 errors. Worker died with SIGSEGV."
        )
        payload = json.loads(parse_stack_trace.invoke({"report": report}))
        assert "NullPointerException" in payload["exceptions"]
        assert "OrderService.java:247" in payload["file_references"]
        assert "500" in payload["http_status_codes"]
        assert payload["has_technical_evidence"] is True

    def test_parse_stack_trace_no_evidence(self) -> None:
        payload = json.loads(
            parse_stack_trace.invoke({"report": "Things feel vaguely wrong."})
        )
        assert payload["has_technical_evidence"] is False

    def test_scan_severity_signals_outage(self) -> None:
        payload = json.loads(scan_severity_signals.invoke({"report": OUTAGE_REPORT}))
        assert payload["broad_impact_signals"]
        assert "critical/high" in payload["heuristic"]

    def test_scan_severity_signals_cosmetic(self) -> None:
        payload = json.loads(
            scan_severity_signals.invoke(
                {"report": "Minor cosmetic misalignment; workaround exists."}
            )
        )
        assert payload["mitigation_signals"]
        assert "lower severity" in payload["heuristic"]

    def test_match_component_keywords_frontend(self) -> None:
        payload = json.loads(
            match_component_keywords.invoke({"report": REACT_FRONTEND_REPORT})
        )
        assert payload["best_match"] == "frontend"

    def test_search_similar_bugs_dry_run_safe(self) -> None:
        payload = json.loads(search_similar_bugs.invoke({"query": CRASH_REPORT[:40]}))
        assert payload["results"] == []
        assert "unavailable" in payload["note"]
