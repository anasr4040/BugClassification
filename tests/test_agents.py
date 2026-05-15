"""Unit tests for individual classification agents (mocked LLM)."""

from __future__ import annotations

from bug_classifier.agents.component_identifier import (
    ComponentIdentification,
    identify_component,
)
from bug_classifier.agents.severity_assessor import SeverityAssessment, assess_severity
from bug_classifier.agents.summary_agent import BugTicketSummary, summarize
from bug_classifier.agents.type_classifier import TypeClassification, classify_type
from bug_classifier.state import create_initial_state
from bug_classifier.tests.conftest import (
    AMBIGUOUS_REPORT,
    COSMETIC_REPORT,
    CRASH_REPORT,
    JAVA_BACKEND_REPORT,
    OUTAGE_REPORT,
    REACT_FRONTEND_REPORT,
)

VALID_TYPES = {"crash", "performance", "logic", "security", "ui"}
AGENT_OWNED_KEYS = {
    "type_classifier": {"bug_type", "confidence_scores"},
    "severity_assessor": {"severity", "confidence_scores"},
    "component_identifier": {"component", "confidence_scores"},
    "summary_agent": {"summary"},
}


class TestClassifyType:
    def test_clear_crash_report(self, patch_type_llm) -> None:
        patch_type_llm(
            TypeClassification(
                type="crash", confidence=0.97, reasoning="Unhandled crash with stack trace."
            )
        )
        result = classify_type(create_initial_state(CRASH_REPORT))
        assert result["bug_type"] == "crash"
        assert result["bug_type"] in VALID_TYPES
        assert result["confidence_scores"]["type"] == 0.97

    def test_ambiguous_report(self, patch_type_llm) -> None:
        patch_type_llm(
            TypeClassification(
                type="logic", confidence=0.55, reasoning="Vague symptoms suggest logic issue."
            )
        )
        result = classify_type(create_initial_state(AMBIGUOUS_REPORT))
        assert result["bug_type"] in VALID_TYPES
        assert result["confidence_scores"]["type"] == 0.55

    def test_guardrails(self, patch_type_llm) -> None:
        patch_type_llm(
            TypeClassification(type="ui", confidence=0.8, reasoning="UI glitch.")
        )
        result = classify_type(create_initial_state("Button color wrong."))
        assert set(result.keys()) <= AGENT_OWNED_KEYS["type_classifier"]
        assert "severity" not in result
        assert "component" not in result
        assert "summary" not in result


class TestAssessSeverity:
    def test_production_outage_is_p0(self, patch_severity_llm) -> None:
        state = create_initial_state(OUTAGE_REPORT)
        state["bug_type"] = "crash"
        patch_severity_llm(
            SeverityAssessment(
                severity="P0", confidence=0.98, reasoning="Total production outage."
            )
        )
        result = assess_severity(state)
        assert result["severity"] == "P0"

    def test_cosmetic_issue_is_p3(self, patch_severity_llm) -> None:
        state = create_initial_state(COSMETIC_REPORT)
        state["bug_type"] = "ui"
        patch_severity_llm(
            SeverityAssessment(
                severity="P3", confidence=0.92, reasoning="Cosmetic alignment only."
            )
        )
        result = assess_severity(state)
        assert result["severity"] == "P3"

    def test_guardrails(self, patch_severity_llm) -> None:
        patch_severity_llm(
            SeverityAssessment(severity="P2", confidence=0.85, reasoning="Medium impact.")
        )
        state = create_initial_state("Slow page load.")
        state["bug_type"] = "performance"
        result = assess_severity(state)
        assert set(result.keys()) <= AGENT_OWNED_KEYS["severity_assessor"]
        assert "bug_type" not in result
        assert "component" not in result


class TestIdentifyComponent:
    def test_java_stack_trace_backend(self, patch_component_llm, classified_state) -> None:
        patch_component_llm(
            ComponentIdentification(
                component="backend",
                confidence=0.94,
                reasoning="Java service stack trace.",
            )
        )
        result = identify_component(classified_state)
        assert result["component"] == "backend"

    def test_react_css_frontend(self, patch_component_llm) -> None:
        state = create_initial_state(REACT_FRONTEND_REPORT)
        state.update({"bug_type": "ui", "severity": "P3"})
        patch_component_llm(
            ComponentIdentification(
                component="frontend",
                confidence=0.91,
                reasoning="React/CSS layout issue.",
            )
        )
        result = identify_component(state)
        assert result["component"] == "frontend"

    def test_guardrails(self, patch_component_llm, classified_state) -> None:
        patch_component_llm(
            ComponentIdentification(
                component="api", confidence=0.8, reasoning="API layer."
            )
        )
        result = identify_component(classified_state)
        assert set(result.keys()) <= AGENT_OWNED_KEYS["component_identifier"]
        assert "severity" not in result
        assert "bug_type" not in result


class TestSummarize:
    def test_summary_contains_classification_fields(self, full_state) -> None:
        result = summarize(full_state)
        summary = result["summary"]
        assert "ui" in summary
        assert "P3" in summary
        assert "frontend" in summary
        assert "**Classification Result**" in summary
        assert "**Summary**" in summary
        assert "**Suggested Action**" in summary
        assert "**Confidence**" in summary

    def test_summary_llm_path(self, patch_summary_llm) -> None:
        state = create_initial_state("Something weird happened.")
        state.update(
            {
                "bug_type": "logic",
                "severity": "P2",
                "component": "backend",
                "confidence_scores": {"type": 0.5, "severity": 0.9, "component": 0.9},
            }
        )
        patch_summary_llm(
            BugTicketSummary(
                summary=(
                    "**Classification Result**\n"
                    "- Type: logic\n- Severity: P2\n- Component: backend\n\n"
                    "**Summary**: Odd behavior reported.\n\n"
                    "**Suggested Action**: Schedule fix.\n\n"
                    "**Confidence**: Mixed."
                )
            )
        )
        result = summarize(state)
        assert "logic" in result["summary"]
        assert set(result.keys()) <= AGENT_OWNED_KEYS["summary_agent"]

    def test_guardrails(self, full_state) -> None:
        result = summarize(full_state)
        assert "bug_type" not in result
        assert "severity" not in result
        assert "component" not in result
