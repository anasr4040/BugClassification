"""Shared pytest fixtures for bug_classifier tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

# Set required env vars before any bug_classifier imports in test modules.
os.environ.setdefault("XAI_API_KEY", "test-xai-key")
os.environ.setdefault("GROK_API_KEY", "test-xai-key")
os.environ.setdefault("NOTION_API_KEY", "test-notion-key")
os.environ.setdefault("NOTION_DATABASE_ID", "00000000-0000-0000-0000-000000000001")
os.environ.setdefault("NOTION_DRY_RUN", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from bug_classifier.agents.component_identifier import ComponentIdentification
from bug_classifier.agents.severity_assessor import SeverityAssessment
from bug_classifier.agents.summary_agent import BugTicketSummary
from bug_classifier.agents.type_classifier import TypeClassification
from bug_classifier.state import BugState, create_initial_state


# ---------------------------------------------------------------------------
# Sample reports
# ---------------------------------------------------------------------------
CRASH_REPORT = (
    "App crashes on startup with SIGSEGV. Stack trace: main.cpp:42. "
    "Happens on every launch for all users."
)
AMBIGUOUS_REPORT = "Something feels off sometimes but hard to reproduce."
OUTAGE_REPORT = (
    "URGENT: Production is completely down. All users affected. "
    "Database unreachable. Started 10 minutes ago."
)
COSMETIC_REPORT = (
    "Submit button has 2px misalignment on the settings page footer. "
    "Purely cosmetic, no functional impact."
)
JAVA_BACKEND_REPORT = (
    "NullPointerException in OrderService.java:247 when processing checkout. "
    "Stack trace points to com.example.orders.OrderService.processPayment."
)
REACT_FRONTEND_REPORT = (
    "React component ContactForm throws CSS layout error; flexbox footer "
    "overlaps the submit button in Chrome mobile view."
)


@pytest.fixture
def crash_state() -> BugState:
    return create_initial_state(CRASH_REPORT)


@pytest.fixture
def ambiguous_state() -> BugState:
    return create_initial_state(AMBIGUOUS_REPORT)


@pytest.fixture
def classified_state() -> BugState:
    """State after type + severity classification."""
    state = create_initial_state(JAVA_BACKEND_REPORT)
    state.update(
        {
            "bug_type": "crash",
            "severity": "P1",
            "confidence_scores": {"type": 0.92, "severity": 0.88},
        }
    )
    return state


@pytest.fixture
def full_state() -> BugState:
    """Fully classified state with high confidence (summary fast-path)."""
    state = create_initial_state(REACT_FRONTEND_REPORT)
    state.update(
        {
            "bug_type": "ui",
            "severity": "P3",
            "component": "frontend",
            "confidence_scores": {"type": 0.95, "severity": 0.91, "component": 0.93},
        }
    )
    return state


class MockStructuredLLM:
    """Stand-in for ``llm.with_structured_output(...).invoke(...)``."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.invoke = MagicMock(return_value=response)


class MockChatXAI:
    """Stand-in for ChatXAI that returns a fixed structured output."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def with_structured_output(self, schema: Any) -> MockStructuredLLM:
        return MockStructuredLLM(self._response)


def patch_llm(monkeypatch: pytest.MonkeyPatch, target: str, response: Any) -> None:
    """Patch ``_build_llm`` in an agent module to return deterministic output."""
    monkeypatch.setattr(target, lambda: MockChatXAI(response))


@pytest.fixture
def patch_type_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    def _patch(response: TypeClassification) -> None:
        patch_llm(monkeypatch, "bug_classifier.agents.type_classifier._build_llm", response)

    return _patch


@pytest.fixture
def patch_severity_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    def _patch(response: SeverityAssessment) -> None:
        patch_llm(
            monkeypatch, "bug_classifier.agents.severity_assessor._build_llm", response
        )

    return _patch


@pytest.fixture
def patch_component_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    def _patch(response: ComponentIdentification) -> None:
        patch_llm(
            monkeypatch, "bug_classifier.agents.component_identifier._build_llm", response
        )

    return _patch


@pytest.fixture
def patch_summary_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    def _patch(response: BugTicketSummary) -> None:
        patch_llm(monkeypatch, "bug_classifier.agents.summary_agent._build_llm", response)

    return _patch


@pytest.fixture
def patch_all_agents_default(
    monkeypatch: pytest.MonkeyPatch,
    patch_type_llm,
    patch_severity_llm,
    patch_component_llm,
) -> None:
    """Patch type, severity, and component agents with sensible defaults."""
    patch_type_llm(
        TypeClassification(type="logic", confidence=0.9, reasoning="Default mock type.")
    )
    patch_severity_llm(
        SeverityAssessment(severity="P2", confidence=0.9, reasoning="Default mock severity.")
    )
    patch_component_llm(
        ComponentIdentification(
            component="backend", confidence=0.9, reasoning="Default mock component."
        )
    )


@pytest.fixture
def fresh_app(patch_all_agents_default):
    """Compiled graph with mocked LLM agents."""
    from bug_classifier.graph.workflow import build_graph

    return build_graph()
