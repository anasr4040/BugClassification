"""Shared LangGraph state for the multi-agent bug classification system.

Every agent reads from and writes to a single ``BugState`` instance. Because
the type, severity, and component specialists run **in parallel**, fields that
can be written concurrently use reducers (``Annotated[..., reducer]``) so
LangGraph merges simultaneous updates instead of raising an
``InvalidUpdateError``.

The state also carries the supervisor's control-flow fields (verdict, revision
feedback, revision round) that drive the dynamic review/revise loop.
"""

from typing import Annotated, Any, Required, TypedDict


def merge_dicts(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Reducer: merge dict updates from parallel agents (right wins per key)."""
    if right is None:
        return left
    if left is None:
        return dict(right)
    return {**left, **right}


def append_notes(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Reducer: accumulate agent notes from all agents across all rounds."""
    return list(left or []) + list(right or [])


class BugState(TypedDict, total=False):
    """State carried through the bug classification graph.

    Agents receive the accumulated state and return partial updates (dicts)
    that LangGraph merges into this structure. Reducer-annotated fields merge
    concurrent writes from the parallel specialist fan-out.

    Fields:
        raw_report:
            Original bug report text from a user or developer. Set once at
            graph entry (via :func:`create_initial_state`); agents should not
            overwrite it unless correcting input.

        bug_type:
            One of ``"crash"``, ``"performance"``, ``"logic"``,
            ``"security"``, ``"ui"``. Set by the **type classifier** agent
            (``agents/type_classifier.py``).

        severity:
            One of ``"P0"``, ``"P1"``, ``"P2"``, ``"P3"``. Set by the
            **severity assessor** agent (``agents/severity_assessor.py``).

        component:
            One of the configured components (default: ``"backend"``,
            ``"frontend"``, ``"api"``, ``"infrastructure"``, ``"database"``,
            ``"authentication"``). Set by the **component identifier** agent
            (``agents/component_identifier.py``).

        summary:
            Structured summary combining classification outputs. Set by the
            **summary** agent (``agents/summary_agent.py``).

        ticket_url:
            URL of the Notion page created for this classification. Set by the
            **Notion logger** (``integrations/notion_logger.py``) after a page
            is created.

        confidence_scores:
            Per-head confidences, e.g.
            ``{"type": 0.92, "severity": 0.85, "component": 0.78}``.
            Written concurrently by the parallel specialists — merged by
            :func:`merge_dicts`.

        agent_notes:
            Append-only inter-agent message log. Each entry is a dict such as
            ``{"agent": "type_classifier", "note": "...", "tools_used": [...]}``.
            This is how agents (and the supervisor) communicate findings and
            critiques to one another. Merged by :func:`append_notes`.

        supervisor_verdict:
            ``"approve"`` or ``"revise"``. Written by the **supervisor** agent
            after reviewing the joined specialist outputs.

        revision_feedback:
            Per-dimension critique from the supervisor, e.g.
            ``{"type": "Confidence too low; check the stack trace."}``.
            Specialists incorporate their dimension's feedback on rerun.
            Merged by :func:`merge_dicts`.

        revision_targets:
            Dimensions the supervisor wants re-run **this round** (fresh each
            supervisor pass — unlike ``revision_feedback``, which accumulates).

        revision_round:
            Number of revision rounds dispatched so far. Bounds the
            supervisor's review loop (see ``config.MAX_REVISION_ROUNDS``).

        needs_review:
            Set ``True`` by the supervisor when it approves a result it could
            not fully verify (e.g. revision budget exhausted with lingering
            low-confidence dimensions).

        error:
            Human-readable processing or integration failure message. Any agent
            or integration may set this when aborting or degrading gracefully.
    """

    raw_report: Required[str]
    bug_type: str | None
    severity: str | None
    component: str | None
    summary: str | None
    ticket_url: str | None
    confidence_scores: Annotated[dict[str, float] | None, merge_dicts]
    agent_notes: Annotated[list[dict[str, Any]], append_notes]
    supervisor_verdict: str | None
    revision_feedback: Annotated[dict[str, str] | None, merge_dicts]
    revision_targets: list[str] | None
    revision_round: int
    needs_review: bool
    error: str | None


def create_initial_state(raw_report: str) -> BugState:
    """Build starting graph state with only ``raw_report`` populated.

    All classification and downstream fields are explicitly set to ``None``
    (or empty/zero for supervisor bookkeeping) so later nodes can distinguish
    "not yet run" from legitimate values.

    Args:
        raw_report: Verbatim bug report text submitted for classification.

    Returns:
        A ``BugState`` ready to pass to the compiled LangGraph ``invoke`` /
        ``ainvoke`` entrypoint.
    """
    return {
        "raw_report": raw_report,
        "bug_type": None,
        "severity": None,
        "component": None,
        "summary": None,
        "ticket_url": None,
        "confidence_scores": None,
        "agent_notes": [],
        "supervisor_verdict": None,
        "revision_feedback": None,
        "revision_targets": None,
        "revision_round": 0,
        "needs_review": False,
        "error": None,
    }
