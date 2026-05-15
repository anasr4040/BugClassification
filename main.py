#!/usr/bin/env python3
"""Bug classifier CLI — single-report and demo modes.

Usage:
    python -m bug_classifier.main --report "App crashes on login..."
    python -m bug_classifier.main --demo
    python -m bug_classifier.main --demo --dry-run --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import Counter
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEMO_EXAMPLES: list[dict[str, str]] = [
    {
        "label": "Example 1 — P0 / crash / authentication (emergency path)",
        "report": (
            "URGENT: Production down. All users getting 500 errors on login. "
            "OAuth token refresh failing with NullPointerException in "
            "AuthService.java:247. Started after last deployment 30 minutes ago. "
            "No users can access the platform."
        ),
    },
    {
        "label": "Example 2 — P2 / performance / database",
        "report": (
            "Dashboard page takes 15+ seconds to load when filtering by date range "
            "> 30 days. Profiling shows slow SQL query on the analytics_events table "
            "— missing index on created_at column. Workaround: use shorter date ranges."
        ),
    },
    {
        "label": "Example 3 — P3 / ui / frontend",
        "report": (
            "The submit button on the contact form overlaps with the footer text on "
            "mobile devices (iPhone SE screen size). Form still works but looks broken. "
            "Low priority."
        ),
    },
    {
        "label": "Example 4 — P1 / security / api",
        "report": (
            "API endpoint /api/v2/users/{id}/profile returns full user data including "
            "hashed_password and internal_notes fields to any authenticated user, not "
            "just admins. This is a data exposure issue."
        ),
    },
    {
        "label": "Example 5 — P2 / logic / backend",
        "report": (
            "Discount calculation applies the coupon twice when user adds items to cart, "
            "removes them, and re-adds. Total shows 40% off instead of 20%. "
            "Reproducible every time."
        ),
    },
    {
        "label": "Example 6 — P0 / security / database (emergency path)",
        "report": (
            "CRITICAL: SQL injection vulnerability confirmed on /api/v1/search endpoint. "
            "Attacker can extract full users table including emails, hashed passwords, "
            "and billing addresses using a crafted query parameter. Exploit is trivial — "
            "no authentication required. Discovered during routine pen test at 2:15 AM."
        ),
    },
    {
        "label": "Example 7 — P1 / crash / backend",
        "report": (
            "Payment processing service crashes with OutOfMemoryError after handling "
            "~500 concurrent transactions. Heap dump shows unbounded list accumulation "
            "in TransactionBatchProcessor.java:189. Affects peak-hour checkout flow. "
            "Service auto-restarts but drops in-flight payments."
        ),
    },
    {
        "label": "Example 8 — P3 / ui / frontend",
        "report": (
            "Dark mode toggle on the settings page does not update the navbar background "
            "color until the user refreshes the page. All other components switch "
            "correctly. Minor visual inconsistency, no functional impact."
        ),
    },
    {
        "label": "Example 9 — P2 / logic / api",
        "report": (
            "Pagination on GET /api/v2/orders returns duplicate records when sorting by "
            "created_at descending. Page 2 repeats the last 3 items from page 1. "
            "Caused by non-unique sort key — orders created in the same second collide. "
            "Workaround: add secondary sort by order_id."
        ),
    },
    {
        "label": "Example 10 — P1 / performance / infrastructure",
        "report": (
            "Kubernetes pods in the recommendation-service deployment are OOMKilled "
            "every 4-6 hours. Memory usage climbs linearly from 512MB to the 2GB limit. "
            "Suspected memory leak in the feature vector cache. Rolling restarts mask "
            "the issue but cause 30-second latency spikes for users each time."
        ),
    },
    {
        "label": "Example 11 — P0 / crash / infrastructure (emergency path)",
        "report": (
            "OUTAGE: Primary Redis cluster failed over at 03:42 UTC. Sentinel promoted "
            "replica but application pool still pointing to old master. All session data "
            "lost — 100% of logged-in users forcibly signed out. Rate limiter also down, "
            "API is accepting unlimited requests. Immediate action required."
        ),
    },
    {
        "label": "Example 12 — P3 / logic / backend",
        "report": (
            "Email notification for 'order shipped' includes the wrong timezone offset. "
            "Timestamps show UTC instead of the user's local timezone. Users in PST see "
            "shipping times 8 hours ahead. Cosmetic confusion only — actual delivery "
            "dates on the tracking page are correct."
        ),
    },
    {
        "label": "Example 13 — P2 / security / authentication",
        "report": (
            "Password reset tokens do not expire after use. A user who resets their "
            "password can reuse the same reset link 24 hours later to set another new "
            "password. Token is only invalidated after the 48-hour TTL, not on "
            "consumption. No active exploitation detected yet."
        ),
    },
    {
        "label": "Example 14 — P1 / logic / database",
        "report": (
            "Inventory count goes negative for high-demand SKUs during flash sales. "
            "Concurrent checkout requests pass the stock > 0 check simultaneously before "
            "any decrement is committed. Resulted in 37 oversold units on last campaign. "
            "Missing row-level lock or atomic decrement in inventory_service."
        ),
    },
    {
        "label": "Example 15 — P3 / performance / frontend",
        "report": (
            "Product image carousel on the listing page loads all 40+ high-res images "
            "eagerly on page load instead of lazy-loading visible ones. Initial page "
            "weight is 18MB on mobile. No crash or error — just slow first paint on "
            "3G connections. Works fine on Wi-Fi."
        ),
    },
    {
        "label": "Example 16 — P0 / security / api (emergency path)",
        "report": (
            "CRITICAL: Admin endpoint /internal/api/config is publicly accessible "
            "without any authentication middleware. Returns full application config "
            "including database connection strings, third-party API keys, and JWT "
            "signing secrets. Discovered by external security researcher who reported "
            "via responsible disclosure."
        ),
    },
    {
        "label": "Example 17 — P2 / crash / frontend",
        "report": (
            "React app throws 'Cannot read properties of undefined' when user navigates "
            "to /profile before account data finishes loading. White screen with error "
            "boundary fallback. Only happens on slow connections. Refreshing the page "
            "fixes it. Missing null check in ProfileHeader component."
        ),
    },
    {
        "label": "Example 18 — P1 / performance / database",
        "report": (
            "Nightly analytics aggregation job that used to complete in 12 minutes now "
            "takes over 3 hours after the users table exceeded 10 million rows. Full "
            "table scan on users JOIN events with no partition pruning. Downstream "
            "dashboards show stale data until job completes. Blocking morning standup."
        ),
    },
    {
        "label": "Example 19 — P3 / ui / frontend",
        "report": (
            "Tooltip text on the 'Export CSV' button says 'Export to XLS' — copy was "
            "never updated after the format migration six months ago. Actual exported "
            "file is correctly CSV. Only the tooltip label is wrong."
        ),
    },
    {
        "label": "Example 20 — P1 / crash / api",
        "report": (
            "GraphQL mutation createInvoice throws 500 Internal Server Error when the "
            "line_items array contains more than 100 entries. Stack trace shows "
            "PayloadTooLargeError in express body-parser. Enterprise clients regularly "
            "submit bulk invoices with 200+ line items. No workaround — they cannot "
            "split submissions."
        ),
    },
]

NODE_LABELS: dict[str, str] = {
    "classify_type": "Type Classifier",
    "assess_severity": "Severity Assessor",
    "identify_component": "Component Identifier",
    "summarize": "Summary Agent",
    "log_to_notion": "Notion Logger",
    "emergency_handler": "Emergency Handler (P0)",
}


class Spinner:
    """Terminal spinner shown while waiting for the next graph node."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Spinner:
        if not sys.stdout.isatty():
            print(f"  {self.message}")
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 72 + "\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        frames = "|/-\\"
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            sys.stdout.write(f"\r{YELLOW}  ⏳ {self.message} {frame}{RESET}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.12)


def _colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def _print_header(title: str) -> None:
    line = "═" * 72
    print(_colorize(line, CYAN))
    print(_colorize(f"  {title}", BOLD + CYAN))
    print(_colorize(line, CYAN))


def _format_confidence(scores: dict[str, float] | None) -> str:
    if not scores:
        return _colorize("  (no confidence scores)", DIM)
    parts = [f"{key}: {value:.0%}" for key, value in scores.items()]
    return _colorize("  Confidence → " + ", ".join(parts), GREEN)


def _print_node_output(
    node_name: str,
    update: dict[str, Any],
    elapsed: float,
    *,
    verbose: bool,
    merged_state: dict[str, Any],
) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    timing = _colorize(f"({elapsed:.2f}s)", DIM)

    print(f"\n{BLUE}  ▶ {label}{RESET} {timing}")

    if node_name == "classify_type":
        print(_colorize(f"    Type: {update.get('bug_type', '—')}", GREEN))
    elif node_name == "assess_severity":
        severity = update.get("severity", "—")
        color = RED if severity == "P0" else GREEN
        print(_colorize(f"    Severity: {severity}", color))
        if severity == "P0":
            print(_colorize("    ⚠ P0 detected — routing to emergency path", YELLOW))
    elif node_name == "identify_component":
        print(_colorize(f"    Component: {update.get('component', '—')}", GREEN))
    elif node_name == "summarize":
        summary = update.get("summary", "")
        preview = summary[:280] + ("…" if len(summary) > 280 else "")
        print(_colorize(f"    Summary:\n{preview}", GREEN))
    elif node_name in ("log_to_notion", "emergency_handler"):
        if update.get("ticket_url"):
            print(_colorize(f"    Notion ticket: {update['ticket_url']}", GREEN))
        if update.get("error"):
            print(_colorize(f"    Notion error: {update['error']}", RED))

    if update.get("confidence_scores"):
        print(_format_confidence(update["confidence_scores"]))

    if verbose:
        print(_colorize("    Full state:", DIM))
        print(json.dumps(merged_state, indent=2, default=str))


def _print_final_summary(
    state: dict[str, Any], total_elapsed: float, node_timings: dict[str, float]
) -> None:
    print(f"\n{BOLD}{'─' * 72}{RESET}")
    print(_colorize("  Pipeline complete", BOLD + GREEN))
    print(f"  Total time: {total_elapsed:.2f}s")
    if node_timings:
        breakdown = ", ".join(f"{k}: {v:.2f}s" for k, v in node_timings.items())
        print(_colorize(f"  Per-node: {breakdown}", DIM))

    print(f"  Type:      {state.get('bug_type', '—')}")
    print(f"  Severity:  {state.get('severity', '—')}")
    print(f"  Component: {state.get('component', '—')}")
    print(_format_confidence(state.get("confidence_scores")))

    if state.get("ticket_url"):
        print(_colorize(f"  Ticket URL: {state['ticket_url']}", GREEN))
    if state.get("error"):
        print(_colorize(f"  Error: {state['error']}", RED))
    print(f"{BOLD}{'─' * 72}{RESET}\n")


def _stream_graph(app: Any, initial: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield graph update chunks, showing a spinner while each node executes."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def worker() -> None:
        try:
            for chunk in app.stream(initial, stream_mode="updates"):
                result_queue.put(("chunk", chunk))
        except Exception as exc:
            result_queue.put(("error", exc))
        finally:
            result_queue.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        with Spinner("Running next agent..."):
            kind, payload = result_queue.get()

        if kind == "done":
            break
        if kind == "error":
            raise payload
        yield payload


def run_interactive(
    app: Any,
    bug_report: str,
    *,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, float], float]:
    """Stream the graph node-by-node with live CLI output and timing."""
    from bug_classifier.observability import log_classification_metrics
    from bug_classifier.state import create_initial_state

    initial = create_initial_state(bug_report)
    merged: dict[str, Any] = dict(initial)
    node_timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    last_mark = pipeline_start

    try:
        for chunk in _stream_graph(app, initial):
            for node_name, update in chunk.items():
                now = time.perf_counter()
                elapsed = now - last_mark
                last_mark = now
                node_timings[node_name] = elapsed
                merged.update(update)
                _print_node_output(
                    node_name, update, elapsed, verbose=verbose, merged_state=merged
                )

        total_elapsed = time.perf_counter() - pipeline_start
        log_classification_metrics(merged)  # type: ignore[arg-type]
        _print_final_summary(merged, total_elapsed, node_timings)
        return merged, node_timings, total_elapsed

    except Exception as exc:
        total_elapsed = time.perf_counter() - pipeline_start
        merged["error"] = str(exc)
        print(_colorize(f"\n  Pipeline failed: {exc}", RED))
        return merged, node_timings, total_elapsed


def _print_demo_aggregate(
    results: list[dict[str, Any]],
    total_wall_time: float,
) -> None:
    count = len(results)
    confidences: list[float] = []
    types: Counter[str] = Counter()
    severities: Counter[str] = Counter()
    components: Counter[str] = Counter()
    errors = 0

    for state in results:
        if state.get("error"):
            errors += 1
        if state.get("bug_type"):
            types[state["bug_type"]] += 1
        if state.get("severity"):
            severities[state["severity"]] += 1
        if state.get("component"):
            components[state["component"]] += 1
        scores = state.get("confidence_scores") or {}
        if scores:
            confidences.append(sum(scores.values()) / len(scores))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    _print_header("DEMO AGGREGATE SUMMARY")
    print(f"  Bugs classified: {count}")
    print(f"  Errors:          {errors}")
    print(f"  Avg confidence:  {avg_conf:.0%}")
    print(f"  Total demo time: {total_wall_time:.2f}s")
    print(_colorize("\n  Type distribution:", BOLD))
    for key, value in sorted(types.items()):
        print(f"    {key}: {value}")
    print(_colorize("  Severity distribution:", BOLD))
    for key, value in sorted(severities.items()):
        print(f"    {key}: {value}")
    print(_colorize("  Component distribution:", BOLD))
    for key, value in sorted(components.items()):
        print(f"    {key}: {value}")
    print()


def run_demo(app: Any, *, verbose: bool = False, pause_seconds: float = 2.0) -> None:
    """Run all demo examples sequentially."""
    _print_header("BUG CLASSIFIER — LIVE DEMO")
    print(_colorize(f"  Running {len(DEMO_EXAMPLES)} example reports\n", DIM))

    results: list[dict[str, Any]] = []
    demo_start = time.perf_counter()

    for index, example in enumerate(DEMO_EXAMPLES, start=1):
        print(f"\n{BOLD}{'━' * 72}{RESET}")
        print(_colorize(f"  {example['label']}", BOLD + CYAN))
        print(f"{BOLD}{'━' * 72}{RESET}")
        print(_colorize("\n  Raw report:", BOLD))
        print(f"  {example['report']}\n")

        state, _, _ = run_interactive(app, example["report"], verbose=verbose)
        results.append(state)

        if index < len(DEMO_EXAMPLES):
            print(_colorize(f"  Pausing {pause_seconds:.0f}s before next example…", DIM))
            time.sleep(pause_seconds)

    _print_demo_aggregate(results, time.perf_counter() - demo_start)


def run_single(app: Any, report: str, *, verbose: bool = False) -> None:
    """Classify a single bug report."""
    _print_header("BUG CLASSIFIER — SINGLE REPORT")
    print(_colorize("\n  Raw report:", BOLD))
    print(f"  {report}\n")
    run_interactive(app, report, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-agent bug classification pipeline (LangGraph + Notion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m bug_classifier.main --report "App crashes on login..."\n'
            "  python -m bug_classifier.main --demo\n"
            "  python -m bug_classifier.main --demo --dry-run --verbose\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--report",
        metavar="TEXT",
        help="Classify a single bug report.",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run five diverse example reports for a live demo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force NOTION_DRY_RUN=true (print ticket payload, no API call).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full merged state after each agent completes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        os.environ["NOTION_DRY_RUN"] = "true"

    # Import after env overrides so config picks up --dry-run.
    from bug_classifier import config
    from bug_classifier.graph.workflow import app

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")

    if config.NOTION_DRY_RUN:
        print(_colorize("  NOTION_DRY_RUN active — tickets will not be created.\n", YELLOW))

    if args.demo:
        run_demo(app, verbose=args.verbose)
    else:
        run_single(app, args.report, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
