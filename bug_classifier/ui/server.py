"""Local web UI for the multi-agent bug classifier.

Serves a single-page control room (``index.html``) and streams live graph
execution events over Server-Sent Events, so the browser can animate each
agent receiving its request, calling tools, responding, and handing off to
the supervisor.

Endpoints
---------
- ``GET  /``                    → the UI page
- ``GET  /api/meta``            → demo examples + runtime flags
- ``POST /api/classify``        → start a classification run ``{"report": ...}``
- ``POST /api/tests``           → run the pytest suite in a subprocess
- ``GET  /api/events/<run_id>`` → SSE stream of that run's events

Only stdlib is used (``http.server`` + threads) so the UI adds no
dependencies. One run executes at a time; the UI disables its controls while
a run is active.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _UI_DIR.parent.parent

_QUEUE_SENTINEL = None
_SSE_HEARTBEAT_SECONDS = 15.0


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of graph state into JSON-serializable data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RunRegistry:
    """Tracks event queues for active runs; enforces one run at a time."""

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._active = False

    def create(self) -> str | None:
        """Reserve a new run; returns ``None`` when another run is active."""
        with self._lock:
            if self._active:
                return None
            self._active = True
            run_id = uuid.uuid4().hex[:12]
            self._queues[run_id] = queue.Queue()
            return run_id

    def get(self, run_id: str) -> queue.Queue | None:
        with self._lock:
            return self._queues.get(run_id)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        target = self.get(run_id)
        if target is not None:
            target.put(event)

    def finish(self, run_id: str) -> None:
        """Mark the run finished and signal the SSE stream to close."""
        with self._lock:
            self._active = False
            target = self._queues.get(run_id)
        if target is not None:
            target.put(_QUEUE_SENTINEL)

    def discard(self, run_id: str) -> None:
        with self._lock:
            self._queues.pop(run_id, None)


REGISTRY = RunRegistry()


def _classification_worker(run_id: str, report: str) -> None:
    """Execute the graph and translate its stream into UI events."""
    # Imported lazily so the server can start (and report config errors over
    # SSE) even before env vars are fully sorted.
    from bug_classifier.graph.workflow import app
    from bug_classifier.main import _merge_state_update
    from bug_classifier.observability import log_classification_metrics
    from bug_classifier.state import create_initial_state

    def emit(event_type: str, **payload: Any) -> None:
        REGISTRY.publish(run_id, {"type": event_type, **payload})

    initial = create_initial_state(report)
    merged: dict[str, Any] = dict(initial)
    emit("run_started", report=report)

    try:
        for mode, chunk in app.stream(initial, stream_mode=["debug", "updates"]):
            if mode == "debug" and isinstance(chunk, dict):
                if chunk.get("type") == "task":
                    node = (chunk.get("payload") or {}).get("name")
                    if node:
                        emit("node_start", node=node)
            elif mode == "updates" and isinstance(chunk, dict):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    _merge_state_update(merged, update)
                    emit("node_end", node=node, update=_json_safe(update))

        log_classification_metrics(merged)  # type: ignore[arg-type]
        emit("run_complete", state=_json_safe(merged))
    except Exception as exc:
        logger.exception("UI classification run failed")
        emit("run_error", message=str(exc))
    finally:
        REGISTRY.finish(run_id)


def _tests_worker(run_id: str) -> None:
    """Run the pytest suite and stream its output line by line."""
    import os

    def emit(event_type: str, **payload: Any) -> None:
        REGISTRY.publish(run_id, {"type": event_type, **payload})

    emit("tests_started")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", "bug_classifier/tests", "-q"],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "NOTION_DRY_RUN": "true"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            emit("test_line", line=line.rstrip("\n"))
        exit_code = process.wait()
        emit("tests_complete", exit_code=exit_code)
    except Exception as exc:
        logger.exception("UI test run failed")
        emit("run_error", message=str(exc))
    finally:
        REGISTRY.finish(run_id)


class UIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ util
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("ui-http: " + format, *args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index()
        elif path == "/api/meta":
            self._serve_meta()
        elif path.startswith("/api/events/"):
            self._serve_events(path.rsplit("/", 1)[-1])
        else:
            self._send_json({"error": "not found"}, status=404)

    def _serve_index(self) -> None:
        try:
            body = (_UI_DIR / "index.html").read_bytes()
        except OSError:
            self._send_json({"error": "index.html missing"}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_meta(self) -> None:
        from bug_classifier import config
        from bug_classifier.main import DEMO_EXAMPLES

        self._send_json(
            {
                "dry_run": config.NOTION_DRY_RUN,
                "confidence_threshold": config.CONFIDENCE_REVIEW_THRESHOLD,
                "max_revision_rounds": config.MAX_REVISION_ROUNDS,
                "examples": DEMO_EXAMPLES,
            }
        )

    def _serve_events(self, run_id: str) -> None:
        events = REGISTRY.get(run_id)
        if events is None:
            self._send_json({"error": "unknown run"}, status=404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                try:
                    event = events.get(timeout=_SSE_HEARTBEAT_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                if event is _QUEUE_SENTINEL:
                    self.wfile.write(b"data: {\"type\": \"stream_end\"}\n\n")
                    self.wfile.flush()
                    break
                data = json.dumps(event, default=str)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            REGISTRY.discard(run_id)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/classify":
            self._start_classification()
        elif path == "/api/tests":
            self._start_tests()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _start_classification(self) -> None:
        body = self._read_json_body()
        report = (body.get("report") or "").strip()
        if not report:
            self._send_json({"error": "report text is required"}, status=400)
            return
        run_id = REGISTRY.create()
        if run_id is None:
            self._send_json({"error": "another run is in progress"}, status=409)
            return
        threading.Thread(
            target=_classification_worker, args=(run_id, report), daemon=True
        ).start()
        self._send_json({"run_id": run_id})

    def _start_tests(self) -> None:
        run_id = REGISTRY.create()
        if run_id is None:
            self._send_json({"error": "another run is in progress"}, status=409)
            return
        threading.Thread(target=_tests_worker, args=(run_id,), daemon=True).start()
        self._send_json({"run_id": run_id})


def serve(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True):
    """Start the UI server (blocking). Returns the server on ``port=0`` tests."""
    server = ThreadingHTTPServer((host, port), UIRequestHandler)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Bug classifier UI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI server.")
    finally:
        server.server_close()


def create_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Build (but do not run) a server — used by tests."""
    return ThreadingHTTPServer((host, port), UIRequestHandler)
