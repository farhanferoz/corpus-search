"""Warm-cache daemon: keeps the embedder/reranker/per-repo indexes loaded in
memory across CLI invocations and MCP sessions, so repeat queries skip the cold
model-load and every caller shares one ~1.25GB copy instead of loading its own.

Managed explicitly (`corpus-search daemon start|stop|status`) or spun up on
demand via `ensure_running()` — the latter lets the MCP server serve a query
without cold-loading the model into its own (long-lived) process. The daemon
self-exits after DEFAULT_IDLE_TIMEOUT of inactivity, so on-demand spin-up costs
memory only during an active burst of queries.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

from corpus_search.config import Config

# Idle window before the daemon self-exits and releases its ~1.25GB. Kept short
# so an on-demand spin-up (or a walk-away after a query burst) frees memory
# promptly rather than parking it for an hour; long enough to span the gaps
# within a normal working session so mid-session queries stay warm.
DEFAULT_IDLE_TIMEOUT = 1200.0


def socket_path(cfg: Config) -> Path:
    return cfg.store.dir / "daemon.sock"


def pid_path(cfg: Config) -> Path:
    return cfg.store.dir / "daemon.pid"


def log_path(cfg: Config) -> Path:
    return cfg.store.dir / "daemon.log"


def is_running(cfg: Config) -> bool:
    sock = socket_path(cfg)
    if not sock.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(str(sock))
        return True
    except OSError:
        return False


def _handle_query(req: dict) -> dict:
    from corpus_search.config import load_config
    from corpus_search.query import query_repo, snippet

    cfg = load_config()
    # rerank may be None (→ use config default), True, or False; don't coerce None.
    rerank = req.get("rerank")
    hits = query_repo(
        cfg,
        req["repo"],
        req["query"],
        k=int(req.get("k", 10)),
        rerank=rerank,
    )
    return {
        "ok": True,
        "hits": [
            {"file_path": h.file_path, "heading": h.heading, "score": h.score, "snippet": snippet(h.text)}
            for h in hits
        ],
    }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.server.last_active = time.monotonic()
        line = self.rfile.readline()
        if not line:
            return
        try:
            req = json.loads(line.decode())
        except json.JSONDecodeError as e:
            self._reply({"ok": False, "error": f"bad request: {e}"})
            return

        cmd = req.get("cmd")
        if cmd == "ping":
            self._reply({"ok": True})
        elif cmd == "shutdown":
            self._reply({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        elif cmd == "invalidate":
            from corpus_search.query import invalidate_index

            invalidate_index(req.get("repo", ""))
            self._reply({"ok": True})
        else:
            try:
                self._reply(_handle_query(req))
            except Exception as e:
                self._reply({"ok": False, "error": str(e)})

        self.server.last_active = time.monotonic()

    def _reply(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode())


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def _idle_watchdog(server: _Server, idle_timeout: float) -> None:
    while True:
        time.sleep(5)
        if time.monotonic() - server.last_active > idle_timeout:
            server.shutdown()
            return


def run(cfg: Config, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
    """Run the daemon in the foreground (blocks until shutdown/idle-timeout/signal)."""
    from corpus_search.query import warmup

    # Warm up BEFORE binding the socket, so is_running() only answers true once the
    # daemon can actually serve fast — `daemon start` then blocks until warm, and the
    # first real query never eats the model-load cost.
    warmup(cfg)

    sock_path = socket_path(cfg)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    sock_path.parent.chmod(0o700)
    if sock_path.exists():
        sock_path.unlink()

    server = _Server(str(sock_path), _Handler)
    server.last_active = time.monotonic()
    sock_path.chmod(0o700)
    pid_path(cfg).write_text(str(os.getpid()))

    def _cleanup(*_a):
        sock_path.unlink(missing_ok=True)
        pid_path(cfg).unlink(missing_ok=True)

    def _on_signal(signum, frame):
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    threading.Thread(target=_idle_watchdog, args=(server, idle_timeout), daemon=True).start()

    try:
        server.serve_forever()
    finally:
        _cleanup()


def stop(cfg: Config) -> bool | None:
    """Signal the daemon to stop.

    Returns True if a stop signal was sent, False if it wasn't running (no
    pidfile), or None if the pidfile was stale — the process it named is
    already gone, so the orphaned pidfile/socket are cleaned up here rather
    than left behind to make a dead daemon look alive.
    """
    pf = pid_path(cfg)
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pf.unlink(missing_ok=True)
        socket_path(cfg).unlink(missing_ok=True)
        return None
    except ValueError:
        pass
    return True


def start_detached(cfg: Config, idle_timeout: float = DEFAULT_IDLE_TIMEOUT, start_timeout: float = 180.0) -> None:
    """Launch the daemon as a detached background process; block until its socket answers."""
    import subprocess

    lp = log_path(cfg)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.chmod(0o700)
    with open(lp, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "corpus_search.daemon_main", "--idle-timeout", str(idle_timeout)],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        if is_running(cfg):
            return
        time.sleep(0.5)
    raise TimeoutError(f"corpus-search daemon didn't come up within {start_timeout}s; check {lp}")


def ensure_running(cfg: Config, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> bool:
    """Guarantee a shared daemon is up, spawning one if needed. Returns True on success.

    Race-safe across concurrent callers (e.g. several cages first-querying at once):
    an exclusive file lock serializes the spawn so only one daemon is ever launched —
    the losers re-check under the lock and find the winner's daemon already serving.
    Never raises: on any spawn failure it returns False so the caller can fall back
    to a cold in-process load rather than error out.
    """
    if is_running(cfg):
        return True
    import fcntl

    lock = cfg.store.dir / "daemon.start.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.chmod(0o700)
    try:
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            # Double-check under the lock: another caller may have won the race
            # and already brought the daemon up while we waited for the lock.
            if is_running(cfg):
                return True
            start_detached(cfg, idle_timeout=idle_timeout)
            return is_running(cfg)
    except (OSError, TimeoutError):
        return False


def _request(cfg: Config, payload: dict, timeout: float = 300.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(socket_path(cfg)))
        s.sendall((json.dumps(payload) + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            buf = s.recv(65536)
            if not buf:
                break
            chunks.append(buf)
    return json.loads(b"".join(chunks).decode())


def query_via_daemon(cfg: Config, repo: str, question: str, k: int, rerank: bool | None = None) -> list[dict]:
    resp = _request(cfg, {"repo": repo, "query": question, "k": k, "rerank": rerank})
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "daemon error"))
    return resp["hits"]


def invalidate(cfg: Config, repo_slug: str) -> None:
    """Best-effort: tell a running daemon to drop its cached index for repo_slug."""
    try:
        _request(cfg, {"cmd": "invalidate", "repo": repo_slug}, timeout=2.0)
    except OSError:
        pass
