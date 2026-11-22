"""corpus-search CLI: index and query per-repo hybrid search.

The `query` (warm path), `daemon`, and top-level commands stay light — the heavy
ML stack (txtai/torch) is imported lazily, only in the code paths that actually
load models (index build, and the cold-query fallback when no daemon is running).
So `corpus-search daemon status` and daemon-served queries don't pay a torch
import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from corpus_search import daemon as daemon_mod
from corpus_search.config import Config, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
daemon_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Manage the warm-cache daemon.")
app.add_typer(daemon_app, name="daemon")


@app.command()
def index(
    repo: str = typer.Argument(..., help="Repo slug from corpus-search.toml [repos]"),
    full: bool = typer.Option(
        False,
        "--full",
        help="Force a full rebuild instead of the default per-file incremental update "
        "(use after changing the embed model or chunk settings)",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to corpus-search.toml"),
):
    """Build the hybrid index for a repo.

    Incremental by default: only files whose mtime advanced (or that are new) are
    re-embedded, and chunks for changed/removed files are dropped — falling back
    to a full build when the repo has no index yet. --full forces a rebuild.
    """
    from corpus_search.indexer import build_index, update_index

    cfg = load_config(config)
    result = build_index(cfg, repo) if full else update_index(cfg, repo)
    typer.echo(json.dumps(result, indent=2))
    if daemon_mod.is_running(cfg):
        daemon_mod.invalidate(cfg, repo)


@app.command("index-all")
def index_all(
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip repos that already have a built index"
    ),
    changed_only: bool = typer.Option(
        False,
        "--changed-only",
        help="Only (re)index repos missing an index or whose sources changed since the last build",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Force a full rebuild of each repo instead of the default per-file incremental update",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to corpus-search.toml"),
):
    """Build the hybrid index for every repo in [repos].

    Each repo is updated incrementally by default (only changed files are
    re-embedded), falling back to a full build when it has no index. With
    --changed-only, repos whose source files are all older than their last index
    build are skipped entirely via a cheap stat walk — so a routine refresh
    touches only repos that changed, and within those only the files that changed.
    --full forces a full rebuild of each repo it does process.
    """
    from corpus_search.indexer import build_index, is_stale, store_path, update_index

    cfg = load_config(config)
    running = daemon_mod.is_running(cfg)
    results = []
    for slug in cfg.repos:
        if changed_only and not is_stale(cfg, slug):
            typer.echo(f"[{slug}] up to date — skip")
            continue
        if skip_existing and store_path(cfg, slug).exists():
            typer.echo(f"[{slug}] skip (index exists)")
            continue
        typer.echo(f"[{slug}] indexing…")
        try:
            result = build_index(cfg, slug) if full else update_index(cfg, slug)
        except (FileNotFoundError, ValueError) as exc:
            # Skip, don't abort: this loop used to stop at the first unindexable repo and leave
            # every repo after it in the config unbuilt.
            typer.echo(f"[{slug}] skipped — {exc}", err=True)
            continue
        results.append(result)
        extra = "" if result["mode"] == "full" else f" ({result['mode']})"
        typer.echo(f"[{slug}] done — {result['chunks']} chunks, {result['documents']}{extra}")
        if running:
            daemon_mod.invalidate(cfg, slug)
    typer.echo(json.dumps(results, indent=2))


def _warn_if_stale(cfg: Config, repo: str) -> None:
    """Best-effort: warn on stderr if repo has a built index older than its
    source files. Stat-only (see indexer.is_stale), so cheap enough to run on
    every query; never lets a staleness check itself break the query.
    """
    from corpus_search.indexer import is_stale, store_path

    try:
        if store_path(cfg, repo).exists() and is_stale(cfg, repo):
            typer.echo(
                f"[{repo}] index is older than source files — run: corpus-search index {repo}",
                err=True,
            )
    except KeyError:
        pass  # repo dropped from config since it was indexed; not our problem here


def _print_hits(hits: list[dict], show_repo: bool = False) -> None:
    for i, h in enumerate(hits, start=1):
        heading = f":{h['heading']}" if h["heading"] else ""
        prefix = f"[{h['repo']}] " if show_repo else ""
        typer.echo(f"{i}. {prefix}{h['file_path']}{heading}  (score={h['score']:.3f})")
        typer.echo(f"   {h['snippet']}")


@app.command()
def query(
    repo: str = typer.Argument(..., help="Repo slug from corpus-search.toml [repos]"),
    question: str = typer.Argument(..., help="Natural-language query"),
    k: int = typer.Option(10, "--k", help="Number of results to return"),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Override cross-encoder reranking (default: config [rerank] enabled)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to corpus-search.toml"),
):
    """Query a repo's index and print ranked, provenance-carrying hits.

    Uses the warm daemon (`corpus-search daemon start`) if it's running;
    otherwise falls back to a one-off cold-loaded query.
    """
    cfg = load_config(config)
    _warn_if_stale(cfg, repo)

    # Warm the shared daemon on demand: the first search in a fresh session pays
    # the one-time model load, then it and every later query (from any cage or
    # the MCP server) run against one warm copy. Only if the daemon can't be
    # started do we cold-load in this short-lived process.
    if daemon_mod.is_running(cfg) or daemon_mod.ensure_running(cfg):
        try:
            hits = daemon_mod.query_via_daemon(cfg, repo, question, k=k, rerank=rerank)
        except RuntimeError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=1)
        _print_hits(hits)
        return

    typer.echo(
        "(couldn't start the daemon — cold-loading models for this one query)",
        err=True,
    )
    from corpus_search.query import query_repo, snippet

    try:
        results = query_repo(cfg, repo, question, k=k, rerank=rerank)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    _print_hits(
        [{"file_path": h.file_path, "heading": h.heading, "score": h.score, "snippet": snippet(h.text)} for h in results]
    )


@app.command("search-all")
def search_all(
    question: str = typer.Argument(..., help="Natural-language query"),
    k: int = typer.Option(10, "--k", help="Number of results to return, merged across all repos"),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Override cross-encoder reranking (default: config [rerank] enabled)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to corpus-search.toml"),
):
    """Query every configured repo and print hits merged by score across repos.

    Uses the warm daemon (`corpus-search daemon start`) if it's running;
    otherwise falls back to a one-off cold-loaded query per repo.
    """
    cfg = load_config(config)
    all_hits: list[dict] = []

    # Warm the shared daemon on demand (see `query`): one model copy serves every
    # per-repo query below, and stays warm for later searches this session.
    if daemon_mod.is_running(cfg) or daemon_mod.ensure_running(cfg):
        for slug in cfg.repos:
            try:
                hits = daemon_mod.query_via_daemon(cfg, slug, question, k=k, rerank=rerank)
            except RuntimeError as e:
                typer.echo(f"[{slug}] error: {e}", err=True)
                continue
            if hits:
                _warn_if_stale(cfg, slug)
            for h in hits:
                h["repo"] = slug
            all_hits.extend(hits)
    else:
        typer.echo(
            "(couldn't start the daemon — cold-loading models per repo for this one query)",
            err=True,
        )
        from corpus_search.query import query_repo, snippet

        for slug in cfg.repos:
            try:
                results = query_repo(cfg, slug, question, k=k, rerank=rerank)
            except FileNotFoundError as e:
                typer.echo(f"[{slug}] error: {e}", err=True)
                continue
            if results:
                _warn_if_stale(cfg, slug)
            all_hits.extend(
                {"file_path": h.file_path, "heading": h.heading, "score": h.score, "snippet": snippet(h.text), "repo": slug}
                for h in results
            )

    all_hits.sort(key=lambda h: h["score"], reverse=True)
    _print_hits(all_hits[:k], show_repo=True)


@daemon_app.command("start")
def daemon_start(
    idle_timeout: float = typer.Option(daemon_mod.DEFAULT_IDLE_TIMEOUT, "--idle-timeout", help="Seconds of inactivity before auto-exit"),
    foreground: bool = typer.Option(False, "--foreground", help="Run in this process instead of detaching"),
):
    """Start the warm-cache daemon (loads the embedder once, serves queries over a local socket)."""
    cfg = load_config()
    if daemon_mod.is_running(cfg):
        typer.echo("already running")
        raise typer.Exit()
    if foreground:
        daemon_mod.run(cfg, idle_timeout=idle_timeout)
    else:
        typer.echo("starting daemon (warming models)…")
        daemon_mod.start_detached(cfg, idle_timeout=idle_timeout)
        typer.echo(f"daemon ready (auto-exits after {idle_timeout:.0f}s idle)")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the warm-cache daemon if running."""
    cfg = load_config()
    result = daemon_mod.stop(cfg)
    if result is True:
        typer.echo("stop signal sent")
    elif result is None:
        typer.echo("daemon was not running (cleaned up stale files)")
    else:
        typer.echo("not running")


@daemon_app.command("status")
def daemon_status():
    """Report whether the warm-cache daemon is running."""
    cfg = load_config()
    typer.echo("running" if daemon_mod.is_running(cfg) else "not running")


def main() -> None:
    app()
