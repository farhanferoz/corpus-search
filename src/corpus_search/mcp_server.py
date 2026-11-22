"""MCP endpoint exposing search_corpus() to an agent, stdio transport."""

from __future__ import annotations

from fastmcp import FastMCP

from corpus_search.config import load_config
from corpus_search import daemon as daemon_mod

mcp = FastMCP("corpus-search")


@mcp.tool()
def search_corpus(repo: str, query: str, k: int = 10) -> list[dict]:
    """Search a configured repo's index; returns provenance-carrying passages.

    Args:
        repo: repo slug from corpus-search.toml [repos]
        query: natural-language question (concept-recall, not keyword search)
        k: number of ranked results to return
    """
    cfg = load_config()

    # Always serve through the shared daemon: one loaded model serves every MCP
    # session and every CLI query alike, so N connected agents cost one ~1.25GB
    # copy, not N. If no daemon is up we spin one up on demand (race-safe across
    # cages) and query it — this MCP process stays a thin client and never pulls
    # torch in, keeping the long-lived cage session lightweight. The daemon
    # self-exits on idle, so the memory is only held during an active burst.
    if daemon_mod.is_running(cfg) or daemon_mod.ensure_running(cfg):
        return daemon_mod.query_via_daemon(cfg, repo, query, k=k)

    # Last resort only: the daemon couldn't be started (e.g. spawn blocked). Fall
    # back to a one-off in-process load so a search still works — accepting that
    # this process now holds the model for its lifetime. Import is lazy so the
    # torch cost lands only here, never on the warm path above.
    from corpus_search.query import query_repo, snippet

    hits = query_repo(cfg, repo, query, k=k)
    return [
        {
            "file_path": hit.file_path,
            "heading": hit.heading,
            "score": hit.score,
            "snippet": snippet(hit.text),
        }
        for hit in hits
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
