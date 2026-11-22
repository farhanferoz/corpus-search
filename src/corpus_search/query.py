"""Query a per-repo index: hybrid retrieve, then optional cross-encoder rerank.

Reranking is off by default (see corpus-search.toml [rerank]) — hybrid retrieval
alone is near-instant and matched the reranker's top-5 quality on the pilot, while
the cross-encoder pass costs minutes per query on CPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from corpus_search.config import Config
from corpus_search.indexer import load_index

_reranker_cache: dict[str, object] = {}
_index_cache: dict[str, object] = {}

# Max combined (query + passage) token length for the cross-encoder reranker.
# The tokenizer truncates to this, so whole-section chunks can't overflow the model.
_RERANK_MAX_TOKENS = 512


def _get_index(cfg: Config, repo_slug: str):
    if repo_slug not in _index_cache:
        _index_cache[repo_slug] = load_index(cfg, repo_slug)
    return _index_cache[repo_slug]


def invalidate_index(repo_slug: str) -> None:
    _index_cache.pop(repo_slug, None)


@dataclass
class Hit:
    text: str
    file_path: str
    heading: str
    score: float


def _get_reranker(model: str):
    if model not in _reranker_cache:
        from sentence_transformers import CrossEncoder

        # max_length caps the combined (query, passage) token length and truncates —
        # essential for small cross-encoders (MiniLM = 512) fed whole-section chunks.
        _reranker_cache[model] = CrossEncoder(model, max_length=_RERANK_MAX_TOKENS)
    return _reranker_cache[model]


def _retrieve(cfg: Config, repo_slug: str, question: str, k: int) -> list[Hit]:
    embeddings = _get_index(cfg, repo_slug)
    sql = (
        "select text, file_path, heading, score from txtai "
        f"where similar(:q) limit {int(k)}"
    )
    rows = embeddings.search(sql, parameters={"q": question})
    return [
        Hit(text=r["text"], file_path=r["file_path"], heading=r["heading"], score=r["score"])
        for r in rows
    ]


def query_repo(
    cfg: Config,
    repo_slug: str,
    question: str,
    k: int = 10,
    rerank: bool | None = None,
) -> list[Hit]:
    """Query a repo. rerank=None (default) uses cfg.rerank.enabled; pass True/False to override."""
    if rerank is None:
        rerank = cfg.rerank.enabled

    candidates = _retrieve(cfg, repo_slug, question, cfg.rerank.top_k if rerank else k)
    if not rerank or not candidates:
        return candidates[:k]

    reranker = _get_reranker(cfg.rerank.model)
    scores = reranker.predict([(question, c.text) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    return [
        Hit(
            text=candidates[i].text,
            file_path=candidates[i].file_path,
            heading=candidates[i].heading,
            score=float(scores[i]),
        )
        for i in order[:k]
    ]


def warmup(cfg: Config) -> None:
    """Eagerly load every configured repo's index (and the reranker only if enabled).

    Called at daemon startup so the load cost lands predictably on `daemon start`,
    not on whatever query happens to hit the daemon first.
    """
    if cfg.rerank.enabled:
        _get_reranker(cfg.rerank.model)
    for slug in cfg.repos:
        # A repo with no index -- or whose directory has been moved away entirely -- must not stop
        # the daemon starting for all the others. `cli.search_all` already skips such a repo on
        # every query, so warming was the one place that treated it as fatal, and one unindexed
        # repo out of thirteen made `daemon start` impossible.
        try:
            _get_index(cfg, slug)
        except FileNotFoundError as exc:
            print(f"[{slug}] skipped while warming: {exc}", file=sys.stderr)


def snippet(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
