"""Build and persist a per-repo txtai hybrid (BM25 + dense) index.

Chunks are stored under deterministic string ids (``<file_path>#<chunk_index>``)
and a sidecar ``manifest.json`` records each file's mtime and chunk count. That
lets :func:`update_index` re-embed only the files that actually changed (delete
the old chunk ids for changed/removed files, upsert the new ones) instead of
rebuilding the whole repo — the finer-grained counterpart to the repo-level
``is_stale`` skip. A repo indexed before manifests existed (or with no index at
all) transparently falls back to a full :func:`build_index` on the first update.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from corpus_search.chunk import chunk_document
from corpus_search.config import Config
from corpus_search.ingest import Document, IngestResult, ingest_repo, write_skipped_log

if TYPE_CHECKING:  # annotation-only; the real import stays lazy (torch is heavy)
    from txtai import Embeddings

SELECT_FIELDS = "text, file_path, heading, chunk_index, source_type, score"
MANIFEST_NAME = "manifest.json"


def store_path(cfg: Config, repo_slug: str) -> Path:
    return cfg.store.dir / repo_slug


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / MANIFEST_NAME


def _chunk_id(file_path: str, chunk_index: int) -> str:
    return f"{file_path}#{chunk_index}"


def _records_for_doc(doc: Document, cfg: Config, repo_slug: str) -> list[tuple]:
    """(id, data, tags) txtai records for one document, under deterministic ids."""
    fp = str(doc.file_path)
    records = []
    for c in chunk_document(doc, cfg.chunk):
        records.append(
            (
                _chunk_id(fp, c.chunk_index),
                {
                    "text": c.text,
                    "file_path": fp,
                    "heading": c.heading or "",
                    "chunk_index": c.chunk_index,
                    "source_type": doc.source_type,
                    "repo": repo_slug,
                },
                None,
            )
        )
    return records


def _doc_mtime(doc: Document) -> float:
    try:
        return doc.file_path.stat().st_mtime
    except OSError:
        return 0.0


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = _manifest_path(out_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    tmp.replace(_manifest_path(out_dir))


def _read_manifest(out_dir: Path) -> dict | None:
    try:
        return json.loads(_manifest_path(out_dir).read_text())
    except (OSError, ValueError):
        return None


def _persist(cfg: Config, out_dir: Path, embeddings, result: IngestResult, manifest: dict) -> None:
    """Common save path: chmod-guarded store dir, embeddings, manifest, skipped log."""
    cfg.store.dir.mkdir(parents=True, exist_ok=True)
    cfg.store.dir.chmod(0o700)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings.save(str(out_dir))
    out_dir.chmod(0o700)
    _write_manifest(out_dir, manifest)
    write_skipped_log(result.skipped_pdfs, out_dir / "skipped.jsonl")


def _counts(documents: list[Document]) -> dict:
    counts = {"prose": 0, "pdf": 0, "notebook": 0}
    for doc in documents:
        counts[doc.source_type] += 1
    return counts


def build_index(cfg: Config, repo_slug: str) -> dict:
    """Full (re)build: ingest, chunk, embed every document from scratch."""
    # Late import: txtai drags in torch, so keep it out of stat-only callers
    # (is_stale, index_mtime, store_path) that light paths like the warm
    # `query`/`search-all` CLI commands need to stay cheap.
    from txtai import Embeddings

    repo_root = cfg.repo_path(repo_slug)
    # A configured repo whose directory has been moved away ingests zero documents, and txtai
    # then fails inside `save` with `'NoneType' object has no attribute 'commit'` -- an error
    # that names neither the repo nor the cause. Say which repo and why, here.
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"repo {repo_slug!r} is configured at {repo_root}, which does not exist"
        )
    result = ingest_repo(repo_root, cfg)

    records: list[tuple] = []
    manifest: dict = {}
    for doc in result.documents:
        doc_records = _records_for_doc(doc, cfg, repo_slug)
        records.extend(doc_records)
        manifest[str(doc.file_path)] = {
            "mtime": _doc_mtime(doc),
            "n_chunks": len(doc_records),
            "source_type": doc.source_type,
        }

    # Checked BEFORE the model loads: an empty index cannot be saved, so there is nothing to
    # spend an embedder on.
    if not records:
        raise ValueError(f"repo {repo_slug!r} at {repo_root} holds nothing indexable")

    embeddings = Embeddings(path=cfg.embed.model, content=True, hybrid=True)
    embeddings.index(records)

    out_dir = store_path(cfg, repo_slug)
    _persist(cfg, out_dir, embeddings, result, manifest)

    return {
        "repo": repo_slug,
        "mode": "full",
        "documents": _counts(result.documents),
        "chunks": len(records),
        "skipped_pdfs": len(result.skipped_pdfs),
        "store": str(out_dir),
    }


def update_index(cfg: Config, repo_slug: str) -> dict:
    """Incremental (re)build: re-embed only files whose mtime advanced or that are
    new, drop chunks for changed/removed files, and keep everything else in place.

    Falls back to a full :func:`build_index` when there is no existing index or no
    manifest to diff against (e.g. an index built before manifests existed). The
    repo walk + ingest (including PDF text extraction) still runs in full — only
    the expensive embedding step is scoped to changed files — so dedup decisions
    (a PDF dropping out once its markdown twin appears, and vice-versa) stay
    correct across updates.
    """
    out_dir = store_path(cfg, repo_slug)
    manifest = _read_manifest(out_dir)
    if manifest is None or not (out_dir / "embeddings").exists():
        return build_index(cfg, repo_slug)  # nothing to diff against → full build

    repo_root = cfg.repo_path(repo_slug)
    result = ingest_repo(repo_root, cfg)

    current = {str(doc.file_path): doc for doc in result.documents}
    current_mtime = {fp: _doc_mtime(doc) for fp, doc in current.items()}

    changed = {
        fp for fp, mt in current_mtime.items()
        if fp not in manifest or mt > manifest[fp]["mtime"]
    }
    deleted = {fp for fp in manifest if fp not in current}

    if not changed and not deleted:
        return {
            "repo": repo_slug,
            "mode": "unchanged",
            "documents": _counts(result.documents),
            "chunks": sum(e["n_chunks"] for e in manifest.values()),
            "changed": 0,
            "deleted": 0,
            "upserted_chunks": 0,
            "store": str(out_dir),
        }

    from txtai import Embeddings

    embeddings = Embeddings()
    embeddings.load(str(out_dir))

    # Drop every old chunk id for files that changed or vanished. Old chunk counts
    # come from the manifest, so a file that shrank (fewer chunks now) has all its
    # stale trailing chunks removed too.
    del_ids: list[str] = []
    for fp in changed | deleted:
        n_old = manifest.get(fp, {}).get("n_chunks", 0)
        del_ids.extend(_chunk_id(fp, i) for i in range(n_old))
    if del_ids:
        embeddings.delete(del_ids)

    # Re-embed only the changed files; rebuild the manifest from the current set so
    # deleted files fall out of it.
    new_records: list[tuple] = []
    new_manifest: dict = {}
    for fp, doc in current.items():
        if fp in changed:
            doc_records = _records_for_doc(doc, cfg, repo_slug)
            new_records.extend(doc_records)
            n_chunks = len(doc_records)
        else:
            n_chunks = manifest[fp]["n_chunks"]  # unchanged → reuse count
        new_manifest[fp] = {
            "mtime": current_mtime[fp],
            "n_chunks": n_chunks,
            "source_type": doc.source_type,
        }
    if new_records:
        embeddings.upsert(new_records)

    _persist(cfg, out_dir, embeddings, result, new_manifest)

    return {
        "repo": repo_slug,
        "mode": "incremental",
        "documents": _counts(result.documents),
        "chunks": sum(e["n_chunks"] for e in new_manifest.values()),
        "changed": len(changed),
        "deleted": len(deleted),
        "upserted_chunks": len(new_records),
        "store": str(out_dir),
    }


def index_mtime(cfg: Config, repo_slug: str) -> float:
    """Build time of a repo's index (mtime of its saved embeddings), or 0.0 if none."""
    emb = store_path(cfg, repo_slug) / "embeddings"
    try:
        return emb.stat().st_mtime
    except OSError:
        return 0.0


def is_stale(cfg: Config, repo_slug: str) -> bool:
    """True if the repo has no index yet, or any indexable source file is newer
    than the last build.

    A cheap stat-only walk (no embedding, no file reads), so it is safe to run
    over every repo to decide which actually need re-indexing. Short-circuits on
    the first newer file.
    """
    built = index_mtime(cfg, repo_slug)
    if built == 0.0:
        return True  # never indexed
    # Late import: keep torch-free callers (e.g. the daemon-client path) light.
    from corpus_search.ingest import _iter_files, _matches_glob

    repo_root = cfg.repo_path(repo_slug)
    prune = set(cfg.walk.prune_dirs)
    for f in _iter_files(repo_root, prune):
        # The same set ingest_repo indexes: prose globs + PDFs + notebooks. (A
        # deduped PDF that changed can trigger a needless rebuild — conservative,
        # never a missed change.)
        if _matches_glob(f, cfg.walk.prose_globs) or f.suffix.lower() in (".pdf", ".ipynb"):
            try:
                if f.stat().st_mtime > built:
                    return True
            except OSError:
                continue
    return False


def load_index(cfg: Config, repo_slug: str) -> Embeddings:
    from txtai import Embeddings

    out_dir = store_path(cfg, repo_slug)
    # The DIRECTORY is not the index. A build that died partway leaves the directory holding a
    # `config.json` and no `embeddings`, which passed this check and then failed several frames
    # down in faiss with `could not open .../embeddings for reading` -- a message that names
    # neither the repo nor the remedy. Test for the file the loader actually reads.
    if not (out_dir / "embeddings").exists():
        raise FileNotFoundError(f"no index for repo {repo_slug!r} at {out_dir}; run `corpus-search index {repo_slug}` first")
    embeddings = Embeddings()
    embeddings.load(str(out_dir))
    return embeddings
