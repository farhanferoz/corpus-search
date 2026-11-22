"""Walk a repo, collect prose/PDF/notebook documents, dedup PDFs against markdown twins."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf
import nbformat

from corpus_search.config import Config

_WORD_RE = re.compile(r"\w+")


@dataclass
class Document:
    file_path: Path
    source_type: str  # "prose" | "pdf" | "notebook"
    text: str
    pages: list[str] | None = None  # populated for source_type == "pdf"


@dataclass
class IngestResult:
    documents: list[Document]
    skipped_pdfs: list[dict]


def _iter_files(repo_root: Path, prune_dirs: set[str]) -> list[Path]:
    files = []
    stack = [repo_root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in prune_dirs:
                    stack.append(entry)
            else:
                files.append(entry)
    return files


def _matches_glob(path: Path, globs: list[str]) -> bool:
    return any(path.match(g) for g in globs)


def _wordset(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pdf_text(pdf_path: Path) -> tuple[str, list[str]]:
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(pages), pages


def _notebook_text(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type in ("markdown", "code") and cell.source.strip():
            parts.append(cell.source)
    return "\n\n".join(parts)


def _find_markdown_twin(pdf_path: Path, md_by_dir: dict[Path, set[str]], export_dirs: set[str]) -> Path | None:
    stem = pdf_path.stem
    same_dir = pdf_path.parent
    if stem in md_by_dir.get(same_dir, set()):
        return same_dir / f"{stem}.md"

    if same_dir.name in export_dirs:
        parent_dir = same_dir.parent
        if stem in md_by_dir.get(parent_dir, set()):
            return parent_dir / f"{stem}.md"
    return None


def ingest_repo(repo_root: Path, cfg: Config) -> IngestResult:
    prune_dirs = set(cfg.walk.prune_dirs)
    export_dirs = set(cfg.dedup.export_dirs)
    all_files = _iter_files(repo_root, prune_dirs)

    prose_paths = [f for f in all_files if _matches_glob(f, cfg.walk.prose_globs)]
    pdf_paths = [f for f in all_files if f.suffix.lower() == ".pdf"]
    notebook_paths = [f for f in all_files if f.suffix.lower() == ".ipynb"]

    md_by_dir: dict[Path, set[str]] = {}
    for md in (f for f in prose_paths if f.suffix.lower() == ".md"):
        md_by_dir.setdefault(md.parent, set()).add(md.stem)

    documents: list[Document] = []
    skipped: list[dict] = []

    for path in prose_paths:
        documents.append(Document(file_path=path, source_type="prose", text=path.read_text(errors="replace")))

    for path in notebook_paths:
        try:
            text = _notebook_text(path)
        except Exception as e:  # malformed notebook
            skipped.append({"file": str(path), "reason": f"notebook_parse_error: {e}"})
            continue
        if text.strip():
            documents.append(Document(file_path=path, source_type="notebook", text=text))

    for path in pdf_paths:
        twin = _find_markdown_twin(path, md_by_dir, export_dirs)
        if twin is not None:
            try:
                pdf_full_text, pages = _pdf_text(path)
            except Exception as e:
                skipped.append({"file": str(path), "reason": f"pdf_parse_error: {e}"})
                continue
            md_text = twin.read_text(errors="replace")
            sim = _jaccard(_wordset(pdf_full_text), _wordset(md_text))
            if sim >= cfg.dedup.similarity_threshold:
                skipped.append(
                    {
                        "pdf": str(path),
                        "matched_md": str(twin),
                        "similarity": round(sim, 3),
                        "reason": "markdown_twin",
                    }
                )
                continue
            documents.append(Document(file_path=path, source_type="pdf", text=pdf_full_text, pages=pages))
        else:
            try:
                pdf_full_text, pages = _pdf_text(path)
            except Exception as e:
                skipped.append({"file": str(path), "reason": f"pdf_parse_error: {e}"})
                continue
            documents.append(Document(file_path=path, source_type="pdf", text=pdf_full_text, pages=pages))

    return IngestResult(documents=documents, skipped_pdfs=skipped)


def write_skipped_log(skipped: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for entry in skipped:
            f.write(json.dumps(entry) + "\n")
