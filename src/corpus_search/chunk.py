"""Split ingested documents into indexable chunks with heading/page metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from corpus_search.config import ChunkConfig
from corpus_search.ingest import Document

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RST_UNDERLINE_RE = re.compile(r"^([=\-~^\"*+#:.'`])\1{2,}\s*$")


@dataclass
class Chunk:
    text: str
    heading: str | None
    chunk_index: int


def _window(text: str, window_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= window_tokens:
        return [text]
    step = max(window_tokens - overlap_tokens, 1)
    windows = []
    start = 0
    while start < len(words):
        windows.append(" ".join(words[start : start + window_tokens]))
        if start + window_tokens >= len(words):
            break
        start += step
    return windows


def _split_markdown(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        m = _MD_HEADING_RE.match(line)
        if m:
            if lines:
                sections.append((heading, "\n".join(lines)))
            heading = m.group(2).strip()
            lines = [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines)))
    return sections or [(None, text)]


def _split_rst(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        title = lines[i].strip()
        underline = lines[i + 1].strip() if i + 1 < n else ""
        if title and _RST_UNDERLINE_RE.match(underline) and len(underline) >= max(3, len(title) - 2):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading = title
            buf = [lines[i], lines[i + 1]]
            i += 2
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections or [(None, text)]


def _split_sections(text: str, suffix: str) -> list[tuple[str | None, str]]:
    if suffix == ".md":
        return _split_markdown(text)
    if suffix == ".rst":
        return _split_rst(text)
    return [(None, text)]


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    if doc.source_type == "pdf":
        chunks = []
        for page_num, page_text in enumerate(doc.pages or [], start=1):
            if page_text.strip():
                chunks.append(Chunk(text=page_text, heading=f"page {page_num}", chunk_index=page_num - 1))
        return chunks

    suffix = doc.file_path.suffix.lower() if doc.source_type == "prose" else ".md"
    sections = _split_sections(doc.text, suffix)

    chunks: list[Chunk] = []
    idx = 0
    for heading, section_text in sections:
        for window_text in _window(section_text, cfg.window_tokens, cfg.overlap_tokens):
            chunks.append(Chunk(text=window_text, heading=heading, chunk_index=idx))
            idx += 1
    return chunks
