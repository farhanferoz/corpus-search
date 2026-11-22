"""Load and resolve corpus-search.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "corpus-search.toml"
# The personal repository list lives outside the repository, so a published checkout names no one's projects.
REPOS_ENV = "CORPUS_SEARCH_REPOS"


def user_repos_path() -> Path:
    """The personal [repos] file: $CORPUS_SEARCH_REPOS, else $XDG_CONFIG_HOME/corpus-search/repos.toml."""
    if os.environ.get(REPOS_ENV):
        return Path(os.environ[REPOS_ENV]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "corpus-search" / "repos.toml"


def _repos(raw: dict) -> dict[str, Path]:
    """Repositories from the shipped config, then the personal file on top (a slug there wins)."""
    merged = dict(raw.get("repos", {}))
    personal = user_repos_path()
    if personal.is_file():
        merged.update(tomllib.loads(personal.read_text()).get("repos", {}))
    elif os.environ.get(REPOS_ENV):
        raise FileNotFoundError(f"{REPOS_ENV}={personal} does not exist; create it with a [repos] table or unset it.")
    return {slug: Path(p).expanduser() for slug, p in merged.items()}


@dataclass
class StoreConfig:
    dir: Path


@dataclass
class EmbedConfig:
    model: str


@dataclass
class RerankConfig:
    enabled: bool
    model: str
    top_k: int


@dataclass
class ChunkConfig:
    window_tokens: int
    overlap_tokens: int


@dataclass
class DedupConfig:
    similarity_threshold: float
    export_dirs: list[str]


@dataclass
class WalkConfig:
    prune_dirs: list[str]
    prose_globs: list[str]


@dataclass
class Config:
    store: StoreConfig
    embed: EmbedConfig
    rerank: RerankConfig
    chunk: ChunkConfig
    dedup: DedupConfig
    walk: WalkConfig
    repos: dict[str, Path] = field(default_factory=dict)

    def repo_path(self, slug: str) -> Path:
        try:
            return self.repos[slug]
        except KeyError:
            known = ", ".join(sorted(self.repos)) or "(none configured)"
            raise KeyError(f"unknown repo slug {slug!r}; known repos: {known}") from None


def load_config(path: Path | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    raw = tomllib.loads(path.read_text())

    return Config(
        store=StoreConfig(dir=Path(raw["store"]["dir"]).expanduser()),
        embed=EmbedConfig(model=raw["embed"]["model"]),
        rerank=RerankConfig(
            enabled=raw["rerank"].get("enabled", False),
            model=raw["rerank"]["model"],
            top_k=raw["rerank"].get("top_k", 30),
        ),
        chunk=ChunkConfig(
            window_tokens=raw["chunk"]["window_tokens"],
            overlap_tokens=raw["chunk"]["overlap_tokens"],
        ),
        dedup=DedupConfig(
            similarity_threshold=raw["dedup"]["similarity_threshold"],
            export_dirs=raw["dedup"]["export_dirs"],
        ),
        walk=WalkConfig(
            prune_dirs=raw["walk"]["prune_dirs"],
            prose_globs=raw["walk"]["prose_globs"],
        ),
        repos=_repos(raw),
    )
