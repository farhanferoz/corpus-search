# corpus-search

Local, hybrid (keyword + semantic) search over a research corpus. Per-repo
index, fully local embeddings — nothing leaves the machine.

Part of a Claude Code working setup: see
[farhanferoz/claude-code-setup](https://github.com/farhanferoz/claude-code-setup) for how it fits with the rest.

Retrieval is **hybrid BM25 + dense** and is near-instant. Cross-encoder
reranking is available but **off by default** (see "Reranking" below).

## Install

Needs a C/C++ compiler: `txtai[pipeline]` pulls in `llama-cpp-python`, which PyPI ships only as source
(Debian or Ubuntu: `sudo apt install build-essential`; macOS: `xcode-select --install`).

```
uv tool install --editable .
corpus-search --help
```

This puts `corpus-search` and `corpus-search-mcp` on your `PATH`. For development, `uv sync` creates a
virtual environment, and every command below then runs as `uv run corpus-search ...`.

List the repositories to index in a personal file outside this repository,
`~/.config/corpus-search/repos.toml` (or the file named by `CORPUS_SEARCH_REPOS`), as a `[repos]` table of
`slug = "/absolute/path"` lines. It is read on top of the shipped `corpus-search.toml`, which keeps the other
settings and names no repositories, so your project list never lands in this repository.

## Usage

```
corpus-search index <repo-slug>                # update one index (full build when none exists)
corpus-search index <repo-slug> --full         # force a full rebuild
corpus-search index-all                        # update every configured repo
corpus-search index-all --changed-only         # skip repos whose sources are unchanged
corpus-search index-all --skip-existing        # skip repos that already have an index
corpus-search query <repo-slug> "<question>"   # search one repo
corpus-search search-all "<question>"          # search every repo, merged by score
corpus-search query <repo-slug> "<question>" --k 5
```

Updates are incremental by default. Only files whose modification time advanced are re-embedded.
Use `--full` after changing the embedding model or the chunk settings. Every command takes
`--config <path>` to read another `corpus-search.toml`, and `query` and `search-all` take `--k`
(default 10).

`--changed-only` compares each repo's source-file mtimes against its last index
build and skips repos that are already current — so a routine refresh only
re-embeds what actually changed, not the whole corpus.

### Warm daemon (fast repeat queries)

Each `query` invocation otherwise cold-loads the embedder (~25s of torch import).
Start a warm daemon once and queries drop to ~0.3s:

```
uv run corpus-search daemon start     # loads the embedder once (~20s), then serves
uv run corpus-search daemon status
uv run corpus-search daemon stop
```

The daemon holds ~1.25 GB RAM while running and auto-exits after 20 min idle
(`--idle-timeout <seconds>`). You can start it explicitly, but you don't have to:
`query`/`search-all` **spin one up on demand** when none is running, so the first
search of a session pays the ~20–30 s model load and every search after is ~0.3 s.
Many CLI invocations and MCP sessions then share that one loaded copy. Concurrent
callers are race-safe — only one daemon is ever started. `daemon`/`status`
commands are thin; they don't import the ML stack.

### MCP server (for agent use)

```
uv run corpus-search-mcp
```

Exposes one tool, `search_corpus(repo, query, k)`, returning
`{file_path, heading, score, snippet}` per hit. It routes through the shared
daemon (spinning one up on demand), so the MCP process stays a thin client and
never loads its own model — N connected sessions cost one shared ~1.25 GB daemon,
not N. (For agents that can shell out, the `corpus-search` CLI is an even leaner
path — no tool schema, same daemon.)

## Configuration

`corpus-search.toml` holds every setting. Its values are the defaults.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[store]` | `dir` | `~/.local/share/corpus-search` | Root of the per-repo indexes |
| `[embed]` | `model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `[rerank]` | `enabled` | `false` | Cross-encoder reranking |
| `[rerank]` | `model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `[rerank]` | `top_k` | `30` | Hybrid candidates passed to the reranker |
| `[chunk]` | `window_tokens`, `overlap_tokens` | `512`, `64` | Fixed-window chunking of headless prose |
| `[dedup]` | `similarity_threshold` | `0.6` | Word-set overlap above which a PDF counts as a copy of a markdown file |
| `[dedup]` | `export_dirs` | `pdf`, `build`, `export`, `out`, `_output`, `dist` | Directories searched for a same-name PDF |
| `[walk]` | `prune_dirs`, `prose_globs` | see the file | Directories skipped and prose files indexed |
| `[repos]` | slug = path | empty | Repositories to index (see Install) |

The daemon option `--idle-timeout` defaults to 1200 seconds. `CORPUS_SEARCH_REPOS` names the personal
repository list, and `XDG_CONFIG_HOME` moves its default location.

## Reranking (opt-in)

Hybrid retrieval alone matched a cross-encoder's top-5 quality on the pilot
corpus, while a large reranker cost minutes per query on CPU — so reranking is
**disabled by default**. To enable it, set `[rerank] enabled = true` in
`corpus-search.toml` (default model is a small 22M MiniLM cross-encoder,
~4–7 s/query warm) or override per query with `--rerank` / `--no-rerank`.

## Notes

- Index store lives outside this repo at `~/.local/share/corpus-search/<slug>/`
  (`chmod 700` — embeddings are derived from potentially sensitive source
  material). Never synced or committed.
- PDF ingestion skips a PDF when a markdown file in the same directory (or a
  sibling `pdf/`/`build/`/`export/` dir) is a near-duplicate export of it —
  see `[dedup]` in `corpus-search.toml`.
- Notebooks are ingested via their markdown + code cell source, not outputs.
- Building an index is CPU-heavy (embedding every chunk): ~9 min for a
  ~220-doc / ~3,800-chunk repo. Querying is cheap; building is the cost.

## License

MIT — see [LICENSE](LICENSE).
