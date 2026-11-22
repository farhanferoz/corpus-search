import os

# Every entry point loads its embedder through txtai (the CLI's index builds and cold queries, the MCP
# server's last-resort load, the warm daemon), and txtai puts it on the GPU whenever one exists (`gpu`
# defaults to True). On the shared DGX that is GPU work outside slurm and the GPU lease: the resident
# daemon held ~1.9 GB of the card for 20 idle minutes and made the GPU submission gate refuse real jobs
# (2026-10-15). On the CPU bge-small indexed 19 chunks/s with the machine busy (2026-10-18): a full
# rebuild of the largest repo, 13,072 chunks, takes about 12 minutes, an incremental update re-embeds
# only the changed files, and a query is one short embedding. So the package never sees the GPU. Set
# on package import, before anything loads torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from corpus_search.cli import main  # noqa: E402

__all__ = ["main"]
