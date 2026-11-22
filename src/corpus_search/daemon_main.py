"""Entrypoint for `python -m corpus_search.daemon_main` — runs the daemon in the foreground."""

from __future__ import annotations

import argparse

from corpus_search.config import load_config
from corpus_search.daemon import DEFAULT_IDLE_TIMEOUT, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT)
    args = parser.parse_args()
    run(load_config(), idle_timeout=args.idle_timeout)


if __name__ == "__main__":
    main()
