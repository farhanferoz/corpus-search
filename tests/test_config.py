"""The personal repository list is read from outside the repository and merged over the shipped config."""

from pathlib import Path

import pytest

from corpus_search.config import DEFAULT_CONFIG_PATH, load_config


def test_shipped_config_names_no_repositories_and_the_personal_file_adds_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    monkeypatch.delenv("CORPUS_SEARCH_REPOS", raising=False)
    assert load_config(DEFAULT_CONFIG_PATH).repos == {}
    personal = tmp_path / "repos.toml"
    personal.write_text('[repos]\n"notes" = "~/dev/notes"\n')
    monkeypatch.setenv("CORPUS_SEARCH_REPOS", str(personal))
    assert load_config(DEFAULT_CONFIG_PATH).repos == {"notes": Path("~/dev/notes").expanduser()}


def test_a_named_personal_file_that_is_missing_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_SEARCH_REPOS", str(tmp_path / "absent.toml"))
    with pytest.raises(FileNotFoundError, match="CORPUS_SEARCH_REPOS"):
        load_config(DEFAULT_CONFIG_PATH)
