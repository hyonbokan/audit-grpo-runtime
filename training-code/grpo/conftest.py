import pytest

from testing.judge import private_row, write_corpus


@pytest.fixture
def row():
    return private_row()


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "corpus")
