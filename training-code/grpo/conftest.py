import pytest

from testing.judge import private_rows, write_workspace


@pytest.fixture
def private():
    return private_rows()


@pytest.fixture
def workspace(tmp_path):
    return write_workspace(tmp_path / "workspace")
