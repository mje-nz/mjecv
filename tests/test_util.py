import pytest

from mjecv.util import ensure_open

example_text = "example text"


def test_ensure_open_string():
    assert ensure_open(example_text) == example_text


def test_ensure_open_string_strict():
    with pytest.raises(FileNotFoundError):
        ensure_open(example_text, accept_string=False)


# TODO: factor out


@pytest.fixture
def example_text_path(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text(example_text)
    return path


def test_ensure_open_path(example_text_path):
    assert ensure_open(example_text_path).read() == example_text


def test_ensure_open_filename(example_text_path):
    assert ensure_open(str(example_text_path)).read() == example_text


def test_ensure_open_file(example_text_path):
    assert ensure_open(example_text_path.open()).read() == example_text
