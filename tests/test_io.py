import pytest

from mjecv.io import imread


def test_imread():
    image = imread("data/chess1.png")
    assert image.shape == (240, 320)


def test_imread_missing():
    with pytest.raises(FileNotFoundError):
        imread("missing_file.png")


def test_imread_invalid(tmp_path):
    file = tmp_path.joinpath("temp.png")
    file.touch()
    with pytest.raises(IOError):
        imread(file)
