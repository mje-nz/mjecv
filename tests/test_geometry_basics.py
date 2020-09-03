import numpy as np
import pytest

from mjecv.geometry import Point2, Rect, Size


def test_Point2_create():
    point = Point2(1, 2)
    assert point[0] == point.x == 1
    assert point[1] == point.y == 2


def test_Point2_copy():
    point = Point2(Point2(1, 2))
    assert point[0] == point.x == 1
    assert point[1] == point.y == 2


def test_Point2_create_invalid():
    with pytest.raises(ValueError):
        Point2()
    with pytest.raises(ValueError):
        Point2(1)
    with pytest.raises(ValueError):
        Point2([1])
    with pytest.raises(ValueError):
        Point2(1, 2, 3)
    with pytest.raises(ValueError):
        Point2([1, 2, 3])


def test_Point2_repr():
    assert repr(Point2(1, 2)) == "Point2(1, 2)"
    assert repr(Point2(1.0, 2.0)) == "Point2(1.0, 2.0)"
    assert str(Point2(1, 2)) == "(1, 2)"
    assert str(Point2(1.0, 2.0)) == "(1.0, 2.0)"


def test_Point2_attrs():
    point = Point2(1, 2)
    point[0] = 3
    point[1] = 4
    assert point == Point2(3, 4)
    point.x += 1
    point.y += 2
    assert point == Point2(4, 6)
    point.x *= 2
    point.y *= 3
    assert point == Point2(8, 18)


def test_Point2_attrs_invalid():
    with pytest.raises(AttributeError):
        _ = Point2(1, 2).invalid_attribute
    with pytest.raises(AttributeError):
        Point2(1, 2).invalid_attribute = 1


def test_Point2_equality():
    point = Point2(1, 2)
    assert Point2(1, 2) == point
    assert Point2((1, 2)) == point
    assert Point2([1, 2]) == point
    assert Point2(np.array((1, 2))) == point
    assert not Point2(1, 2) != point
    assert not Point2(3, 4) == point
    assert Point2(3, 4) != point


def test_Point2_add():
    assert Point2(1, 2) + Point2(3, 4) == Point2(4, 6)
    assert Point2(1, 2) + 1 == Point2(2, 3)

    point = Point2(1, 2)
    point += Point2(3, 4)
    assert point == Point2(4, 6)

    point = Point2(1, 2)
    point += 1
    assert point == Point2(2, 3)


def test_Point2_sub():
    assert Point2(1, 2) - Point2(3, 4) == Point2(-2, -2)
    assert Point2(1, 2) - 1 == Point2(0, 1)

    point = Point2(1, 2)
    point -= Point2(3, 4)
    assert point == Point2(-2, -2)

    point = Point2(1, 2)
    point -= 1
    assert point == Point2(0, 1)


def test_Point2_mul():
    assert Point2(1, 2) * 2 == 2 * Point2(1, 2) == Point2(2, 4)

    point = Point2(1, 2)
    point *= 2
    assert point == Point2(2, 4)


def test_Point2_div():
    assert Point2(1, 2) / 2 == Point2(0.5, 1)

    point = Point2(1, 2)
    point /= 2
    assert point == Point2(0.5, 1)


def test_Size_create():
    size = Size(1, 2)
    assert size[0] == size.width == 1
    assert size[1] == size.height == 2


def test_Size_repr():
    assert repr(Size(1, 2)) == "Size(1, 2)"
    assert repr(Size(1.0, 2.0)) == "Size(1.0, 2.0)"
    assert str(Size(1, 2)) == "Size(1, 2)"
    assert str(Size(1.0, 2.0)) == "Size(1.0, 2.0)"


def test_Size_add():
    res = Size(1, 2) + Size(3, 4)
    assert res == Size(4, 6)
    assert type(res) == Size


def test_Rect_create():
    rect = Rect(1, 2, 3, 4)
    assert rect[0] == rect.x == 1
    assert rect[1] == rect.y == 2
    assert rect[2] == rect.width == 3
    assert rect[3] == rect.height == 4

    assert rect == Rect(1, 2, 3, 4)
    assert rect == Rect((1, 2), (3, 4))


@pytest.mark.parametrize(
    "pt",
    (
        # Corners
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        # Inside points
        (0.5, 0.5),
        Point2(0.5, 0.5),
    ),
)
def test_Rect_contains(pt):
    rect = Rect(0, 0, 1, 1)
    assert rect.contains(pt)
    assert pt in rect


@pytest.mark.parametrize(
    "pt",
    (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (1, -1),
        (0, -1),
    ),
)
def test_Rect_not_contains(pt):
    rect = Rect(0, 0, 1, 1)
    assert not rect.contains(pt)
    assert pt not in rect
