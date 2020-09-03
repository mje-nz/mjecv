from typing import Tuple, Union, overload

import numpy as np

__all__ = ["Point2", "Size", "Rect"]


def _generate_math_operator(operator_name):
    """Helper function for _Vector operators."""

    def operator(self, other):
        return type(self)(self._operator(operator_name, other))

    return operator


class _Vector:
    def __init__(self, *args):
        if (
            len(args) == 1
            and hasattr(args[0], "__len__")
            and len(args[0]) == self._length
        ):
            self._value = np.array(args[0])
        elif len(args) == self._length:
            self._value = np.array(args)
        else:
            raise ValueError(f"{args} is not a valid {type(self).__name__}")

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self._value))

    def __getattr__(self, name):
        if name in self._aliases:
            return self._value[self._aliases.index(name)]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._aliases:
            self._value[self._aliases.index(name)] = value
        elif name == "_value":
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(name)

    def __getitem__(self, index):
        return self._value[index]

    def __setitem__(self, index, value):
        self._value[index] = value

    def __len__(self):
        return len(self._value)

    def __eq__(self, other):
        # TODO: comparisons against compatible values
        # TODO: more comparison operators
        if not type(other) == type(self):
            return NotImplemented
        return np.all(self._value == other._value)

    def _operator(self, operator, other):
        """Delegate an operator to the underlying ndarray."""
        if type(other) == type(self):
            other = other._value
        return getattr(np.ndarray, operator)(self._value, other)

    # Delegate math operators to the underlying ndarray via _operator.  Anything
    # cleverer than this is too clever for mypy or PyCharm to recognise.
    __add__ = _generate_math_operator("__add__")
    __sub__ = _generate_math_operator("__sub__")
    __mul__ = _generate_math_operator("__mul__")
    __rmul__ = _generate_math_operator("__rmul__")
    __truediv__ = _generate_math_operator("__truediv__")

    # TODO: iterable, norm


Point2Like = Union["Point2", np.ndarray, Tuple[float, float]]


class Point2(_Vector):
    """2D point class modelled on OpenCV's Point2"""

    _length = 2
    _aliases = ("x", "y")

    @overload
    def __init__(self, vector: Point2Like):
        pass

    @overload
    def __init__(self, x: float, y: float):
        pass

    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return str(tuple(self._value))


SizeLike = Union["Size", np.ndarray, Tuple[float, float]]


class Size(_Vector):

    _length = 2
    _aliases = ("width", "height")

    @overload
    def __init__(self, vector: SizeLike):
        pass

    @overload
    def __init__(self, width: float, height: float):
        pass

    def __init__(self, *args):
        super().__init__(*args)

    # TODO: area


class Rect:

    _aliases = ("x", "y", "width", "height")

    @overload
    def __init__(self, origin: Point2Like, size: SizeLike):
        pass

    @overload
    def __init__(self, x: float, y: float, width: float, height: float):
        pass

    def __init__(self, *args):
        if len(args) == 2:
            self.origin = Point2(args[0])
            self.size = Size(args[1])
            # TODO: Rect(point1, point2)
        elif len(args) == 4:
            self.origin = Point2(args[0], args[1])
            self.size = Size(args[2], args[3])
        else:
            raise ValueError(f"{args} is not a valid rect")

    def __repr__(self):
        return f"Rect({self.origin!r}, {self.size!r})"

    def __getitem__(self, index):
        if index in (0, 1):
            return self.origin[index]
        elif index in (2, 3):
            return self.size[index - 2]
        else:
            raise IndexError()

    def __getattr__(self, name):
        if name in self._aliases:
            return self[self._aliases.index(name)]
        else:
            raise AttributeError(name)

    # TODO: setitem and setattr

    def __eq__(self, other):
        # TODO: comparisons against compatible values
        if not type(other) == type(self):
            return NotImplemented
        return self.origin == other.origin and self.size == other.size

    def __contains__(self, point):
        return self.contains(point)

    def contains(self, point):
        point = Point2(point)
        return (self.x <= point.x <= self.x + self.width) and (
            self.y <= point.y <= self.y + self.height
        )

    @property
    def x_min(self):
        return self.x

    @property
    def x_max(self):
        return self.x + self.width

    @property
    def y_min(self):
        return self.y

    @property
    def y_max(self):
        return self.y + self.height

    # TODO: area, topleft, bottomright, iterable
