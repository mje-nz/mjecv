from typing import Tuple

import cv2
import numpy as np


def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return 1


def validate_colour(image, colour):
    if len(image.shape) > 2 and safe_len(colour) != image.shape[2]:
        raise ValueError(
            "Invalid colour for {}-channel image: {}".format(image.shape[2], colour)
        )
    if len(image.shape) == 2 and safe_len(colour) > 1:
        raise ValueError("Invalid colour for 1-channel image: {}".format(colour))


def draw_circle(
    image: np.ndarray,
    centre: Tuple[float, float],
    radius: float,
    colour,
    thickness: int = 1,
    fill=False,
):
    """Draw a circle onto an image.

    Args:
        image: The image to draw onto.
        centre: (x, y) coords of circle centre.
        radius: Circle radius.
        colour: Line or fill colour, as a tuple with a value for each channel.
        thickness: Line thickness (ignored if fill is True).
        fill: Draw a filled circle instead.
    """
    if fill:
        thickness = -1
    else:
        assert thickness > 0
    validate_colour(image, colour)

    shift = 4
    x, y = centre
    x_fixed = int(x * 2 ** shift)
    y_fixed = int(y * 2 ** shift)
    radius_fixed = int(radius * 2 ** shift)
    cv2.circle(
        image,
        (x_fixed, y_fixed),
        radius=radius_fixed,
        color=colour,
        thickness=int(thickness),
        shift=shift,
        lineType=cv2.LINE_AA,
    )


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[float, float],
    colour,
    thickness: int = 1,
    font_type: str = "sans",
    scale: float = 1,
    align: str = "lower left",
    image_origin_at_bottom=False,
):
    """Draw text onto an image.

    Args:
        image: The image to draw onto.
        text: The text to draw.
        origin: (x, y) coords of the origin of of the text (see `align`).
        colour: Text colour, as a tuple with a value for each channel.
        thickness: Line thickness.
        font_type: "sans" or "serif".
        scale: Scale factor for font (on top of font's base height).
        align: Which corner is the origin of the text
            ("top/center/bottom left/centre/right").
        image_origin_at_bottom: Whether the image's origin is the bottom left or the top
        left.
    """
    if font_type == "sans":
        font = cv2.FONT_HERSHEY_DUPLEX
    elif font_type == "serif":
        font = cv2.FONT_HERSHEY_TRIPLEX
    else:
        raise ValueError("Unknown font type {}".format(font_type))
    validate_colour(image, colour)
    x, y = int(origin[0]), int(origin[1])
    thickness = int(thickness)

    vertical_align, horizontal_align = align.split()
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    assert vertical_align in ("top", "centre", "bottom")
    delta_y = 0
    if vertical_align == "top":
        delta_y = height
    if vertical_align == "centre":
        delta_y = height // 2
    if image_origin_at_bottom:
        y -= delta_y
    else:
        y += delta_y
    assert horizontal_align in ("left", "centre", "right")
    if horizontal_align == "centre":
        x -= width // 2
    if horizontal_align == "right":
        x -= width

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
        image_origin_at_bottom,
    )

    # TODO: support multiple lines


# TODO: add tests
