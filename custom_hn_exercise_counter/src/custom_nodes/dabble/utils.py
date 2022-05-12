from typing import List, Tuple

import cv2
import numpy as np


def map_bbox_to_image_coords(
    bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
    """First helper function to convert relative bounding box coordinates to
    absolute image coordinates.
    Bounding box coords ranges from 0 to 1
    where (0, 0) = image top-left, (1, 1) = image bottom-right.

    Args:
       bbox (List[float]): List of 4 floats x1, y1, x2, y2
       image_size (Tuple[int, int]): Width, Height of image

    Returns:
       List[int]: x1, y1, x2, y2 in integer image coords
    """
    width, height = image_size[0], image_size[1]
    x1, y1, x2, y2 = bbox
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    return int(x1), int(y1), int(x2), int(y2)


def map_keypoint_to_image_coords(
    keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
    """Second helper function to convert relative keypoint coordinates to
    absolute image coordinates.
    Keypoint coords ranges from 0 to 1
    where (0, 0) = image top-left, (1, 1) = image bottom-right.

    Args:
       bbox (List[float]): List of 2 floats x, y (relative)
       image_size (Tuple[int, int]): Width, Height of image

    Returns:
       List[int]: x, y in integer image coords
    """
    width, height = image_size[0], image_size[1]
    x, y = keypoint
    x *= width
    y *= height
    return int(x), int(y)


def draw_text(
    img: np.ndarray, text: str, org: Tuple[int, int], *args, **kwargs
) -> None:
    """Helper function to call opencv's drawing function,
    to improve code readability in node's run() method.

    Args:
        img (np.ndarray): The image to draw on.
        text (str): The text to put on the image.
        org (Tuple[int, int]): The (x, y) coordinates of the text.
        *args: Arguments to pass to cv2.putText().
        **kwargs: Keyword arguments to pass to cv2.putText().
    """

    cv2.putText(img, text, org, *args, **kwargs)
