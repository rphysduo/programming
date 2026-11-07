"""Shared raw-pixel ray and nominal board-plane geometry."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def pixel_ray_in_board(
    pixel: Sequence[float], K: np.ndarray, D: np.ndarray,
    R_board_to_cam: np.ndarray, t_board_to_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera centre and unit pixel-ray direction in board coordinates."""
    normalized = cv2.undistortPoints(
        np.asarray(pixel, np.float64).reshape(1, 1, 2), K, D
    ).reshape(2)
    direction_cam = np.array([normalized[0], normalized[1], 1.0], np.float64)
    R = np.asarray(R_board_to_cam, np.float64)
    t = np.asarray(t_board_to_cam, np.float64).reshape(3)
    centre_board = -R.T @ t
    direction_board = R.T @ direction_cam
    direction_board /= np.linalg.norm(direction_board)
    return centre_board, direction_board


def intersect_ray_with_board_plane(
    centre_board: np.ndarray, direction_board: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Intersect a board-frame ray with ``Z=0`` without altering stereo data."""
    denominator = float(direction_board[2])
    if not np.isfinite(denominator) or abs(denominator) < 1e-6:
        raise ValueError("camera ray is parallel or nearly parallel to the board plane")
    distance = -float(centre_board[2]) / denominator
    if distance <= 0:
        raise ValueError("board plane is behind the camera for this pixel")
    point = np.asarray(centre_board, np.float64) + distance * np.asarray(direction_board, np.float64)
    point[2] = 0.0
    return point, distance
