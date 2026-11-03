"""Shared capture guards: raw-pixel ROIs, unique IDs and new candidate files."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def validate_roi(roi, image_shape):
    """Require a convex polygon entirely inside an original-resolution image."""
    polygon = np.asarray(roi, np.float32)
    h, w = image_shape
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.all(np.isfinite(polygon)) or np.any(polygon < 0)
            or np.any(polygon[:, 0] >= w) or np.any(polygon[:, 1] >= h)
            or not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1):
        raise ValueError("ROI must be a nondegenerate convex polygon inside the raw image")
    return polygon


def add_capture_arguments(parser):
    parser.add_argument("--roi-file", help="JSON direct-board polygons: cam0/cam1; individual mode uses camera")
    parser.add_argument("--allow-full-frame", action="store_true",
                        help="Explicitly accept full-frame detection only when no reflected/other board is visible")


def load_capture_rois(args, keys):
    """Require explicit reflection control before opening camera hardware."""
    if args.roi_file:
        data = json.loads(Path(args.roi_file).read_text())
        if not isinstance(data, dict) or any(k not in data for k in keys):
            raise ValueError(f"ROI file requires polygons for {keys}")
        return {k: data[k] for k in keys}
    if not args.allow_full_frame:
        raise ValueError("supply --roi-file for the DIRECT board or explicitly --allow-full-frame for an isolated target")
    return {k: None for k in keys}


def unique_corners(corners, ids, corner_count):
    """Reject every occurrence of an ambiguous chess-corner ID, never pick first."""
    if corners is None or ids is None:
        return None, None
    pixels = np.asarray(corners).reshape(-1, 2)
    identifiers = np.asarray(ids).reshape(-1)
    if len(pixels) != len(identifiers):
        raise ValueError("corner/ID counts differ")
    values, counts = np.unique(identifiers, return_counts=True)
    unique = set(values[counts == 1])
    keep = np.array([i in unique and 0 <= i < corner_count and np.all(np.isfinite(p))
                     for i, p in zip(identifiers, pixels)], bool)
    if not keep.any():
        return None, None
    return pixels[keep].astype(np.float32).reshape(-1, 1, 2), identifiers[keep].astype(np.int32).reshape(-1, 1)


class CaptureDetector:
    """Mask before decoding without crop/rotation, then remove duplicate IDs."""

    def __init__(self, detector, board, roi=None):
        self.detector, self.roi = detector, roi
        self.corner_count = len(board.getChessboardCorners())

    def detectBoard(self, gray):
        image = gray
        polygon = None
        if self.roi is not None:
            polygon = validate_roi(self.roi, gray.shape[:2])
            mask = np.zeros(gray.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
            image = np.where(mask, gray, 255).astype(np.uint8)
        corners, ids, markers, marker_ids = self.detector.detectBoard(image)
        corners, ids = unique_corners(corners, ids, self.corner_count)
        if ids is not None and polygon is not None:
            keep = np.array([cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) >= 0
                             for p in corners.reshape(-1, 2)])
            corners, ids = (corners[keep], ids[keep]) if keep.any() else (None, None)
        return corners, ids, markers, marker_ids


def averaged_stereo_points(capture, board_points):
    """Build one correspondence per ID visible in both averaged camera maps."""
    ids0 = {cid for cam, cid in capture if cam == 0}
    ids1 = {cid for cam, cid in capture if cam == 1}
    common = sorted(ids0 & ids1)
    obj = np.asarray([board_points[cid] for cid in common], np.float32).reshape(-1, 3)
    p0 = np.asarray([capture[(0, cid)] for cid in common], np.float32).reshape(-1, 2)
    p1 = np.asarray([capture[(1, cid)] for cid in common], np.float32).reshape(-1, 2)
    return obj, p0, p1


def require_image_size(actual, expected, label):
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{label} size {actual} differs from calibration {expected}; no calibration saved")


def world_samples_ready(captures, target):
    """Both cameras must independently reach the requested frame count."""
    return target > 0 and all(len(captures.get(k, [])) >= target for k in ("cam0", "cam1"))


def new_candidate_path(path):
    """Preserve existing files, including active calibration paths."""
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new candidate output file")
    return path


def save_candidate_npz(path, **values):
    """Exclusive write: refuse nonfinite data and refuse replacing existing files."""
    for name, value in values.items():
        a = np.asarray(value)
        if np.issubdtype(a.dtype, np.number) and not np.all(np.isfinite(a)):
            raise ValueError(f"cannot save nonfinite calibration field {name}")
    path = new_candidate_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **values)
    return path
