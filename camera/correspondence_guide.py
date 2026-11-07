"""Read-only cross-camera guides in original, distorted image coordinates.

The stereo convention is X1 = R @ X0 + T. An epipolar curve is a locus,
not a unique correspondence. The optional unique point explicitly assumes
the saved world-board plane; it never constrains stereo reconstruction.
"""
from __future__ import annotations

import cv2
import numpy as np

if __package__:
    from .camera_geometry import pixel_ray_in_board, intersect_ray_with_board_plane
else:
    from camera_geometry import pixel_ray_in_board, intersect_ray_with_board_plane

def _project(points, K, D):
    return cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 1, 3),
                             np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def _inside(point, size):
    return bool(np.all(np.isfinite(point)) and
                0 <= point[0] <= size[0] - 1 and 0 <= point[1] <= size[1] - 1)


def _curve_segments(line, K, D, size):
    """Sample an ideal epipolar line and distort it back to the raw image.

    Undistorted boundary samples bound the visible line. Sampling at roughly
    one ideal pixel keeps the displayed polyline smooth for these calibrated
    pinhole lenses. Out-of-frame/nonfinite runs are split, never bridged.
    """
    width, height = size
    xs, ys = np.linspace(0, width - 1, 65), np.linspace(0, height - 1, 65)
    border = np.concatenate((np.column_stack((xs, np.zeros_like(xs))),
                             np.column_stack((xs, np.full_like(xs, height - 1))),
                             np.column_stack((np.zeros_like(ys), ys)),
                             np.column_stack((np.full_like(ys, width - 1), ys))))
    ideal = cv2.undistortPoints(border.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    if not np.all(np.isfinite(ideal)):
        return []
    lower, upper = ideal.min(axis=0), ideal.max(axis=0)
    margin = np.maximum((upper - lower) * 0.05, 1e-6)
    lower, upper = lower - margin, upper + margin
    a, b, c = line
    count = min(8193, max(513, 2 * max(width, height) + 1))
    if abs(b) >= abs(a):
        x = np.linspace(lower[0], upper[0], count)
        y = -(a * x + c) / b
    else:
        y = np.linspace(lower[1], upper[1], count)
        x = -(b * y + c) / a
    raw = _project(np.column_stack((x, y, np.ones(count))), K, D)
    segments, run = [], []
    for point in raw:
        if _inside(point, size):
            run.append(point.tolist())
        else:
            if len(run) > 1:
                segments.append(run)
            run = []
    if len(run) > 1:
        segments.append(run)
    return segments


def predict_correspondence(cal, source_camera, source_px, *, board_plane=False):
    """Return a bidirectional epipolar curve and optional board-plane marker.

    ``source_px`` is a raw full-resolution pixel. ``board_plane=True`` asks
    for the independent saved-world-pose prediction at board Z=0; this is
    unsuitable for arbitrary off-plane objects. All returned pixels are raw.
    No calibration, correspondence, triangulation or validity state is changed.
    Invalid requests raise ValueError; geometric degeneracies return warnings.
    """
    if isinstance(source_camera, bool) or source_camera not in (0, 1):
        raise ValueError("source_camera must be 0 or 1")
    if not isinstance(board_plane, bool):
        raise ValueError("board_plane must be true or false")
    try:
        point = np.asarray(source_px, np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_px must contain two finite raw pixel coordinates") from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_px must contain two finite raw pixel coordinates")
    source_camera = int(source_camera)
    target = 1 - source_camera
    cameras = ((cal.K0, cal.D0, cal.size0), (cal.K1, cal.D1, cal.size1))
    Ks, Ds, size = cameras[source_camera]
    Kt, Dt, target_size = cameras[target]
    if not _inside(point, size):
        raise ValueError("source_px is outside the calibrated raw image")
    uv = cv2.undistortPoints(point.reshape(1, 1, 2), Ks, Ds).reshape(2)
    ray = np.r_[uv, 1.]
    if not np.all(np.isfinite(ray)):
        raise ValueError("source pixel could not be undistorted")
    tx, ty, tz = cal.T
    essential = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]]) @ cal.R
    line = (essential if source_camera == 0 else essential.T) @ ray
    norm = np.linalg.norm(line[:2])
    warnings = []
    segments = []
    if not np.isfinite(norm) or norm < 1e-12:
        warnings.append("Epipolar curve is undefined at this epipole or zero baseline.")
    else:
        segments = _curve_segments(line / norm, Kt, Dt, target_size)
        if not segments:
            warnings.append("The epipolar curve does not intersect the other image.")
    result = {"source_camera": source_camera, "target_camera": target,
              "source_px": point.tolist(), "curve_segments_px": segments,
              "board_plane": None, "warnings": warnings,
              "coordinate_convention": "raw full-resolution pixels; x right, y down",
              "note": "A curve constrains a match; it does not determine depth. "
                      "Click the same physical feature, not just any point on the curve."}
    if board_plane:
        # The independent world poses deliberately remain independent of R/T.
        # Their prediction can disagree with the stereo curve: display both.
        Rs = getattr(cal, f"R_board_to_cam{source_camera}")
        ts = getattr(cal, f"t_board_to_cam{source_camera}")
        Rt = getattr(cal, f"R_board_to_cam{target}")
        tt = getattr(cal, f"t_board_to_cam{target}")
        center, direction = pixel_ray_in_board(point, Ks, Ds, Rs, ts)
        try:
            board, _distance = intersect_ray_with_board_plane(center, direction)
        except ValueError as exc:
            warnings.append(str(exc))
        else:
            target_point = Rt @ board + tt
            if not np.all(np.isfinite(target_point)) or target_point[2] <= 0:
                warnings.append("Board-plane intersection is behind a camera.")
            else:
                pixel = _project(target_point, Kt, Dt)[0]
                if np.all(np.isfinite(pixel)):
                    result["board_plane"] = {
                        "target_px": pixel.tolist(), "board_xyz_mm": board.tolist(),
                        "inside_target_image": _inside(pixel, target_size),
                        "assumption": "Point is on saved world-board Z=0; independent world poses."}
                    if not _inside(pixel, target_size):
                        warnings.append("Board-plane prediction is outside the other image.")
                else:
                    warnings.append("Board-plane projection is nonfinite.")
    return result
