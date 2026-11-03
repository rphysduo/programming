"""Validate stereo geometry against known ChArUco board coordinates.

The diagnostic deliberately keeps three estimates separate for every matched
corner:

1. the corner coordinate supplied by the ChArUco board definition;
2. each camera ray intersected independently with the board plane; and
3. an unconstrained stereo triangulation, transformed back to board coordinates.

The board plane is a nominal reference, never a correction to stereo output.
Use a user-inspected direct-board ROI: a coherent reflection can also fit a
homography. Duplicate IDs and RANSAC exclusions are reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from charuco_common import add_board_argument, get_board
from validate_stereo_3d import triangulate_point


@dataclass(frozen=True)
class StereoBoardCalibration:
    """Parameters needed to compare stereo and board-frame geometry."""

    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R_board_to_cam0: np.ndarray
    t_board_to_cam0: np.ndarray
    R_board_to_cam1: np.ndarray
    t_board_to_cam1: np.ndarray
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("K0", "K1", "R", "R_board_to_cam0", "R_board_to_cam1"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (3, 3) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite (3, 3) array")
            object.__setattr__(self, name, array)
        for name in ("T", "t_board_to_cam0", "t_board_to_cam1"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, array)
        for name in ("D0", "D1"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if not array.size or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite distortion coefficients")
            object.__setattr__(self, name, array)
        for K in (self.K0, self.K1):
            if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
                raise ValueError("intrinsics must have positive focal lengths and K[2]=[0,0,1]")
        if np.linalg.norm(self.T) == 0:
            raise ValueError("stereo baseline must be nonzero")
        for name in ("R", "R_board_to_cam0", "R_board_to_cam1"):
            rotation = getattr(self, name)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) \
                    or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError(f"{name} must be a proper rotation matrix")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must be a positive (width, height) pair")
        object.__setattr__(self, "image_size", tuple(int(v) for v in self.image_size))

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def F(self) -> np.ndarray:
        """Fundamental matrix derived from the exact R/T used to triangulate."""
        x, y, z = self.T
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.linalg.inv(self.K1).T @ cross @ self.R @ np.linalg.inv(self.K0)

    @classmethod
    def from_files(
        cls,
        param0: str | Path,
        param1: str | Path,
        stereo: str | Path,
        tank_alignment: str | Path,
    ) -> "StereoBoardCalibration":
        """Load and validate the four calibration artifacts."""
        with np.load(param0) as intr0, np.load(param1) as intr1, \
                np.load(stereo) as stereo_data, np.load(tank_alignment) as tank:
            size0 = tuple(int(v) for v in np.asarray(intr0["image_size"]).reshape(-1))
            size1 = tuple(int(v) for v in np.asarray(intr1["image_size"]).reshape(-1))
            if size0 != size1:
                raise ValueError(
                    f"intrinsic image sizes disagree: cam0={size0}, cam1={size1}")
            if tuple(tank["image_size"].reshape(-1)) != size0:
                raise ValueError("tank/intrinsic image sizes disagree")
            for i, intr in enumerate((intr0, intr1)):
                for key, stereo_key in (("camera_matrix", f"K{i}"), ("dist_coeffs", f"D{i}")):
                    a, b = intr[key].reshape(-1), stereo_data[stereo_key].reshape(-1)
                    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9):
                        raise ValueError(f"cam{i} {key} differs from stereo {stereo_key}; choose a consistent file set")
            return cls(
                K0=np.asarray(intr0["camera_matrix"], np.float64),
                D0=np.asarray(intr0["dist_coeffs"], np.float64),
                K1=np.asarray(intr1["camera_matrix"], np.float64),
                D1=np.asarray(intr1["dist_coeffs"], np.float64),
                R=np.asarray(stereo_data["R"], np.float64),
                T=np.asarray(stereo_data["T"], np.float64).reshape(3),
                R_board_to_cam0=np.asarray(tank["R_board_to_cam0"], np.float64),
                t_board_to_cam0=np.asarray(tank["t_board_to_cam0"], np.float64).reshape(3),
                R_board_to_cam1=np.asarray(tank["R_board_to_cam1"], np.float64),
                t_board_to_cam1=np.asarray(tank["t_board_to_cam1"], np.float64).reshape(3),
                image_size=size0,
            )


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Explicit demonstration thresholds used only to classify diagnostics."""

    plane_error_mm: float = 5.0
    stereo_error_mm: float = 5.0
    rotation_error_deg: float = 1.0
    translation_error_mm: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.plane_error_mm,
            self.stereo_error_mm,
            self.rotation_error_deg,
            self.translation_error_mm,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("all diagnostic thresholds must be positive and finite")


def _as_point_map(
    corners: np.ndarray,
    ids: np.ndarray,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Return unique ID -> pixel mappings and IDs excluded as duplicates.

    A duplicated ChArUco ID can indicate that a reflected copy and the physical
    board were both decoded.  There is no safe automatic correspondence in that
    case, so the ambiguous ID is omitted.
    """
    if ids is None or corners is None:
        return {}, []
    grouped: dict[int, list[np.ndarray]] = {}
    for corner, corner_id in zip(
        np.asarray(corners, np.float64).reshape(-1, 2),
        np.asarray(ids).reshape(-1),
    ):
        grouped.setdefault(int(corner_id), []).append(corner)
    duplicates = sorted(corner_id for corner_id, points in grouped.items() if len(points) != 1)
    points = {
        corner_id: candidates[0]
        for corner_id, candidates in grouped.items()
        if len(candidates) == 1
    }
    return points, duplicates


def detect_charuco_points(
    image: np.ndarray,
    detector,
    roi: Sequence[Sequence[float]] | None = None,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Mask reflections BEFORE decoding; retain original full-resolution pixels."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygon = validate_roi(roi, image.shape[:2]) if roi is not None else None
    if polygon is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        gray = np.where(mask, gray, 255).astype(np.uint8)
    corners, ids, _, _ = detector.detectBoard(gray)
    points, duplicates = _as_point_map(corners, ids)
    if polygon is not None:
        points = {i: p for i, p in points.items()
                  if cv2.pointPolygonTest(polygon, tuple(p), False) >= 0}
    return points, duplicates


def validate_roi(roi, image_shape) -> np.ndarray:
    """Validate a simple convex raw-pixel polygon; refuse invalid/cropped coordinates."""
    polygon = np.asarray(roi, np.float32)
    h, w = image_shape
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.all(np.isfinite(polygon))
            or np.any(polygon < 0) or np.any(polygon[:, 0] >= w)
            or np.any(polygon[:, 1] >= h)
            or not cv2.isContourConvex(polygon)
            or abs(cv2.contourArea(polygon)) < 1):
        raise ValueError("ROI must be a nondegenerate convex polygon inside the raw image")
    return polygon


def _homography_inliers(
    known_points: np.ndarray,
    image_points: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(known_points) < 4:
        raise ValueError("at least four points are required for a homography")
    H, mask = cv2.findHomography(
        np.asarray(known_points, np.float64).reshape(-1, 2),
        np.asarray(image_points, np.float64).reshape(-1, 2),
        cv2.RANSAC,
        float(threshold_px),
    )
    if H is None or mask is None:
        raise ValueError("could not fit a coherent board homography")
    projected = cv2.perspectiveTransform(
        np.asarray(known_points, np.float64).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    residual = np.linalg.norm(projected - image_points, axis=1)
    return mask.reshape(-1).astype(bool), residual


def _select_spread_ids(
    corner_ids: Sequence[int],
    board_points: np.ndarray,
    maximum: int,
) -> list[int]:
    """Deterministically select spatially distributed board corners."""
    ids = sorted(int(value) for value in corner_ids)
    if maximum <= 0 or len(ids) <= maximum:
        return ids
    xy = np.asarray([board_points[corner_id, :2] for corner_id in ids], np.float64)
    scale = np.ptp(xy, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (xy - xy.min(axis=0)) / scale
    selected = [int(np.argmin(normalized[:, 0] + normalized[:, 1]))]
    remaining = set(range(len(ids))) - set(selected)
    while remaining and len(selected) < maximum:
        next_index = max(
            remaining,
            key=lambda index: min(
                float(np.linalg.norm(normalized[index] - normalized[chosen]))
                for chosen in selected
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return [ids[index] for index in selected]


def select_coherent_common_corners(
    points0: Mapping[int, np.ndarray],
    points1: Mapping[int, np.ndarray],
    board_points: np.ndarray,
    *,
    ransac_threshold_px: float = 3.0,
    max_corners: int = 20,
    min_corners: int = 10,
    calibration: StereoBoardCalibration | None = None,
) -> tuple[list[int], dict[str, float | int]]:
    """Keep a coherent, spatially distributed set visible in both cameras.

    Independent homographies are fitted in the two images.  A corner must be a
    RANSAC inlier in both views, which removes detections that do not belong to
    the same planar board instance (including isolated reflected copies).
    """
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise ValueError("ransac_threshold_px must be positive and finite")
    if min_corners < 4:
        raise ValueError("min_corners must be at least four")
    if max_corners > 0 and max_corners < min_corners:
        raise ValueError("max_corners must be zero or at least min_corners")
    common = sorted(set(points0) & set(points1))
    common = [corner_id for corner_id in common if 0 <= corner_id < len(board_points)]
    if len(common) < min_corners:
        raise ValueError(
            f"only {len(common)} common board corners; need at least {min_corners}")
    known = np.asarray([board_points[corner_id, :2] for corner_id in common], np.float64)
    image0 = np.asarray([points0[corner_id] for corner_id in common], np.float64)
    image1 = np.asarray([points1[corner_id] for corner_id in common], np.float64)
    if calibration is not None:
        # A plane homography is exact in ideal pinhole pixels, not distorted raw pixels.
        image0 = cv2.undistortPoints(image0.reshape(-1, 1, 2), calibration.K0,
                                   calibration.D0, P=calibration.K0).reshape(-1, 2)
        image1 = cv2.undistortPoints(image1.reshape(-1, 1, 2), calibration.K1,
                                   calibration.D1, P=calibration.K1).reshape(-1, 2)
    inlier0, residual0 = _homography_inliers(known, image0, ransac_threshold_px)
    inlier1, residual1 = _homography_inliers(known, image1, ransac_threshold_px)
    coherent = [corner_id for index, corner_id in enumerate(common) if inlier0[index] and inlier1[index]]
    if len(coherent) < min_corners:
        raise ValueError(
            f"only {len(coherent)} coherent corners remain after reflection/outlier rejection; "
            f"need at least {min_corners}")
    selected = _select_spread_ids(coherent, board_points, max_corners)
    summary: dict[str, float | int] = {
        "common_detected": len(common),
        "coherent_in_both": len(coherent),
        "selected": len(selected),
        "cam0_homography_median_px": float(np.median(residual0[inlier0])),
        "cam1_homography_median_px": float(np.median(residual1[inlier1])),
        "homography_pixel_space": "undistorted" if calibration is not None else "supplied",
        "candidates": [
            {"corner_id": cid, "pixel_cam0": np.asarray(points0[cid]).tolist(),
             "pixel_cam1": np.asarray(points1[cid]).tolist(),
             "homography_residual_cam0_px": float(residual0[j]),
             "homography_residual_cam1_px": float(residual1[j]),
             "selected": cid in selected,
             "exclusion": (None if cid in selected else
                           "spatial_subsample" if cid in coherent else "homography_outlier")}
            for j, cid in enumerate(common)],
    }
    return selected, summary


def pixel_ray_in_board(
    pixel: Sequence[float],
    K: np.ndarray,
    D: np.ndarray,
    R_board_to_cam: np.ndarray,
    t_board_to_cam: np.ndarray,
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
    centre_board: np.ndarray,
    direction_board: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Intersect a board-frame ray with ``Z=0`` without altering stereo data."""
    denominator = float(direction_board[2])
    if not np.isfinite(denominator) or abs(denominator) < 1e-6:
        raise ValueError("camera ray is parallel or nearly parallel to the board plane")
    distance = -float(centre_board[2]) / denominator
    if distance <= 0:
        raise ValueError("board plane is behind the camera for this pixel")
    point = np.asarray(centre_board, np.float64) + distance * np.asarray(
        direction_board, np.float64)
    point[2] = 0.0
    return point, distance


def triangulate_unconstrained(
    pixel0: Sequence[float],
    pixel1: Sequence[float],
    calibration: StereoBoardCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate freely in 3-D and return camera-0 and camera-1 coordinates."""
    result = triangulate_point(pixel0, pixel1, calibration)
    point0 = np.asarray(result["X_cam0_mm"])
    point1 = np.asarray(result["X_cam1_mm"])
    if not np.all(np.isfinite(point0)):
        raise ValueError("triangulation produced a point at infinity")
    if point0[2] <= 0 or point1[2] <= 0:
        raise ValueError("triangulated point is behind a camera")
    return point0, point1


def rotation_difference_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Return the geodesic angular difference between two rotation matrices."""
    relative = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compare_relative_geometry(calibration: StereoBoardCalibration) -> dict[str, object]:
    """Compare saved stereo ``R,T`` with the transform implied by world poses."""
    R_world = calibration.R_board_to_cam1 @ calibration.R_board_to_cam0.T
    T_world = (
        calibration.t_board_to_cam1
        - R_world @ calibration.t_board_to_cam0
    )
    stereo_baseline = float(np.linalg.norm(calibration.T))
    world_baseline = float(np.linalg.norm(T_world))
    return {
        "stereo_R": calibration.R.tolist(),
        "stereo_T_mm": calibration.T.tolist(),
        "world_implied_R": R_world.tolist(),
        "world_implied_T_mm": T_world.tolist(),
        "rotation_difference_deg": rotation_difference_deg(calibration.R, R_world),
        "translation_difference_mm": float(np.linalg.norm(calibration.T - T_world)),
        "stereo_baseline_mm": stereo_baseline,
        "world_implied_baseline_mm": world_baseline,
        "baseline_difference_mm": world_baseline - stereo_baseline,
        "translation_delta_world_minus_stereo_mm": (T_world - calibration.T).tolist(),
        "world_board_origin_prediction_error_mm": float(np.linalg.norm(
            calibration.R @ calibration.t_board_to_cam0 + calibration.T
            - calibration.t_board_to_cam1)),
        "camera_centres_board_mm": [(-R.T @ t).tolist() for R, t in (
            (calibration.R_board_to_cam0, calibration.t_board_to_cam0),
            (calibration.R_board_to_cam1, calibration.t_board_to_cam1))],
        "comparison_frame": "cam0 -> cam1; both translation vectors expressed in cam1 axes",
    }


def _parallax_deg(
    centre0: np.ndarray,
    direction0: np.ndarray,
    centre1: np.ndarray,
    direction1: np.ndarray,
) -> float:
    del centre0, centre1  # centres document the common board-frame convention
    cosine = np.clip(float(np.dot(direction0, direction1)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def analyze_corner(
    corner_id: int,
    pixel0: Sequence[float],
    pixel1: Sequence[float],
    known_board_point: Sequence[float],
    calibration: StereoBoardCalibration,
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, object]:
    """Retain signed residuals and invalid reconstructions without correcting XYZ.

    Error vectors are estimated/projected minus known/observed. ``valid`` is
    the existing stereo geometric gate, never a physical accuracy verdict.
    """
    thresholds = thresholds or DiagnosticThresholds()
    known = np.asarray(known_board_point, np.float64).reshape(3)
    centre0, direction0 = pixel_ray_in_board(
        pixel0,
        calibration.K0,
        calibration.D0,
        calibration.R_board_to_cam0,
        calibration.t_board_to_cam0,
    )
    centre1, direction1 = pixel_ray_in_board(
        pixel1,
        calibration.K1,
        calibration.D1,
        calibration.R_board_to_cam1,
        calibration.t_board_to_cam1,
    )
    rec = triangulate_point(pixel0, pixel1, calibration)
    flags = list(rec["flags"])
    planes = []
    for i, (centre, direction) in enumerate(((centre0, direction0), (centre1, direction1))):
        try:
            plane, _ = intersect_ray_with_board_plane(centre, direction)
        except ValueError as exc:
            plane = np.full(3, np.nan)
            flags.append(f"cam{i}_plane_intersection_failed: {exc}")
        planes.append(plane)
    plane0, plane1 = planes
    stereo0, stereo1 = np.asarray(rec["X_cam0_mm"]), np.asarray(rec["X_cam1_mm"])
    stereo_board0 = calibration.R_board_to_cam0.T @ (
        stereo0 - calibration.t_board_to_cam0
    )
    stereo_board1 = calibration.R_board_to_cam1.T @ (
        stereo1 - calibration.t_board_to_cam1
    )
    row = {
        "corner_id": int(corner_id),
        "pixel_cam0": np.asarray(pixel0, np.float64).reshape(2).tolist(),
        "pixel_cam1": np.asarray(pixel1, np.float64).reshape(2).tolist(),
        "known_board_mm": known.tolist(),
        "cam0_plane_mm": plane0.tolist(),
        "cam1_plane_mm": plane1.tolist(),
        "cam0_plane_error_mm": float(np.linalg.norm(plane0 - known)),
        "cam1_plane_error_mm": float(np.linalg.norm(plane1 - known)),
        "plane_intersection_disagreement_mm": float(np.linalg.norm(plane0 - plane1)),
        "stereo_cam0_mm": stereo0.tolist(),
        "stereo_board_via_cam0_mm": stereo_board0.tolist(),
        "stereo_board_via_cam1_mm": stereo_board1.tolist(),
        "stereo_error_mm": float(np.linalg.norm(stereo_board0 - known)),
        "stereo_xy_error_mm": float(np.linalg.norm(stereo_board0[:2] - known[:2])),
        "stereo_plane_error_mm": abs(float(stereo_board0[2])),
        "stereo_world_pose_disagreement_mm": float(
            np.linalg.norm(stereo_board0 - stereo_board1)
        ),
        "parallax_deg": rec["parallax_deg"],
        "world_pose_parallax_deg": _parallax_deg(centre0, direction0, centre1, direction1),
        "valid": rec["valid"],
        "flags": flags,
        "epipolar_sampson_px": rec["epipolar_sampson_px"],
        "depth_cam0_mm": rec["depth_z_cam0_mm"],
        "depth_cam1_mm": rec["depth_z_cam1_mm"],
        "stereo_error_vector_mm": (stereo_board0 - known).tolist(),
        "plane_disagreement_vector_mm": (plane0 - plane1).tolist(),
    }
    for i, (pixel, K, D, R, t, plane) in enumerate((
        (pixel0, calibration.K0, calibration.D0, calibration.R_board_to_cam0,
         calibration.t_board_to_cam0, plane0),
        (pixel1, calibration.K1, calibration.D1, calibration.R_board_to_cam1,
         calibration.t_board_to_cam1, plane1))):
        projection = cv2.projectPoints(known.reshape(1, 3), cv2.Rodrigues(R)[0], t, K, D)[0].reshape(2)
        residual = projection - np.asarray(pixel)
        row[f"cam{i}_known_projected_px"] = projection.tolist()
        row[f"cam{i}_world_residual_vector_px"] = residual.tolist()
        row[f"cam{i}_world_reproj_error_px"] = float(np.linalg.norm(residual))
        row[f"cam{i}_plane_error_vector_mm"] = (plane - known).tolist()
        row[f"cam{i}_stereo_projected_px"] = rec[f"reproj_cam{i}_px"]
        row[f"cam{i}_stereo_residual_vector_px"] = (np.asarray(rec[f"reproj_cam{i}_px"]) - pixel).tolist()
        row[f"cam{i}_stereo_reproj_error_px"] = rec[f"reproj_error_cam{i}_px"]
        if (R @ known + t)[2] <= 0:
            flags.append(f"known_point_behind_cam{i}_world_pose")
        if row[f"cam{i}_plane_error_mm"] > thresholds.plane_error_mm:
            flags.append(f"cam{i}_plane_error_above_demo_limit")
    if row["stereo_error_mm"] > thresholds.stereo_error_mm:
        flags.append("stereo_error_above_demo_limit")
    return row


def _finite_median(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def summarize_diagnostic(
    rows: Sequence[Mapping[str, object]],
    relative_geometry: Mapping[str, object],
    thresholds: DiagnosticThresholds,
) -> dict[str, object]:
    """Summarize errors and classify which part of the geometry is failing."""
    if not rows:
        raise ValueError("cannot summarize an empty diagnostic")
    med0 = _finite_median(rows, "cam0_plane_error_mm")
    med1 = _finite_median(rows, "cam1_plane_error_mm")
    stereo = _finite_median(rows, "stereo_error_mm")
    summary: dict[str, object] = {
        "corners_analyzed": len(rows),
        "median_cam0_plane_error_mm": med0,
        "median_cam1_plane_error_mm": med1,
        "median_plane_intersection_disagreement_mm": _finite_median(
            rows, "plane_intersection_disagreement_mm"),
        "median_stereo_error_mm": stereo,
        "median_stereo_xy_error_mm": _finite_median(rows, "stereo_xy_error_mm"),
        "median_stereo_plane_error_mm": _finite_median(rows, "stereo_plane_error_mm"),
        "median_stereo_world_pose_disagreement_mm": _finite_median(
            rows, "stereo_world_pose_disagreement_mm"),
        "median_parallax_deg": _finite_median(rows, "parallax_deg"),
        "thresholds": {
            "plane_error_mm": thresholds.plane_error_mm,
            "stereo_error_mm": thresholds.stereo_error_mm,
            "rotation_error_deg": thresholds.rotation_error_deg,
            "translation_error_mm": thresholds.translation_error_mm,
        },
    }
    planes_good = (np.isfinite(med0) and np.isfinite(med1)
                   and max(med0, med1) <= thresholds.plane_error_mm)
    stereo_good = stereo <= thresholds.stereo_error_mm
    relative_good = (
        float(relative_geometry["rotation_difference_deg"])
        <= thresholds.rotation_error_deg
        and float(relative_geometry["translation_difference_mm"])
        <= thresholds.translation_error_mm
    )
    if not np.isfinite(stereo):
        diagnosis = "invalid_stereo_correspondences_or_geometry"
    elif planes_good and stereo_good and relative_good:
        diagnosis = "geometry_consistent"
    elif planes_good and (not stereo_good or not relative_good):
        diagnosis = "stereo_geometry_mismatch"
    else:
        diagnosis = "camera_pose_intrinsics_detection_or_board_mismatch"
    summary["diagnosis"] = diagnosis
    summary["thresholds_are_calibrated"] = False
    from charuco_diagnostic_report import error_statistics
    summary["statistics"] = error_statistics(rows)
    summary["invalid_geometric_count"] = sum(not r.get("valid", True) for r in rows)
    summary["interpretation"] = (
        "Internal model consistency only. Board geometry and world poses share the reference; "
        "independent lengths/heights are still required. valid does not imply physical accuracy.")
    return summary


def run_diagnostic(
    image0: np.ndarray,
    image1: np.ndarray,
    calibration: StereoBoardCalibration,
    board,
    detector,
    *,
    max_corners: int = 20,
    min_corners: int = 10,
    ransac_threshold_px: float = 3.0,
    thresholds: DiagnosticThresholds | None = None,
    roi0=None,
    roi1=None,
) -> dict[str, object]:
    """Run the complete automatic dry-board geometry diagnostic."""
    if image0 is None or image1 is None:
        raise ValueError("both camera images are required")
    height0, width0 = image0.shape[:2]
    height1, width1 = image1.shape[:2]
    if (width0, height0) != calibration.image_size:
        raise ValueError(
            f"cam0 image size {(width0, height0)} != calibration {calibration.image_size}")
    if (width1, height1) != calibration.image_size:
        raise ValueError(
            f"cam1 image size {(width1, height1)} != calibration {calibration.image_size}")
    points0, duplicates0 = detect_charuco_points(image0, detector, roi0)
    points1, duplicates1 = detect_charuco_points(image1, detector, roi1)
    board_points = np.asarray(board.getChessboardCorners(), np.float64)
    selected, selection = select_coherent_common_corners(
        points0,
        points1,
        board_points,
        ransac_threshold_px=ransac_threshold_px,
        max_corners=max_corners,
        min_corners=min_corners,
        calibration=calibration,
    )
    rows = [
        analyze_corner(
            corner_id,
            points0[corner_id],
            points1[corner_id],
            board_points[corner_id],
            calibration,
            thresholds,
        )
        for corner_id in selected
    ]
    relative = compare_relative_geometry(calibration)
    summary = summarize_diagnostic(
        rows, relative, thresholds or DiagnosticThresholds())
    selection.update({
        "cam0_duplicate_ids_excluded": duplicates0,
        "cam1_duplicate_ids_excluded": duplicates1,
        "ransac_threshold_px": ransac_threshold_px,
        "roi_cam0": roi0,
        "roi_cam1": roi1,
        "reflection_control": "user_supplied_roi_requires_overlay_review" if roi0 is not None and roi1 is not None
        else "UNVERIFIED_full_frame_may_include_coherent_reflections",
        "cam0_detected_ids": sorted(points0),
        "cam1_detected_ids": sorted(points1),
    })
    return {
        "selection": selection,
        "relative_geometry": relative,
        "summary": summary,
        "corners": rows,
    }


def _flatten_row(row: Mapping[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {"corner_id": row["corner_id"]}
    vectors = {
        "known": "known_board_mm",
        "cam0_plane": "cam0_plane_mm",
        "cam1_plane": "cam1_plane_mm",
        "stereo": "stereo_board_via_cam0_mm",
    }
    for prefix, key in vectors.items():
        vector = list(row[key])
        for axis, value in zip("xyz", vector):
            flat[f"{prefix}_{axis}_mm"] = value
    for key in (
        "cam0_plane_error_mm",
        "cam1_plane_error_mm",
        "plane_intersection_disagreement_mm",
        "stereo_error_mm",
        "stereo_xy_error_mm",
        "stereo_plane_error_mm",
        "stereo_world_pose_disagreement_mm",
        "parallax_deg",
    ):
        flat[key] = row[key]
    for key, value in row.items():
        if isinstance(value, list) and key != "flags":
            axes = "uv" if len(value) == 2 else "xyz"
            for axis, entry in zip(axes, value):
                flat[f"{key}_{axis}"] = entry
        elif not isinstance(value, (dict, list)):
            flat[key] = value
    flat["flags"] = ";".join(row.get("flags", []))
    return flat


def write_results(result: Mapping[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    """Write a fresh report directory; refuse to overwrite any existing run."""
    from charuco_diagnostic_report import json_safe
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "charuco_geometry_diagnostic.json"
    csv_path = output / "charuco_geometry_corners.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat_rows = [json_safe(_flatten_row(row)) for row in result.get("corners", [])]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else ["corner_id"])
        writer.writeheader()
        writer.writerows(flat_rows)
    return json_path, csv_path


def calibration_provenance(
    pair_metadata: str | Path,
    calibration_paths: Mapping[str, str | Path],
) -> dict[str, object]:
    """Compare calibration files with the hashes recorded at pair capture.

    A raw image pair can intentionally be reprocessed with a newer calibration,
    but that choice must be visible in the report rather than silently mixing
    calibration generations.
    """
    metadata_path = Path(pair_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    recorded = metadata.get("calibration_hashes", {})
    files: dict[str, object] = {}
    mismatched = []
    for name, path_value in calibration_paths.items():
        path = Path(path_value)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = recorded.get(name)
        matches = expected == digest if expected is not None else None
        files[name] = {
            "path": str(path.resolve()),
            "recorded_sha256": expected,
            "actual_sha256": digest,
            "matches_capture_metadata": matches,
        }
        if matches is False:
            mismatched.append(name)
    return {
        "pair_metadata": str(metadata_path.resolve()),
        "recorded_hashes_available": bool(recorded),
        "mismatched_files": mismatched,
        "files": files,
        "capture_metadata": metadata,
    }


def print_report(result: Mapping[str, object]) -> None:
    """Print a compact human-readable diagnostic report."""
    relative = result["relative_geometry"]
    audit = result.get("audit", {})
    for name, entry in audit.get("files", {}).items():
        print(f"{name}: {entry['path']}\n  SHA256 {entry['sha256']}")
    print(
        f"Stereo/world-implied baseline: {relative['stereo_baseline_mm']:.4f} / "
        f"{relative['world_implied_baseline_mm']:.4f} mm; "
        f"rotation difference {relative['rotation_difference_deg']:.4f} deg; "
        f"translation difference {relative['translation_difference_mm']:.4f} mm")
    print(f"World minus stereo T (cam1 axes, mm): {relative['translation_delta_world_minus_stereo_mm']}")
    print(f"Board->tank identity placeholder: {audit.get('tank_transform_identity_placeholder', 'unknown')}; tank axes not verified")
    if "selection" not in result:
        return
    selection = result["selection"]
    relative = result["relative_geometry"]
    summary = result["summary"]
    print("\n=== Automatic ChArUco geometry diagnostic ===")
    print(
        f"Corners: {selection['common_detected']} common, "
        f"{selection['coherent_in_both']} coherent, {selection['selected']} analyzed")
    print(
        "Independent ray/plane median error: "
        f"cam0={summary['median_cam0_plane_error_mm']:.3f} mm, "
        f"cam1={summary['median_cam1_plane_error_mm']:.3f} mm")
    print(
        f"Unconstrained stereo median error: {summary['median_stereo_error_mm']:.3f} mm "
        f"(plane |Z|={summary['median_stereo_plane_error_mm']:.3f} mm)")
    print(
        "Stereo vs world-implied geometry: "
        f"rotation={relative['rotation_difference_deg']:.3f} deg, "
        f"translation={relative['translation_difference_mm']:.3f} mm, "
        f"baselines={relative['stereo_baseline_mm']:.3f}/"
        f"{relative['world_implied_baseline_mm']:.3f} mm")
    print(f"Diagnosis: {summary['diagnosis']}")
    print(f"Geometric invalids retained: {summary['invalid_geometric_count']}/{len(result['corners'])}")
    print(f"Reflection control: {selection['reflection_control']}")
    print(summary["interpretation"])
    provenance = result.get("calibration_provenance")
    if provenance and provenance.get("mismatched_files"):
        names = ", ".join(provenance["mismatched_files"])
        print(
            "Calibration provenance warning: this pair was captured with "
            f"different recorded files ({names}).")
    print("Thresholds are diagnostic demonstration values, not experimentally calibrated limits.")


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare known ChArUco corners with independent camera-plane "
            "intersections and unconstrained stereo triangulation."))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--inspect-calibration", action="store_true", help="Audit files and compare full R/T without images")
    source.add_argument(
        "--pair-dir",
        help="Directory containing cam0.png and cam1.png",
    )
    source.add_argument(
        "--cam0-image",
        help="Camera-0 image; requires --cam1-image",
    )
    parser.add_argument("--cam1-image", help="Camera-1 image")
    parser.add_argument("--roi-file", help="JSON with cam0/cam1 convex polygons in original raw pixels")
    parser.add_argument("--preview-only", action="store_true", help="Save raw-size detections/ROI for inspection before reconstruction")
    parser.add_argument("--allow-unverified-full-frame", action="store_true", help="Explicitly allow analysis without direct-board ROI (unverified reflections)")
    parser.add_argument("--param0", default="cam0_intrinsics.npz")
    parser.add_argument("--param1", default="cam1_intrinsics.npz")
    parser.add_argument("--stereo", default="stereo_params.npz")
    parser.add_argument("--tank", default="tank_alignment.npz")
    parser.add_argument("--max-corners", type=int, default=20)
    parser.add_argument("--min-corners", type=int, default=10)
    parser.add_argument("--ransac-threshold-px", type=float, default=3.0)
    parser.add_argument("--plane-error-mm", type=float, default=5.0)
    parser.add_argument("--stereo-error-mm", type=float, default=5.0)
    parser.add_argument("--rotation-error-deg", type=float, default=1.0)
    parser.add_argument("--translation-error-mm", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for JSON and CSV (for example outputs/charuco_diagnostic)",
    )
    add_board_argument(parser, default="long")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.board != "long":
        parser.error("this diagnostic requires the original --board long (4x35, 44/33 mm)")
    if args.output_dir and Path(args.output_dir).exists():
        parser.error("output directory already exists; choose a new run directory to preserve evidence")
    paths = {"cam0_intrinsics.npz": args.param0, "cam1_intrinsics.npz": args.param1,
             "stereo_params.npz": args.stereo, "tank_alignment.npz": args.tank}
    from charuco_diagnostic_report import audit_calibration, annotate_image, json_safe, write_visuals
    spec = get_board(args.board)
    calibration = StereoBoardCalibration.from_files(*paths.values())
    audit = audit_calibration(paths, spec)
    if args.inspect_calibration:
        result = {"audit": audit, "relative_geometry": compare_relative_geometry(calibration)}
        print_report(result)
        if args.output_dir:
            write_results(result, args.output_dir)
        return
    if not args.pair_dir and not args.cam0_image:
        parser.error("supply --pair-dir, --cam0-image/--cam1-image, or --inspect-calibration")
    if args.pair_dir:
        pair_dir = Path(args.pair_dir)
        image0_path = pair_dir / "cam0.png"
        image1_path = pair_dir / "cam1.png"
    else:
        if not args.cam1_image:
            raise SystemExit("--cam1-image is required with --cam0-image")
        image0_path = Path(args.cam0_image)
        image1_path = Path(args.cam1_image)
    image0, image1 = _read_image(image0_path), _read_image(image1_path)
    for i, image in enumerate((image0, image1)):
        if (image.shape[1], image.shape[0]) != calibration.image_size:
            parser.error(f"cam{i} image size differs from calibration; use original raw images")
    rois = json.loads(Path(args.roi_file).read_text()) if args.roi_file else {}
    if args.roi_file:
        for i in (0, 1):
            if f"cam{i}" not in rois:
                parser.error("ROI JSON requires both cam0 and cam1 polygons")
            validate_roi(rois[f"cam{i}"], image0.shape[:2])
    if args.preview_only:
        if not args.output_dir:
            parser.error("--preview-only requires a fresh --output-dir")
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        for i, image in enumerate((image0, image1)):
            points, duplicates = detect_charuco_points(image, spec.detector, rois.get(f"cam{i}"))
            overlay = annotate_image(image, i, points, duplicates, rois.get(f"cam{i}"))
            if not cv2.imwrite(str(out / f"cam{i}_preview.png"), overlay):
                raise OSError("could not write preview")
        (out / "audit.json").write_text(json.dumps(json_safe(audit), indent=2, allow_nan=False) + "\n")
        print(f"Preview saved to {out}; review physical board ROI before reconstruction.")
        return
    if not args.roi_file and not args.allow_unverified_full_frame:
        parser.error("supply --roi-file for direct board, or use --preview-only first; full-frame analysis requires explicit --allow-unverified-full-frame")
    result = run_diagnostic(
        image0,
        image1,
        calibration,
        spec.board,
        spec.detector,
        max_corners=args.max_corners,
        min_corners=args.min_corners,
        ransac_threshold_px=args.ransac_threshold_px,
        roi0=rois.get("cam0"), roi1=rois.get("cam1"),
        thresholds=DiagnosticThresholds(
            plane_error_mm=args.plane_error_mm,
            stereo_error_mm=args.stereo_error_mm,
            rotation_error_deg=args.rotation_error_deg,
            translation_error_mm=args.translation_error_mm,
        ),
    )
    result["inputs"] = {
        "cam0_image": str(image0_path.resolve()),
        "cam1_image": str(image1_path.resolve()),
        "param0": str(Path(args.param0).resolve()),
        "param1": str(Path(args.param1).resolve()),
        "stereo": str(Path(args.stereo).resolve()),
        "tank": str(Path(args.tank).resolve()),
        "board": args.board,
        "roi_file": str(Path(args.roi_file).resolve()) if args.roi_file else None,
        "image_sha256": {f"cam{i}": hashlib.sha256(p.read_bytes()).hexdigest()
                         for i, p in enumerate((image0_path, image1_path))},
    }
    result["audit"] = audit
    result["calibration_provenance"] = calibration_provenance(image0_path.parent / "pair.json", paths)
    for name, path in paths.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != audit["files"][name]["sha256"]:
            raise RuntimeError(f"{name} changed during analysis; report would mix calibrations")
    result["audit"]["calibration_hashes_unchanged_after_analysis"] = True
    print_report(result)
    if args.output_dir:
        json_path, csv_path = write_results(result, args.output_dir)
        import contextlib
        with (Path(args.output_dir) / "summary.txt").open("x") as handle:
            with contextlib.redirect_stdout(handle):
                print_report(result)
        note = write_visuals(result, image0, image1, spec.detector, args.output_dir)
        if note:
            print(note)
        print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
