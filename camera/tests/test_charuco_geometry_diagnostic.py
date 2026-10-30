from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


CAMERA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMERA_DIR))

from charuco_common import get_board  # noqa: E402
from charuco_geometry_diagnostic import (  # noqa: E402
    DiagnosticThresholds,
    StereoBoardCalibration,
    analyze_corner,
    calibration_provenance,
    compare_relative_geometry,
    run_diagnostic,
    select_coherent_common_corners,
    summarize_diagnostic,
)


def _synthetic_calibration(stereo_translation_x: float = -100.0):
    K = np.array(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.zeros((1, 5), dtype=np.float64)
    # Cameras are 1 m above the board and look down.  Camera 1 is 100 mm
    # across the board from camera 0.
    R0 = np.diag([1.0, -1.0, -1.0])
    centre0 = np.array([0.0, 0.0, 1000.0])
    centre1 = np.array([100.0, 0.0, 1000.0])
    t0 = -R0 @ centre0
    t1 = -R0 @ centre1
    calibration = StereoBoardCalibration(
        K0=K,
        D0=D,
        K1=K.copy(),
        D1=D.copy(),
        R=np.eye(3),
        T=np.array([stereo_translation_x, 0.0, 0.0]),
        R_board_to_cam0=R0,
        t_board_to_cam0=t0,
        R_board_to_cam1=R0.copy(),
        t_board_to_cam1=t1,
        image_size=(1280, 720),
    )
    return calibration


def _project_board(point, K, D, R, t):
    rvec, _ = cv2.Rodrigues(R)
    pixel, _ = cv2.projectPoints(
        np.asarray(point, np.float64).reshape(1, 1, 3), rvec, t, K, D
    )
    return pixel.reshape(2)


def test_exact_geometry_recovers_known_board_corner():
    calibration = _synthetic_calibration()
    known = np.array([44.0, 220.0, 0.0])
    pixel0 = _project_board(
        known,
        calibration.K0,
        calibration.D0,
        calibration.R_board_to_cam0,
        calibration.t_board_to_cam0,
    )
    pixel1 = _project_board(
        known,
        calibration.K1,
        calibration.D1,
        calibration.R_board_to_cam1,
        calibration.t_board_to_cam1,
    )

    row = analyze_corner(17, pixel0, pixel1, known, calibration)

    assert row["cam0_plane_mm"] == pytest.approx(known, abs=1e-8)
    assert row["cam1_plane_mm"] == pytest.approx(known, abs=1e-8)
    assert row["stereo_board_via_cam0_mm"] == pytest.approx(known, abs=1e-8)
    assert row["stereo_error_mm"] == pytest.approx(0.0, abs=1e-8)
    relative = compare_relative_geometry(calibration)
    assert relative["rotation_difference_deg"] == pytest.approx(0.0, abs=1e-8)
    assert relative["translation_difference_mm"] == pytest.approx(0.0, abs=1e-8)


def test_good_plane_intersections_isolate_bad_stereo_geometry():
    calibration = _synthetic_calibration(stereo_translation_x=-60.0)
    rows = []
    for corner_id, known in enumerate(
        np.array([[x, y, 0.0] for y in (80.0, 160.0) for x in (20.0, 60.0, 100.0)])
    ):
        pixel0 = _project_board(
            known,
            calibration.K0,
            calibration.D0,
            calibration.R_board_to_cam0,
            calibration.t_board_to_cam0,
        )
        pixel1 = _project_board(
            known,
            calibration.K1,
            calibration.D1,
            calibration.R_board_to_cam1,
            calibration.t_board_to_cam1,
        )
        rows.append(analyze_corner(corner_id, pixel0, pixel1, known, calibration))

    relative = compare_relative_geometry(calibration)
    summary = summarize_diagnostic(rows, relative, DiagnosticThresholds())

    assert summary["median_cam0_plane_error_mm"] < 1e-7
    assert summary["median_cam1_plane_error_mm"] < 1e-7
    assert summary["median_stereo_error_mm"] > 100.0
    assert relative["translation_difference_mm"] == pytest.approx(40.0)
    assert summary["diagnosis"] == "stereo_geometry_mismatch"


def test_homography_gate_rejects_incoherent_reflection_or_outlier():
    board_points = np.array(
        [[x * 44.0, y * 44.0, 0.0] for y in range(4) for x in range(3)],
        dtype=np.float64,
    )
    points0 = {
        corner_id: np.array([200.0 + point[0] * 2.0, 100.0 + point[1] * 1.5])
        for corner_id, point in enumerate(board_points)
    }
    points1 = {
        corner_id: np.array([260.0 + point[0] * 1.8, 90.0 + point[1] * 1.6])
        for corner_id, point in enumerate(board_points)
    }
    # Simulate a point selected from an acrylic reflection instead of the
    # coherent physical board.
    points1[5] = points1[5] + np.array([240.0, -170.0])

    selected, selection = select_coherent_common_corners(
        points0,
        points1,
        board_points,
        min_corners=8,
        max_corners=20,
        ransac_threshold_px=2.0,
    )

    assert 5 not in selected
    assert selection["common_detected"] == 12
    assert selection["coherent_in_both"] == 11


def test_checked_in_pair_runs_without_camera_hardware():
    pair = CAMERA_DIR / "validation" / "pairs" / "pair_0018_20261029_142637"
    calibration = StereoBoardCalibration.from_files(
        CAMERA_DIR / "cam0_intrinsics.npz",
        CAMERA_DIR / "cam1_intrinsics.npz",
        CAMERA_DIR / "stereo_params.npz",
        CAMERA_DIR / "tank_alignment.npz",
    )
    spec = get_board("long")
    result = run_diagnostic(
        cv2.imread(str(pair / "cam0.png")),
        cv2.imread(str(pair / "cam1.png")),
        calibration,
        spec.board,
        spec.detector,
        max_corners=12,
        min_corners=10,
    )

    assert result["selection"]["coherent_in_both"] >= 10
    assert len(result["corners"]) == 12
    # The updated world poses fit the known board well, while the stereo
    # transform still produces a large unconstrained vertical error.
    assert result["summary"]["median_cam0_plane_error_mm"] < 2.0
    assert result["summary"]["median_cam1_plane_error_mm"] < 2.0
    assert result["summary"]["median_stereo_plane_error_mm"] > 50.0
    assert result["summary"]["diagnosis"] == "stereo_geometry_mismatch"

    provenance = calibration_provenance(
        pair / "pair.json",
        {
            "cam0_intrinsics.npz": CAMERA_DIR / "cam0_intrinsics.npz",
            "cam1_intrinsics.npz": CAMERA_DIR / "cam1_intrinsics.npz",
            "stereo_params.npz": CAMERA_DIR / "stereo_params.npz",
            "tank_alignment.npz": CAMERA_DIR / "tank_alignment.npz",
        },
    )
    assert provenance["mismatched_files"] == [
        "stereo_params.npz",
        "tank_alignment.npz",
    ]
