from __future__ import annotations

import sys
import json
import hashlib
import subprocess
from dataclasses import replace
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
    detect_charuco_points,
    intersect_ray_with_board_plane,
    write_results,
    _as_point_map,
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


def observations(cal, point):
    return tuple(_project_board(point, getattr(cal, f"K{i}"), getattr(cal, f"D{i}"),
                                getattr(cal, f"R_board_to_cam{i}"),
                                getattr(cal, f"t_board_to_cam{i}")) for i in (0, 1))


def test_opposite_roll_distorted_cameras_preserve_full_frame_conventions():
    cal = _synthetic_calibration()
    roll = cv2.Rodrigues(np.array([0.03, -0.02, 3.05]))[0]
    cal = replace(cal, R=roll, T=roll @ cal.T,
                  R_board_to_cam1=roll @ cal.R_board_to_cam1,
                  t_board_to_cam1=roll @ cal.t_board_to_cam1,
                  D0=np.array([0.03, -0.02, 0.001, -0.002, 0.01]),
                  D1=np.array([-0.01, 0.01, -0.001, 0.002, 0.02]))
    point = np.array([88., 440., 0.])
    row = analyze_corner(0, *observations(cal, point), point, cal)
    assert row["stereo_board_via_cam0_mm"] == pytest.approx(point, abs=2e-4)
    assert row["cam0_plane_mm"] == pytest.approx(point, abs=2e-4)
    assert row["cam1_plane_mm"] == pytest.approx(point, abs=2e-4)
    assert row["cam0_world_residual_vector_px"] == pytest.approx([0, 0], abs=1e-7)
    assert row["valid"]
    relative = compare_relative_geometry(cal)
    assert relative["rotation_difference_deg"] < 2e-6
    assert relative["translation_difference_mm"] < 1e-9


def test_low_reprojection_error_can_have_large_metric_error():
    exact = _synthetic_calibration()
    point = np.array([44., 220., 0.])
    row = analyze_corner(0, *observations(exact, point), point,
                         replace(exact, T=exact.T * 0.7))
    assert row["valid"]
    assert row["cam0_stereo_reproj_error_px"] < 1e-8
    assert row["cam1_stereo_reproj_error_px"] < 1e-8
    assert row["stereo_error_mm"] > 300
    assert abs(row["stereo_error_vector_mm"][2]) > 200


def test_swapped_cameras_retained_as_invalid_not_clamped():
    cal = _synthetic_calibration()
    point = np.array([44., 220., 0.])
    a, b = observations(cal, point)
    row = analyze_corner(0, b, a, point, cal)
    assert not row["valid"]
    assert "behind_cam0" in row["flags"]
    assert np.isfinite(row["stereo_error_mm"])
    assert abs(row["stereo_board_via_cam0_mm"][2]) > 100


def test_perturbed_world_pose_changes_own_projection_but_not_stereo():
    cal = _synthetic_calibration()
    point = np.array([44., 220., 0.])
    pixels = observations(cal, point)
    bad = replace(cal, t_board_to_cam1=cal.t_board_to_cam1 + np.array([20., 0., 0.]))
    row = analyze_corner(0, *pixels, point, bad)
    assert row["stereo_error_mm"] < 1e-7
    assert row["cam0_world_reproj_error_px"] < 1e-7
    assert row["cam1_world_reproj_error_px"] > 10
    assert row["cam1_plane_error_vector_mm"][0] == pytest.approx(-20.)
    assert compare_relative_geometry(bad)["translation_difference_mm"] == pytest.approx(20.)


def test_duplicate_ids_are_excluded_instead_of_first_match():
    points, duplicates = _as_point_map(np.array([[10., 10.], [110., 110.], [20., 20.]]),
                                     np.array([3, 3, 4]))
    assert duplicates == [3]
    assert sorted(points) == [4]


def test_masking_repeated_board_before_decoding_preserves_raw_pixels():
    spec = get_board("small")
    board = spec.board.generateImage((400, 560), marginSize=20)
    # A second coherent board image models a competing decoded wall copy.
    image = np.hstack([board, board])
    left, _ = detect_charuco_points(image, spec.detector, [[0, 0], [399, 0], [399, 559], [0, 559]])
    right, _ = detect_charuco_points(image, spec.detector, [[400, 0], [799, 0], [799, 559], [400, 559]])
    assert len(left) == len(right) == 35
    for cid in left:
        assert right[cid] - left[cid] == pytest.approx([400., 0.], abs=0.01)


def test_mismatched_corner_id_triggers_world_error_even_if_stereo_is_consistent():
    cal = _synthetic_calibration()
    point = np.array([44., 220., 0.])
    wrong_known = point + np.array([44., 0., 0.])
    row = analyze_corner(1, *observations(cal, point), wrong_known, cal)
    assert row["valid"]
    assert row["stereo_error_vector_mm"] == pytest.approx([-44., 0., 0.], abs=1e-7)
    assert row["cam0_world_reproj_error_px"] > 30


def test_parallel_ray_plane_and_missing_geometry_remain_explicit(tmp_path):
    with pytest.raises(ValueError, match="parallel"):
        intersect_ray_with_board_plane(np.array([0, 0, 1000.]), np.array([1., 0., 1e-8]))
    cal = _synthetic_calibration()
    # Identical pixels with parallel cameras make stereo depth undefined.
    row = analyze_corner(0, [640., 360.], [640., 360.], [0., 0., 0.], cal)
    assert not row["valid"]
    result = {"corners": [row]}
    out = tmp_path / "new_run"
    j, csv_path = write_results(result, out)
    saved = json.loads(j.read_text(), parse_constant=lambda x: pytest.fail(f"nonstandard JSON {x}"))
    assert saved["corners"][0]["stereo_board_via_cam0_mm"] == [None, None, None]
    assert "flags" in csv_path.read_text()
    with pytest.raises(FileExistsError):
        write_results(result, out)


def test_provenance_hashes_present_without_pair_metadata(tmp_path):
    f = tmp_path / "input.npz"
    f.write_bytes(b"opaque input bytes")
    result = calibration_provenance(tmp_path / "missing.json", {"input.npz": f})
    assert not result["recorded_hashes_available"]
    assert result["files"]["input.npz"]["actual_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_roi_run_cli_outputs_and_preserves_calibrations(tmp_path):
    names = ["cam0_intrinsics.npz", "cam1_intrinsics.npz", "stereo_params.npz", "tank_alignment.npz"]
    files = [CAMERA_DIR / n for n in names]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    out = tmp_path / "report"
    cmd = [sys.executable, "charuco_geometry_diagnostic.py", "--pair-dir",
           "validation/pairs/pair_0018_20261029_142637", "--roi-file",
           "diagnostics/pair_0018_direct_board_roi.json", "--output-dir", str(out)]
    result = subprocess.run(cmd, cwd=CAMERA_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads((out / "charuco_geometry_diagnostic.json").read_text())
    assert data["audit"]["tank_transform_identity_placeholder"]
    assert len(data["audit"]["files"]) == 4
    assert data["summary"]["diagnosis"] == "stereo_geometry_mismatch"
    assert "candidates" in data["selection"]
    for name in ("cam0_annotated.png", "cam1_annotated.png"):
        assert cv2.imread(str(out / name)).shape[:2] == (1080, 1920)
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]


def test_calibration_loader_refuses_mixed_intrinsics_and_size(tmp_path):
    names = ["cam0_intrinsics.npz", "cam1_intrinsics.npz", "stereo_params.npz", "tank_alignment.npz"]
    for name in names:
        with np.load(CAMERA_DIR / name) as data:
            values = {k: data[k] for k in data.files}
        if name == "cam0_intrinsics.npz":
            values["camera_matrix"] = values["camera_matrix"].copy()
            values["camera_matrix"][0, 0] += 1
        np.savez(tmp_path / name, **values)
    with pytest.raises(ValueError, match="consistent file set"):
        StereoBoardCalibration.from_files(*(tmp_path / n for n in names))


def test_shared_change_of_reference_frame_leaves_relative_rig_unchanged():
    cal = _synthetic_calibration()
    S = cv2.Rodrigues(np.array([0.1, -0.2, 0.4]))[0]
    q = np.array([120., -40., 30.])
    R0, R1 = cal.R_board_to_cam0 @ S.T, cal.R_board_to_cam1 @ S.T
    changed = replace(cal, R_board_to_cam0=R0, R_board_to_cam1=R1,
                      t_board_to_cam0=cal.t_board_to_cam0 - R0 @ q,
                      t_board_to_cam1=cal.t_board_to_cam1 - R1 @ q)
    relative = compare_relative_geometry(changed)
    assert relative["rotation_difference_deg"] < 2e-6
    assert relative["translation_difference_mm"] < 1e-8
    assert relative["baseline_difference_mm"] == pytest.approx(0, abs=1e-8)


def test_undefined_plane_cannot_be_classified_as_consistent():
    cal = _synthetic_calibration()
    point = np.array([44., 220., 0.])
    row = analyze_corner(0, *observations(cal, point), point, cal)
    row["cam1_plane_error_mm"] = float("nan")
    summary = summarize_diagnostic([row], compare_relative_geometry(cal), DiagnosticThresholds())
    assert summary["diagnosis"] == "camera_pose_intrinsics_detection_or_board_mismatch"
