"""Keep board-Z signs separate from camera cheirality and explain rejections."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

CAMERA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMERA_DIR))

from reconstruction_diagnostics import describe_point_flags
from validate_stereo_3d import (
    Calibration, HIGH_REPROJ_PX, HIGH_EPIPOLAR_PX, REJECT_PARALLAX_DEG,
    board_to_cam0, board_up_direction, create_label_web_app, enrich_labeled_points,
    project_board_frame_px, triangulate_point,
)


def synthetic_pair(board_z):
    """Opposite camera rolls, board +Z downward, cameras above Z=0."""
    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    R = np.diag([-1., -1., 1.])
    T = np.array([-33., 0., 0.])
    tx = np.array([[0., -T[2], T[1]], [T[2], 0., -T[0]], [-T[1], T[0], 0.]])
    E = tx @ R
    tank = {"R_board_to_cam0": np.eye(3), "t_board_to_cam0": np.array([0., 0., 1000.]),
            "R_board_to_cam1": R, "t_board_to_cam1": np.array([-33., 0., 1000.]),
            "T_world_board": np.eye(4)}
    cal = Calibration(paths={}, hashes={}, K0=K, D0=np.zeros(5), K1=K.copy(),
                      D1=np.zeros(5), R=R, T=T, E=E,
                      F=np.linalg.inv(K).T @ E @ np.linalg.inv(K),
                      size0=(1280, 720), size1=(1280, 720), tank=tank)
    X0 = board_to_cam0(np.array([40., 176., board_z]), cal)
    X1 = R @ X0 + T
    def project(X):
        return cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                 K, np.zeros(5))[0].reshape(2).tolist()
    return cal, {"id": 0, "name": "A", "cam0": project(X0), "cam1": project(X1)}


def explain(point):
    return describe_point_flags(point, reprojection_limit_px=HIGH_REPROJ_PX,
                                epipolar_limit_px=HIGH_EPIPOLAR_PX,
                                rejected_parallax_deg=REJECT_PARALLAX_DEG)


@pytest.mark.parametrize("board_z", [-50., 0., 50.])
def test_both_board_z_signs_and_plane_are_accepted(board_z):
    cal, point = synthetic_pair(board_z)
    enriched, _ = enrich_labeled_points([point], cal, {"board_thickness_mm": 8.})
    result = enriched[0]
    assert result["valid"]
    assert result["X_board_mm"] == pytest.approx([40., 176., board_z], abs=1e-8)
    assert result["height_above_board_mm"] == pytest.approx(-board_z, abs=1e-8)
    assert result["height_above_tank_floor_mm"] == pytest.approx(8. - board_z, abs=1e-8)
    assert result["depth_z_cam0_mm"] > 0 and result["depth_z_cam1_mm"] > 0
    assert result["flags"] == result["geometry_diagnostics"] == []


def test_behind_camera_is_rejected_and_explained_in_camera_frame():
    cal, point = synthetic_pair(-1200.)
    result = triangulate_point(point["cam0"], point["cam1"], cal)
    assert not result["valid"]
    assert {"behind_cam0", "behind_cam1"} <= set(result["flags"])
    assert any("Z0 = -200.000" in reason and "not board Z" in reason
               for reason in explain(result))


def test_negative_board_z_does_not_bypass_bad_correspondence_checks():
    cal, point = synthetic_pair(-50.)
    point["cam1"][1] += 20.
    result = triangulate_point(point["cam0"], point["cam1"], cal)
    assert not result["valid"]
    assert "high_epipolar" in result["flags"]
    assert any("Epipolar error" in reason and "2 px limit" in reason
               for reason in explain(result))
    assert any("reprojection error" in reason for reason in explain(result))


def test_small_off_curve_click_can_pass():
    cal, point = synthetic_pair(-50.)
    point["cam1"][1] += 1.
    result = triangulate_point(point["cam0"], point["cam1"], cal)
    assert 0 < result["epipolar_sampson_px"] < HIGH_EPIPOLAR_PX
    assert result["valid"]


def test_capture_preview_z_points_toward_known_raised_point():
    cal, raised = synthetic_pair(-50.)
    origin = np.array([40., 176., 0.])
    up = np.asarray(board_up_direction(cal)["up_board"])
    preview = project_board_frame_px(cal, origin_board_mm=origin, up_board=up)
    _, on_plane = synthetic_pair(0.)
    for camera in ("cam0", "cam1"):
        assert preview[camera]["origin"] == pytest.approx(on_plane[camera])
        assert preview[camera]["axes"]["Z"] == pytest.approx(raised[camera])


def test_api_recomputes_reasons_and_does_not_modify_saved_pixels(tmp_path):
    cal, point = synthetic_pair(-50.)
    point["cam1"][1] += 20.
    point["geometry_diagnostics"] = ["stale explanation"]
    point["depth_z_cam0_mm"] = -999.
    pair = tmp_path / "pairs" / "pair_0001_test"
    pair.mkdir(parents=True)
    for cam in (0, 1):
        assert cv2.imwrite(str(pair / f"cam{cam}.png"), np.zeros((720, 1280, 3), np.uint8))
    labels = pair / "correspondences.json"
    labels.write_text(json.dumps({"points": [point]}))
    original = labels.read_bytes()
    client = create_label_web_app(tmp_path, cal, [pair], 900).test_client()
    response = client.get(f"/api/pair/{pair.name}")
    assert response.status_code == 200
    result = response.get_json()["points"][0]
    assert not result["valid"]
    assert "stale explanation" not in result["geometry_diagnostics"]
    assert result["depth_z_cam0_mm"] > 0
    assert result["cam0"] == point["cam0"] and result["cam1"] == point["cam1"]
    assert labels.read_bytes() == original
    saved = client.post(f"/api/pair/{pair.name}/save", json={"points": [point]})
    assert saved.status_code == 200
    persisted = json.loads(labels.read_text())["points"][0]
    assert persisted["geometry_diagnostics"] == result["geometry_diagnostics"]
    assert persisted["depth_z_cam0_mm"] == result["depth_z_cam0_mm"]
    assert persisted["cam0"] == point["cam0"] and persisted["cam1"] == point["cam1"]
    page = client.get("/").get_data(as_text=True)
    assert "Board X, Y, Z↑ (mm)" in page and "Geometry diagnostics" in page
    assert "Assumed board-plane X, Y, Z↑ (mm)" in page
    assert "diagnosticsTd.textContent" in page


@pytest.mark.parametrize("flag", ["low_parallax", "rejected_parallel_rays", "geometry_error", "future_flag"])
def test_warning_parallel_and_unexpected_flags_are_visible(flag):
    reasons = explain({"valid": False, "flags": [flag], "parallax_deg": 0.001})
    assert len(reasons) == 1 and reasons[0]


def test_invalid_diagnostic_limit_is_rejected():
    with pytest.raises(ValueError, match="positive and finite"):
        describe_point_flags({}, reprojection_limit_px=float("nan"),
                             epipolar_limit_px=2., rejected_parallax_deg=0.01)
