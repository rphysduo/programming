"""Cross-camera guides must preserve distortion, direction and independent clicks."""
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from correspondence_guide import predict_correspondence
from validate_stereo_3d import Calibration, create_label_web_app


@pytest.fixture
def cal():
    K0 = np.array([[900., 0., 640.], [0., 910., 360.], [0., 0., 1.]])
    K1 = np.array([[870., 0., 630.], [0., 880., 350.], [0., 0., 1.]])
    D = np.array([-.18, .035, .001, -.002, .001])
    R = cv2.Rodrigues(np.array([.08, -.04, 3.05]))[0]
    T = np.array([3., -35., -2.])
    t0 = np.array([0., 0., 800.])
    return Calibration(paths={}, hashes={}, K0=K0, D0=D.copy(), K1=K1,
                       D1=D.copy(), R=R, T=T, E=None, F=None,
                       size0=(1280, 720), size1=(1280, 720), tank={
                           "R_board_to_cam0": np.eye(3), "t_board_to_cam0": t0,
                           "R_board_to_cam1": R, "t_board_to_cam1": R @ t0 + T,
                           "T_world_board": np.eye(4)})


def project(cal, camera, board):
    R = getattr(cal, f"R_board_to_cam{camera}")
    t = getattr(cal, f"t_board_to_cam{camera}")
    K, D = ((cal.K0, cal.D0), (cal.K1, cal.D1))[camera]
    X = R @ np.asarray(board) + t
    return cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(2)


def distance(raw, segments):
    minimum = float("inf")
    for segment in segments:
        a, b = np.asarray(segment[:-1]), np.asarray(segment[1:])
        delta = b - a
        fraction = np.clip(np.sum((raw - a) * delta, axis=1) /
                           np.maximum(np.sum(delta * delta, axis=1), 1e-30), 0, 1)
        minimum = min(minimum, np.linalg.norm(raw - a - fraction[:, None] * delta, axis=1).min())
    return minimum


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("z", [-300., 0., 400.])
def test_distorted_opposite_camera_curve_contains_true_match(cal, source, z):
    board = [70., 60., z]
    guide = predict_correspondence(cal, source, project(cal, source, board))
    assert guide["source_camera"] == source and guide["target_camera"] == 1 - source
    assert guide["board_plane"] is None
    segments = guide["curve_segments_px"]
    assert segments and distance(project(cal, 1 - source, board), segments) < .005
    # Ensure no undistorted straight line was mistakenly drawn on raw images.
    points = np.concatenate(segments)
    _, singular, _ = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    assert singular[1] > .1
    assert np.all(points >= 0) and np.all(points[:, 0] <= 1279) and np.all(points[:, 1] <= 719)


@pytest.mark.parametrize("source", [0, 1])
def test_unique_plane_prediction_is_optional_and_uses_independent_world_poses(cal, source):
    board = [70., 60., 0.]
    source_px = project(cal, source, board)
    guide = predict_correspondence(cal, source, source_px, board_plane=True)
    plane = guide["board_plane"]
    assert plane["target_px"] == pytest.approx(project(cal, 1 - source, board), abs=1e-7)
    assert plane["board_xyz_mm"] == pytest.approx(board, abs=1e-7)
    assert plane["inside_target_image"]
    # A world/stereo inconsistency must remain visible, not be forced onto the line.
    cal.tank[f"t_board_to_cam{1-source}"] += np.array([10., 10., 0.])
    changed = predict_correspondence(cal, source, source_px, board_plane=True)
    assert changed["curve_segments_px"] == guide["curve_segments_px"]
    assert distance(np.array(changed["board_plane"]["target_px"]), changed["curve_segments_px"]) > 1.


def test_off_plane_point_is_not_forced_onto_board(cal):
    board = [70., 60., -300.]
    guide = predict_correspondence(cal, 0, project(cal, 0, board), board_plane=True)
    actual = project(cal, 1, board)
    assert distance(actual, guide["curve_segments_px"]) < .005
    assert np.linalg.norm(actual - guide["board_plane"]["target_px"]) > 10.


@pytest.mark.parametrize("T", [[-35., 0., 0.], [0., -35., 0.]])
def test_horizontal_and_vertical_curves(cal, T):
    cal.R = np.eye(3)
    cal.T = np.array(T)
    cal.D0[:] = 0
    cal.D1[:] = 0
    source = [700., 400.]
    guide = predict_correspondence(cal, 0, source)
    assert guide["curve_segments_px"]
    points = np.concatenate(guide["curve_segments_px"])
    axis = 1 if T[0] else 0
    assert np.ptp(points[:, axis]) < 1e-8


def test_degenerate_geometry_and_unusable_plane_are_explained(cal):
    cal.T[:] = 0.
    guide = predict_correspondence(cal, 0, [640., 360.])
    assert not guide["curve_segments_px"] and "undefined" in guide["warnings"][0]
    cal.tank["R_board_to_cam0"] = cv2.Rodrigues(np.array([np.pi / 2, 0., 0.]))[0]
    guide = predict_correspondence(cal, 0, [640., 360.], board_plane=True)
    assert guide["board_plane"] is None
    assert any("parallel" in warning for warning in guide["warnings"])


@pytest.mark.parametrize("camera,pixel,plane", [
    (2, [10., 10.], False), (True, [10., 10.], False),
    (0, [float("nan"), 10.], False), (0, [-1., 10.], False),
    (0, [1280., 10.], False), (1, [10.], False),
    (None, [10., 10.], False), (0, None, False), (0, [10., 10.], "yes")])
def test_invalid_input_is_rejected(cal, camera, pixel, plane):
    with pytest.raises(ValueError):
        predict_correspondence(cal, camera, pixel, board_plane=plane)


def test_api_is_read_only_and_save_retains_off_curve_raw_clicks(cal, tmp_path, monkeypatch):
    pair = tmp_path / "pairs" / "pair_0001_test"
    pair.mkdir(parents=True)
    for camera in (0, 1):
        assert cv2.imwrite(str(pair / f"cam{camera}.png"), np.zeros((720, 1280, 3), np.uint8))
    point = {"id": 0, "name": "A", "cam0": project(cal, 0, [70., 60., 0.]).tolist(),
             "cam1": (project(cal, 1, [70., 60., 0.]) + [20., 20.]).tolist()}
    labels = pair / "correspondences.json"
    labels.write_text(json.dumps({"points": [point]}))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original_cal = deepcopy(cal)
    monkeypatch.chdir(tmp_path)
    client = create_label_web_app(Path("."), cal, [pair.relative_to(tmp_path)], 900).test_client()
    assert client.get(f"/pair/{pair.name}/cam0.png").status_code == 200
    for source in (0, 1):
        result = client.post(f"/api/pair/{pair.name}/guide", json={
            "source_camera": source, "source_px": point[f"cam{source}"], "board_plane": True})
        assert result.status_code == 200 and result.get_json()["curve_segments_px"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for field in ("K0", "D0", "K1", "D1", "R", "T"):
        np.testing.assert_array_equal(getattr(cal, field), getattr(original_cal, field))
    saved = client.post(f"/api/pair/{pair.name}/save", json={"points": [point]})
    assert saved.status_code == 200
    result = saved.get_json()["points"][0]
    assert result["cam0"] == point["cam0"] and result["cam1"] == point["cam1"]
    assert not result["valid"] and "high_reproj_cam0" in result["flags"]
    assert client.post(f"/api/pair/{pair.name}/guide", json={}).status_code == 400
    assert client.post("/api/pair/unknown/guide", json={}).status_code == 404
    # Refuse rescaled images rather than drawing a misleading guide.
    cv2.imwrite(str(pair / "cam1.png"), np.zeros((360, 640, 3), np.uint8))
    result = client.post(f"/api/pair/{pair.name}/guide", json={"source_camera": 0, "source_px": point["cam0"]})
    assert result.status_code == 400 and "sizes differ" in result.get_json()["error"]
