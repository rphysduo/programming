"""Regression tests for calibration capture guards; never touch active NPZs."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

CAMERA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMERA_DIR))

from charuco_common import get_board
from calibration_capture import (CaptureDetector, averaged_stereo_points, load_capture_rois,
                                 save_candidate_npz, unique_corners, world_samples_ready,
                                 require_image_size)
from calibrate_world import solve_camera_world, combine_camera_world, pack_world_params
import web_calibrate as web
from validate_stereo_3d import load_calibration, create_label_web_app


def world_observations(ids=None):
    spec = get_board("long")
    obj = spec.board.getChessboardCorners()
    ids = np.arange(18) if ids is None else ids
    K = np.array([[1400., 0., 960.], [0., 1400., 540.], [0., 0., 1.]])
    D = np.zeros(5)
    R = cv2.Rodrigues(np.array([0.06, -0.03, 0.02]))[0] @ np.diag([1., -1., -1.])
    rvec = cv2.Rodrigues(R)[0]
    t = np.array([-80., 400., 1000.])
    corners = cv2.projectPoints(obj[ids], rvec, t, K, D)[0]
    return spec, obj, ids.reshape(-1, 1), corners, K, D, rvec, t


def solve(data, **kwargs):
    spec, obj, ids, corners, K, D, _, _ = data
    return solve_camera_world(corners, ids, K, D, spec.board, obj, np.eye(4), **kwargs)


def test_good_pose_and_stationary_duplicate_frames_still_work():
    data = world_observations()
    pose = solve(data)
    assert pose["ok"]
    assert pose["rms"] < 1e-3
    combined = combine_camera_world([pose]*10, data[4], data[5], np.eye(4))
    assert combined["ok"] and combined["frames"] == 10
    assert combined["center_world"] == pytest.approx(pose["center_world"], abs=1e-4)


@pytest.mark.parametrize("case", ["collinear", "nonfinite", "outlier", "duplicate_ids", "bad_ids"])
def test_invalid_world_inputs_rejected(case):
    data = list(world_observations(np.arange(0, 18, 3) if case == "collinear" else None))
    if case == "nonfinite":
        data[3][0, 0, 0] = np.nan
    elif case == "outlier":
        data[3][4, 0] += [60., -40.]
    elif case == "duplicate_ids":
        data[2][1] = data[2][0]
    elif case == "bad_ids":
        data[2][0] = -1
    assert not solve(data)["ok"]


@pytest.mark.parametrize("case", ["nan_pose", "behind_camera"])
def test_bad_opencv_pose_cannot_be_accepted_or_saved(monkeypatch, case):
    data = world_observations()
    r, t = data[-2:]
    if case == "nan_pose":
        t = np.full(3, np.nan)
    else:
        t = np.array([0., 0., -1000.])
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (True, r, t))
    assert not solve(data)["ok"]


def test_pack_refuses_nonfinite_pose():
    data = world_observations()
    pose = solve(data)
    pose["center_world"] = np.array([0., 0., np.nan])
    with pytest.raises(ValueError, match="nonfinite"):
        pack_world_params({"cam0": pose, "cam1": pose}, np.eye(4), data[0], (1920, 1080))


def test_headless_stereo_records_one_entry_per_shared_id():
    board = get_board("small").board.getChessboardCorners()
    capture = {(cam, cid): (100. + cid, 200. + cam) for cam in (0, 1) for cid in range(10)}
    capture[(0, 11)] = (0, 0)  # asymmetric flicker is not a correspondence
    obj, p0, p1 = averaged_stereo_points(capture, board)
    assert obj.shape == (10, 3)
    assert p0.shape == p1.shape == (10, 2)
    assert obj == pytest.approx(board[:10])


def test_duplicate_ids_exclude_all_ambiguous_observations():
    corners, ids = unique_corners(np.array([[10., 10.], [110., 110.], [20., 20.]]),
                                 np.array([3, 3, 4]), 35)
    assert ids.reshape(-1).tolist() == [4]
    assert corners.reshape(-1, 2).tolist() == [[20., 20.]]


def test_capture_roi_masks_second_coherent_board_without_shifting_pixels():
    spec = get_board("small")
    image = spec.board.generateImage((400, 560), marginSize=20)
    combined = np.hstack([image, image])
    detector = CaptureDetector(spec.detector, spec.board, [[400, 0], [799, 0], [799, 559], [400, 559]])
    corners, ids, _, _ = detector.detectBoard(combined)
    assert len(ids) == 35
    assert np.all(corners[:, 0, 0] >= 400)


def test_reflection_policy_requires_explicit_choice():
    with pytest.raises(ValueError, match="DIRECT"):
        load_capture_rois(SimpleNamespace(roi_file=None, allow_full_frame=False), ("cam0", "cam1"))
    assert load_capture_rois(SimpleNamespace(roi_file=None, allow_full_frame=True), ("camera",)) == {"camera": None}


def test_candidate_writes_cannot_replace_files_and_reject_nan(tmp_path):
    path = tmp_path / "active.npz"
    path.write_bytes(b"original calibration bytes")
    with pytest.raises(FileExistsError):
        save_candidate_npz(path, K=np.eye(3))
    assert path.read_bytes() == b"original calibration bytes"
    candidate = tmp_path / "candidate.npz"
    with pytest.raises(ValueError, match="nonfinite"):
        save_candidate_npz(candidate, T=np.array([np.nan, 0, 0]))
    assert not candidate.exists()
    save_candidate_npz(candidate, K=np.eye(3))
    with np.load(candidate) as saved:
        assert saved["K"] == pytest.approx(np.eye(3))


def test_actual_resolution_must_match_intrinsics():
    with pytest.raises(ValueError, match="differs"):
        require_image_size((1280, 720), (1920, 1080), "cam1")


def make_runner(tmp_path, samples=2):
    args = web.build_arg_parser().parse_args([
        "--mode", "world", "--allow-full-frame", "--samples", str(samples),
        "--world-output", str(tmp_path / "candidate.npz"),
        "--param0", str(CAMERA_DIR / "cam0_intrinsics.npz"),
        "--param1", str(CAMERA_DIR / "cam1_intrinsics.npz")])
    state = web.CalibState("world", samples, ("cam0", "cam1"))
    return web.CalibrationRunner(args, state, {}, get_board("long"))


def test_world_loop_does_not_stop_at_ten_and_one(monkeypatch, tmp_path):
    runner = make_runner(tmp_path)
    class Cap:
        def __init__(self): self.count = 0; self.closed = False
        def read(self):
            self.count += 1
            assert self.count <= 3, "world capture did not stop after both targets"
            return True, np.zeros((8, 8, 3), np.uint8)
        def release(self): self.closed = True
    caps = {k: Cap() for k in ("cam0", "cam1")}
    monkeypatch.setattr(runner, "_open", lambda key, source: caps[key])
    runner.capture_event = SimpleNamespace(is_set=lambda: True, clear=lambda: None)
    for detector in runner.detectors.values():
        monkeypatch.setattr(detector, "detectBoard", lambda gray: (None, None, None, None))
    outcomes = iter([True, False, True, True, True])
    monkeypatch.setattr(web, "solve_camera_world", lambda *a: {"ok": next(outcomes)})
    monkeypatch.setattr(runner, "_emit", lambda *a: None)
    finished = []
    monkeypatch.setattr(runner, "_finish_world", lambda captures, *a: finished.append(captures))
    runner._run_world()
    assert [len(finished[0][k]) for k in ("cam0", "cam1")] == [2, 2]
    assert runner.state.captured == 2
    assert all(cap.count == 3 and cap.closed for cap in caps.values())
    assert not world_samples_ready({"cam0": [None]*10, "cam1": [None]}, 10)


def test_finishing_world_early_cannot_save_partial_result(tmp_path):
    runner = make_runner(tmp_path, samples=10)
    runner._finish_world({"cam0": [{}]*10, "cam1": [{}]}, np.eye(4), 0, {}, {})
    assert "EACH camera" in runner.state.result[0]
    assert not Path(runner.args.world_output).exists()


def test_label_ui_discloses_active_old_snapshot_and_hashes_without_mutation():
    root = CAMERA_DIR
    session = root / "validation"
    names = ("cam0_intrinsics.npz", "cam1_intrinsics.npz", "stereo_params.npz", "tank_alignment.npz")
    paths = {n: session / "calibration" / n for n in names}
    protected = list(paths.values()) + [root / n for n in names]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected]
    cal = load_calibration(paths)
    pair = session / "pairs" / "pair_0018_20261029_142637"
    app = create_label_web_app(session, cal, [pair], 900)
    client = app.test_client()
    page = client.get("/").get_data(as_text=True)
    assert "35.9127" in page and "WARNING: different source calibration" in page
    info = client.get("/api/calibration").get_json()
    assert set(info["different_source_files"]) == {"stereo_params.npz", "tank_alignment.npz"}
    assert info["files"]["stereo_params.npz"]["loaded_sha256"] == cal.hashes["stereo_params.npz"]
    assert client.get(f"/api/pair/{pair.name}").get_json()["calibration"] == info
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected]
