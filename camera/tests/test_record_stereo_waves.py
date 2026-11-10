"""Exercise the recorder and labeler with synthetic camera streams."""
import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import record_stereo_waves as recorder
from validate_stereo_3d import create_label_web_app, list_pair_dirs, read_json


@pytest.fixture
def calib_dir(tmp_path):
    path = tmp_path / "calibration"
    path.mkdir()
    K = np.array([[90., 0., 32.], [0., 90., 24.], [0., 0., 1.]])
    D = np.zeros(5)
    R = np.eye(3)
    T = np.array([-100., 0., 0.])
    for name in ("cam0", "cam1"):
        np.savez(path / f"{name}_intrinsics.npz", camera_matrix=K,
                 dist_coeffs=D, image_size=[64, 48])
    np.savez(path / "stereo_params.npz", K0=K, K1=K, D0=D, D1=D, R=R, T=T)
    np.savez(path / "tank_alignment.npz", R_board_to_cam0=R, R_board_to_cam1=R,
             t_board_to_cam0=[0., 0., 500.], t_board_to_cam1=T + [0., 0., 500.],
             T_world_board=np.eye(4))
    return path


class FakeCamera:
    def __init__(self, value):
        self.value = value
        self.released = False

    def read(self):
        time.sleep(0.01)
        return True, np.full((48, 64, 3), self.value, np.uint8)

    def get(self, prop):
        return 30.0 if prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOCUS) else 0.0

    def release(self):
        self.released = True


def test_record_export_and_open_in_labeler(tmp_path, calib_dir, monkeypatch):
    session = tmp_path / "trial"
    cameras = [FakeCamera(30), FakeCamera(180)]
    monkeypatch.setattr(recorder, "open_camera", lambda source, *a, **k: (cameras[int(source)], (64, 48)))
    record = ["record", "--session", str(session), "--calib-dir", str(calib_dir), "--seconds", "0.35"]
    assert recorder.main(record) == 0
    assert all(camera.released for camera in cameras)
    assert recorder.main(record) == 1  # A second run must preserve the videos and labels.
    video = session / "video"
    info = read_json(video / "recording.json")
    assert all(info["streams"][cam]["frames"] > 1 for cam in recorder.CAMERAS)

    export = ["export", "--session", str(session), "--at", "0.05", "0.15", "--max-gap-ms", "50"]
    assert recorder.main(export) == 0
    pairs = list_pair_dirs(session)
    assert len(pairs) == 2
    assert recorder.main(export) == 0
    assert list_pair_dirs(session) == pairs
    pair = pairs[0]
    meta = read_json(pair / "pair.json")
    assert meta["calibration_hashes"] == info["calibration_hashes"]
    for camera, expected in zip(recorder.CAMERAS, (30, 180)):
        frame = cv2.imread(str(pair / f"{camera}.png"))
        assert frame.shape == (48, 64, 3)
        assert float(frame.mean()) == pytest.approx(expected, abs=2)

    calibration = recorder.load_calibration({name: session / "calibration" / name
                                             for name in recorder.CALIB_NAMES})
    client = create_label_web_app(session, calibration, pairs, 900).test_client()
    response = client.get(f"/api/pair/{pair.name}")
    assert response.status_code == 200
    assert response.get_json()["sizes"] == {"cam0": [64, 48], "cam1": [64, 48]}

    # Model a camera delayed by 200 ms. Export must decline an early request.
    path = video / "cam1_frames.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("elapsed_ns", "read_timestamp_ns"):
            row[key] = int(row[key]) + 200_000_000
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=recorder.TIMESTAMP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert recorder.main(["export", "--session", str(session), "--at", "0"]) == 1
    assert list_pair_dirs(session) == pairs
