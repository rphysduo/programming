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
import validate_stereo_3d as labeler
from validate_stereo_3d import create_label_web_app, list_pair_dirs, read_json
from wave_recording_web import WaveRecorder, create_app


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


def test_timed_runs_create_distinct_default_sessions(tmp_path, calib_dir, monkeypatch):
    output = tmp_path / "outputs"
    monkeypatch.setattr(recorder, "DEFAULT_OUTPUT_DIR", output)
    monkeypatch.setattr(recorder, "open_camera",
                        lambda source, *a, **k: (FakeCamera(40), (64, 48)))
    command = ["record", "--calib-dir", str(calib_dir), "--seconds", "0.2"]
    assert recorder.main(command) == 0
    first, = output.glob("wave_*")
    original = (first / "video/recording.json").read_bytes()
    assert recorder.main(command) == 0
    sessions = set(output.glob("wave_*"))
    assert len(sessions) == 2
    second, = sessions - {first}
    assert (second / "video/recording.json").is_file()
    assert (first / "video/recording.json").read_bytes() == original


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


def test_browser_can_preview_record_stop_and_start_another_trial(tmp_path, calib_dir):
    sources = {name: calib_dir / name for name in recorder.CALIB_NAMES}
    cal = recorder.load_calibration(sources)
    with WaveRecorder(tmp_path / "trials", sources, cal,
                      [FakeCamera(40), FakeCamera(150)], (cal.size0, cal.size1),
                      (30, 30), {}) as live:
        client = create_app(live).test_client()
        deadline = time.monotonic() + 3
        while not client.get("/status").get_json()["ready"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.get("/status").get_json()["ready"]

        def check_preview(camera, value):
            with client.get(f"/video/{camera}", buffered=False) as response:
                assert response.mimetype == "multipart/x-mixed-replace"
                chunk = next(iter(response.response))
                jpeg = chunk.split(b"\r\n\r\n", 1)[1][:-2]
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                assert image.mean() == pytest.approx(value, abs=2)

        check_preview("cam0", 40)
        check_preview("cam1", 150)

        sessions = []
        for _ in range(2):
            started = client.post("/recording/start", json={})
            assert started.status_code == 200
            session = Path(started.get_json()["session"])
            assert client.post("/recording/start", json={}).status_code == 409
            assert client.get("/status").get_json()["active"] == str(session)
            assert not (session / "video/recording.json").exists()
            check_preview("cam1", 150)
            time.sleep(0.15)
            stopped = client.post("/recording/stop", json={})
            assert stopped.status_code == 200
            status = client.get("/status").get_json()
            assert status["ready"] and status["active"] is None
            assert status["last"] == stopped.get_json()
            check_preview("cam0", 40)
            sessions.append(session)

        assert sessions[0] != sessions[1]
        for session in sessions:
            assert recorder.main(["export", "--session", str(session), "--at", "0",
                                  "--max-gap-ms", "50"]) == 0
            assert len(list_pair_dirs(session)) == 1


def test_full_export_covers_both_streams_and_labeler_steps_in_time_order(
        tmp_path, calib_dir, monkeypatch):
    session = tmp_path / "trial"
    paths = {name: calib_dir / name for name in recorder.CALIB_NAMES}
    cal = recorder.load_calibration(recorder.ensure_session(session, paths, quiet=True))
    video = session / "video"
    video.mkdir()
    times = ((0, 60, 120), (2, 30, 62, 170))
    epoch = 1_700_000_000_000_000_000
    streams = {}
    for camera, elapsed_ms in zip(recorder.CAMERAS, times):
        with recorder.ExitStack() as resources:
            stream = recorder._RecordingStream(resources, video, camera, (64, 48), 30)
            for i, ms in enumerate(elapsed_ms):
                frame = np.full((48, 64, 3), 40 + i * 30, np.uint8)
                stream.write(frame, ms * 1_000_000, 0, epoch)
        streams[camera] = stream.summary()
    recorder._write_recording_info(video, cal, {}, streams, epoch)

    command = ["export", "--session", str(session)]
    assert recorder.main(command + ["--at", "0.06"]) == 0
    original = list_pair_dirs(session)[0]
    labels = original / "correspondences.json"
    labels.write_text('{"points": []}', encoding="utf-8")
    assert recorder.main(command + ["--all"]) == 0
    assert recorder.main(command + ["--all"]) == 0
    assert labels.read_text(encoding="utf-8") == '{"points": []}'

    pairs = {}
    for path in list_pair_dirs(session):
        meta = read_json(path / "pair.json")
        indices = tuple(meta["video_frame_indices"])
        assert indices not in pairs
        pairs[indices] = path
        for camera, index in zip(recorder.CAMERAS, indices):
            frame = cv2.imread(str(path / f"{camera}.png"))
            assert frame.mean() == pytest.approx(40 + index * 30, abs=2)
    assert set(pairs) == {(0, 0), (0, 1), (1, 2), (2, 3)}
    wide = read_json(pairs[(2, 3)] / "pair.json")
    assert wide["video_read_gap_exceeds_limit"]
    assert wide["inter_camera_read_gap_ms"] == pytest.approx(50)

    run = []

    def check_app(session, cal, ordered, width):
        client = create_label_web_app(session, cal, ordered, width).test_client()
        expected = [pairs[key].name for key in ((0, 0), (0, 1), (1, 2), (2, 3))]
        assert client.get("/api/pairs").get_json()["pairs"] == expected
        warnings = client.get(f"/api/pair/{expected[-1]}").get_json()["warnings"]
        assert any("Large timing gap" in message for message in warnings)
        return type("App", (), {"run": lambda self, **kw: run.append(True)})()

    monkeypatch.setattr(labeler, "create_label_web_app", check_app)
    assert labeler.main(["label", "--session", str(session), "--calib-dir",
                         str(session / "calibration")]) == 0
    assert run
