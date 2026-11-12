"""Synthetic recorded trial shared by offline wave analysis tests."""
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_stereo_3d import CALIB_NAMES, sha256_file


@pytest.fixture
def wave_session(tmp_path):
    session = tmp_path / 'source'
    video = session / 'video'
    calibration = session / 'calibration'
    video.mkdir(parents=True)
    calibration.mkdir()
    K = np.array([[90., 0., 32.], [0., 90., 24.], [0., 0., 1.]])
    D = np.zeros(5)
    R = np.eye(3)
    T = np.array([-10., 0., 0.])
    E = np.array([[0., 0., 0.], [0., 0., 10.], [0., -10., 0.]])
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    for camera in ('cam0', 'cam1'):
        np.savez(calibration / f'{camera}_intrinsics.npz', camera_matrix=K, dist_coeffs=D, image_size=[64, 48])
    np.savez(calibration / 'stereo_params.npz', K0=K, K1=K, D0=D, D1=D, R=R, T=T, E=E, F=F)
    np.savez(calibration / 'tank_alignment.npz', R_board_to_cam0=R, R_board_to_cam1=R,
             t_board_to_cam0=[0., 0., 500.], t_board_to_cam1=T+[0., 0., 500.], T_world_board=np.eye(4))
    hashes = {name: sha256_file(calibration/name) for name in CALIB_NAMES}
    (calibration/'manifest.json').write_text(json.dumps({'files': {k: {'sha256': v} for k,v in hashes.items()}}))
    (session/'session.json').write_text('{}')
    start = 1794337340444523765
    streams = {}
    times = {'cam0': [0, 110_000_000, 250_000_000], 'cam1': [4_000_000, 100_000_000, 180_000_000, 270_000_000]}
    for camera, stamps in times.items():
        writer = cv2.VideoWriter(str(video/f'{camera}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64,48))
        assert writer.isOpened()
        for i in range(len(stamps)):
            writer.write(np.full((48,64,3), 30+i*30, np.uint8))
        writer.release()
        with (video/f'{camera}_frames.csv').open('w',newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['frame_index','elapsed_ns','read_timestamp_ns'])
            csv_writer.writerows((i,t,start+t) for i,t in enumerate(stamps))
        streams[camera] = {'frames':len(stamps), 'video_header_fps':30}
    (video/'recording.json').write_text(json.dumps(dict(start_unix_ns=start, streams=streams,
        camera_settings={'cam0':'0','cam1':'1'},calibration_hashes=hashes)))
    return session
