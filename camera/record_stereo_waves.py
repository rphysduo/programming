#!/usr/bin/env python3
"""Record stereo wave videos and export frame pairs for validate_stereo_3d."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack, closing, contextmanager
import csv
import math
from pathlib import Path
import sys
from threading import Event
import time

import cv2

from charuco_common import open_camera, parse_camera_source
from validate_stereo_3d import (
    CALIB_NAMES,
    _capture_actual_settings,
    _save_pair,
    ensure_session,
    list_pair_dirs,
    load_calibration,
    read_json,
    write_json,
)


CAMERAS = ("cam0", "cam1")
NS_PER_SECOND = 1_000_000_000
TIMESTAMP_COLUMNS = ("frame_index", "elapsed_ns", "read_timestamp_ns")
TIMING_WARNING = (
    "Wave frames are paired by host read time. The C920 cameras are not hardware "
    "synchronized. The displayed gap does not measure exposure-time skew."
)


def _read_calibration(paths):
    calibration = load_calibration(paths)
    if calibration.errors:
        raise ValueError("Calibration errors: " + "; ".join(
            item["message"] for item in calibration.errors))
    return calibration


def _check_frame(frame, size, camera):
    if frame.shape != (size[1], size[0], 3):
        raise ValueError(f"{camera} frame shape {frame.shape} differs from calibrated size {size}")


class _RecordingStream:
    """Write frames and timestamps in the same order for both capture modes."""

    def __init__(self, resources, video_dir, camera, size, fps):
        self.camera = camera
        self.size = size
        self.fps = fps
        self.count = 0
        self.first = self.last = None
        path = video_dir / f"{camera}.avi"
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        resources.callback(self.writer.release)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open the Motion JPEG writer for {path}")
        log = resources.enter_context((video_dir / f"{camera}_frames.csv").open(
            "x", newline="", encoding="utf-8"))
        self.rows = csv.writer(log)
        self.rows.writerow(TIMESTAMP_COLUMNS)

    def write(self, frame, now, start, start_unix):
        _check_frame(frame, self.size, self.camera)
        self.writer.write(frame)
        self.rows.writerow((self.count, now - start, start_unix + now - start))
        if self.first is None:
            self.first = now
        self.last = now
        self.count += 1

    def summary(self):
        if not self.count:
            raise RuntimeError(f"{self.camera} recorded zero frames. Start a new recording and try again.")
        observed = None
        if self.count > 1 and self.last > self.first:
            observed = (self.count - 1) * NS_PER_SECOND / (self.last - self.first)
        return {"frames": self.count, "video_header_fps": self.fps, "observed_fps": observed}


def _record_camera(cap, video_dir, camera, size, fps, start, start_unix, end, stop):
    """Drain warm-up frames and write frames in the timed capture interval."""
    with ExitStack() as resources:
        stream = _RecordingStream(resources, video_dir, camera, size, fps)
        while not stop.is_set():
            ok, frame = cap.read()
            now = time.monotonic_ns()
            if not ok:
                raise RuntimeError(f"{camera} stopped delivering frames")
            if now >= end or stop.is_set():
                break
            if now >= start:
                stream.write(frame, now, start, start_unix)
    return stream.summary()


@contextmanager
def _open_recording_cameras(args, cal):
    """Open both cameras at their calibrated sizes and report video settings."""
    if parse_camera_source(args.cam0) == parse_camera_source(args.cam1):
        raise ValueError("Choose two different camera devices.")
    sizes = (cal.size0, cal.size1)
    camera_settings = {"cam0": args.cam0, "cam1": args.cam1,
                       "requested_fps": args.fps, "requested_focus": args.focus,
                       "requested_exposure": args.exposure}
    with ExitStack() as resources:
        caps = []
        for camera, source, size in zip(CAMERAS, (args.cam0, args.cam1), sizes):
            cap, actual = open_camera(source, *size, args.fps,
                                      focus_val=args.focus, exposure_val=args.exposure)
            resources.callback(cap.release)
            if tuple(actual) != size:
                raise ValueError(f"{camera} opened at {actual}. Calibration requires {size}.")
            caps.append(cap)
        settings = _capture_actual_settings(*caps, *sizes)
        header_rates = []
        for camera in CAMERAS:
            reported = settings[f"actual_fps_{camera}"]
            header_rates.append(reported if math.isfinite(reported) and reported > 0 else args.fps)
        yield caps, sizes, header_rates, {**camera_settings, **settings}


def _write_recording_info(video_dir, cal, settings, streams, start_unix, *, seconds=None,
                          interrupted=False):
    """Publish the metadata only after both videos and timestamp files are closed."""
    write_json(video_dir / "recording.json", {
        "start_unix_ns": start_unix, "requested_seconds": seconds,
        "interrupted": interrupted, "streams": streams,
        "camera_settings": settings, "calibration_hashes": cal.hashes,
        "timing_warning": TIMING_WARNING,
    })


def record(args):
    """Capture two videos and preserve the calibration and acquisition metadata."""
    sources = {name: args.calib_dir / name for name in CALIB_NAMES}
    _read_calibration(sources)
    args.session.mkdir(parents=True, exist_ok=False)
    cal = _read_calibration(ensure_session(args.session, sources))
    video_dir = args.session / "video"
    video_dir.mkdir()
    interrupted = False
    with _open_recording_cameras(args, cal) as (caps, sizes, header_rates, settings):
        stop = Event()
        warmup_ns = NS_PER_SECOND
        start = time.monotonic_ns() + warmup_ns
        start_unix = time.time_ns() + warmup_ns
        end = start + round(args.seconds * NS_PER_SECOND)
        print(f"Recording for {args.seconds:g} seconds after a 1-second warm-up. "
              "Press Ctrl+C to finish early.", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(_record_camera, cap, video_dir, camera, size, fps,
                                start, start_unix, end, stop)
                    for cap, camera, size, fps in zip(caps, CAMERAS, sizes, header_rates)]
            try:
                wait(jobs, return_when=FIRST_EXCEPTION)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                stop.set()
            streams = {name: job.result() for name, job in zip(CAMERAS, jobs)}

    _write_recording_info(video_dir, cal, settings, streams, start_unix,
                          seconds=args.seconds, interrupted=interrupted)
    for camera, stream in streams.items():
        rate = stream["observed_fps"]
        observed = f"{rate:.1f} fps" if rate is not None else "rate unavailable"
        print(f"{camera}: {stream['frames']} frames, {observed} observed")
    print(f"Saved both videos and timestamps to {video_dir}")
    return 0


def _read_timestamps(path, expected_count):
    with path.open(newline="", encoding="utf-8") as file:
        rows = [(int(row["frame_index"]), int(row["elapsed_ns"]), int(row["read_timestamp_ns"]))
                for row in csv.DictReader(file)]
    if not rows or len(rows) != expected_count or any(i != row[0] for i, row in enumerate(rows)):
        raise ValueError(f"Frame indices or frame count are inconsistent in {path}")
    if rows[0][1] < 0 or any(a[1] >= b[1] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Invalid elapsed timestamps in {path}")
    return rows


def _nearest(times, target):
    index = bisect_left(times, target)
    if index == 0:
        return 0
    if index == len(times) or target - times[index - 1] <= times[index] - target:
        return index - 1
    return index


def _selected_frames(path, indices):
    """Decode in order so indices agree with the CSV even without accurate seeking."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not read {path}")
        current = -1
        for index in indices:
            while current < index:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{path} ended before frame {index}")
                current += 1
            yield frame
    finally:
        cap.release()


def export(args):
    """Select timestamp-matched frames and save ordinary stereo labeler pairs."""
    video_dir = args.session / "video"
    recording = read_json(video_dir / "recording.json")
    cal = _read_calibration({name: args.session / "calibration" / name for name in CALIB_NAMES})
    if cal.hashes != recording["calibration_hashes"]:
        raise ValueError("The session calibration files differ from those used during recording.")
    logs = [_read_timestamps(video_dir / f"{cam}_frames.csv", recording["streams"][cam]["frames"])
            for cam in CAMERAS]
    times0, times1 = [[row[1] for row in log] for log in logs]
    latest = min(times0[-1], times1[-1]) / NS_PER_SECOND
    if max(args.at) > latest:
        raise ValueError(f"Requested time exceeds the shared recording. Choose times from 0 to {latest:.3f} s.")

    seen = set()
    for path in list_pair_dirs(args.session):
        meta = path / "pair.json"
        if meta.is_file():
            indices = read_json(meta).get("video_frame_indices")
            if indices is not None:
                seen.add(tuple(indices))
    selected = []
    rejected = False
    for seconds in sorted(args.at):
        i = _nearest(times0, round(seconds * NS_PER_SECOND))
        j = _nearest(times1, times0[i])
        gap_ms = abs(times0[i] - times1[j]) / 1e6
        if gap_ms > args.max_gap_ms:
            print(f"Skipped {seconds:g} s: closest read-time gap {gap_ms:.2f} ms "
                  f"exceeds {args.max_gap_ms:g} ms.", file=sys.stderr)
            rejected = True
        elif (i, j) not in seen:
            seen.add((i, j))
            selected.append((seconds, i, j, gap_ms))

    if selected:
        with (closing(_selected_frames(video_dir / "cam0.avi", [s[1] for s in selected])) as frames0,
              closing(_selected_frames(video_dir / "cam1.avi", [s[2] for s in selected])) as frames1):
            for (seconds, i, j, gap_ms), frame0, frame1 in zip(selected, frames0, frames1):
                _check_frame(frame0, cal.size0, "cam0")
                _check_frame(frame1, cal.size1, "cam1")
                read_times = [logs[0][i][2], logs[1][j][2]]
                path = _save_pair(args.session, frame0, frame1, {
                    "warning": TIMING_WARNING,
                    "timestamp_unix": sum(read_times) / (2 * NS_PER_SECOND),
                    "calibration_hashes": cal.hashes,
                    "camera_settings": recording["camera_settings"],
                    "video_frame_indices": [i, j],
                    "video_requested_seconds": seconds,
                    "video_frame_elapsed_seconds": [times0[i] / NS_PER_SECOND, times1[j] / NS_PER_SECOND],
                }, read_times)
                print(f"Saved {path.name} for {seconds:g} s (frames {i}/{j}, read gap {gap_ms:.2f} ms)")
    else:
        print("No new pairs exported.")
    return int(rejected)


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _elapsed(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite time of zero seconds or greater")
    return number


def _camera_args(parser):
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--cam0", default="0")
    parser.add_argument("--cam1", default="1")
    parser.add_argument("--fps", type=_positive, default=30, help="Requested camera fps (default 30)")
    parser.add_argument("--focus", type=int, default=30)
    parser.add_argument("--exposure", type=int)


def _serve_web(args):
    from wave_recording_web import serve

    return serve(args)


def main(argv=None):
    """Run live or timed capture, or export selected moments from a recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record", help="Record videos into a new session")
    capture.add_argument("--session", type=Path, required=True)
    _camera_args(capture)
    capture.add_argument("--seconds", type=_positive, default=15, help="Recording duration (default 15 s)")
    capture.set_defaults(run=record)

    web = commands.add_parser("web", help="Preview both cameras and start/stop recording in a browser")
    _camera_args(web)
    web.add_argument("--output-dir", type=Path, default=Path("../outputs/wave_trials"))
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=5004)
    web.set_defaults(run=_serve_web)

    pairs = commands.add_parser("export", help="Extract pairs at elapsed times in seconds")
    pairs.add_argument("--session", type=Path, required=True)
    pairs.add_argument("--at", nargs="+", type=_elapsed, required=True)
    pairs.add_argument("--max-gap-ms", type=_positive, default=18,
                       help="Maximum host read-time gap (default 18 ms)")
    pairs.set_defaults(run=export)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, RuntimeError, ValueError, KeyError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
