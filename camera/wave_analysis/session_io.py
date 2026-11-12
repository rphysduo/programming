"""Read recorded trials with original camera ordering, pixels, and host times."""
from collections import Counter
import csv
from pathlib import Path

import cv2
import numpy as np

from charuco_geometry_diagnostic import compare_relative_geometry
from record_stereo_waves import CAMERAS, TIMESTAMP_COLUMNS, _all_frame_pairs
from validate_stereo_3d import CALIB_NAMES, load_calibration, read_json, sha256_file


def read_timestamps(path):
    """Return integer nanosecond records and explicit integrity findings."""
    rows, issues = [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set(TIMESTAMP_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing timestamp columns in {path}')
        for line, row in enumerate(reader, 2):
            try:
                rows.append({key: int(row[key]) for key in TIMESTAMP_COLUMNS})
            except (TypeError, ValueError) as exc:
                issues.append(f'Malformed timestamp at line {line}: {exc}')
    if not rows:
        issues.append('No usable timestamps')
        return rows, issues
    indices = [r['frame_index'] for r in rows]
    if indices != list(range(len(rows))):
        issues.append('Frame indices are missing, duplicated, negative, or out of order')
    for key in ('elapsed_ns', 'read_timestamp_ns'):
        values = [r[key] for r in rows]
        if any(v < 0 for v in values) or any(b <= a for a, b in zip(values, values[1:])):
            issues.append(f'{key} must be nonnegative and strictly increasing')
    offsets = {r['read_timestamp_ns'] - r['elapsed_ns'] for r in rows}
    if len(offsets) > 1:
        issues.append('Read and elapsed timestamps disagree about the session origin')
    return rows, issues


def distribution(values):
    """Describe finite scalar observations, returning nulls for absent data."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(count=0, min=None, median=None, p95=None, max=None)
    return dict(count=int(values.size), min=float(values.min()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


def timestamp_summary(rows):
    """Calculate intervals from integer differences before conversion to seconds."""
    intervals = [(b['elapsed_ns'] - a['elapsed_ns']) / 1e9 for a, b in zip(rows, rows[1:])]
    duration = (rows[-1]['elapsed_ns'] - rows[0]['elapsed_ns']) / 1e9 if len(rows) > 1 else 0
    counts = Counter(r['frame_index'] for r in rows)
    return dict(logged_frames=len(rows), observed_fps=(len(rows)-1)/duration if duration > 0 else None,
                elapsed_span_seconds=duration, intervals_ms=distribution(np.array(intervals)*1000),
                duplicate_frame_indices=[i for i, n in counts.items() if n > 1],
                missing_frame_indices=[i for i in range(max(counts, default=-1)+1) if i not in counts]
                if max(counts, default=0) < 1_000_000 else None,
                duplicate_elapsed_timestamps=len(rows)-len({r['elapsed_ns'] for r in rows}),
                exposure_timestamp_uncertainty_seconds=None)


def file_identity(path):
    """Fingerprint a source file for reproducible, read-only analyses."""
    return dict(sha256=sha256_file(path), size_bytes=path.stat().st_size)


class WaveSession:
    """Load a completed video session without creating or replacing source files.

    ``strict=False`` allows inspection reports for inconsistent inputs. Annotation
    uses strict mode so mislabeled frames or calibration mismatches cannot enter
    new geometry silently. Device indices are recorded identities only.
    """

    def __init__(self, path, *, strict=True):
        self.path = Path(path).resolve()
        self.issues = []
        self.sources = {}
        self.recording = self._json('video/recording.json')
        self.metadata = self._json('session.json')
        manifest = self._json('calibration/manifest.json')
        # Validate nested metadata before using it, so inspect can still report
        # a corrupt session instead of failing midway through its audit.
        for key in ('streams', 'camera_settings', 'calibration_hashes'):
            if not isinstance(self.recording.get(key), dict):
                self.issues.append(f'recording.json has missing or malformed {key}')
                self.recording[key] = {}
        if not isinstance(manifest.get('files'), dict):
            self.issues.append('Calibration manifest has missing or malformed files')
            manifest['files'] = {}
        for camera in CAMERAS:
            if not isinstance(self.recording['streams'].get(camera), dict):
                self.issues.append(f'recording.json has missing or malformed {camera} stream')
                self.recording['streams'][camera] = {}
        for name in CALIB_NAMES:
            if not isinstance(manifest['files'].get(name), dict):
                self.issues.append(f'Calibration manifest missing entry for {name}')
                manifest['files'][name] = {}
        self.rows = {}
        self.cal = None
        paths = {name: self.path / 'calibration' / name for name in CALIB_NAMES}
        for name, file in paths.items():
            self._fingerprint(file)
        if all(p.is_file() for p in paths.values()):
            try:
                self.cal = load_calibration(paths)
            except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, cv2.error) as exc:
                self.issues.append(f'Cannot load calibration: {exc}')
        if self.cal is not None:
            self.issues.extend(c['message'] for c in self.cal.errors)
            for name, digest in self.cal.hashes.items():
                for source, expected in (
                    ('recording', self.recording.get('calibration_hashes', {}).get(name)),
                    ('manifest', manifest.get('files', {}).get(name, {}).get('sha256')),
                ):
                    if expected != digest:
                        self.issues.append(f'{source} calibration hash missing or different for {name}')
            if self.cal.F is None or np.asarray(self.cal.F).shape != (3, 3) or not np.all(np.isfinite(self.cal.F)) or np.linalg.norm(self.cal.F) == 0:
                self.issues.append('Saved fundamental matrix is missing or unusable')
        for camera in CAMERAS:
            video = self.path / 'video' / f'{camera}.avi'
            log = self.path / 'video' / f'{camera}_frames.csv'
            self._fingerprint(video)
            self._fingerprint(log)
            try:
                rows, errors = read_timestamps(log)
            except (OSError, ValueError) as exc:
                rows, errors = [], [str(exc)]
            self.rows[camera] = rows
            self.issues.extend(f'{camera}: {message}' for message in errors)
            expected = self.recording.get('streams', {}).get(camera, {}).get('frames')
            if expected != len(rows):
                self.issues.append(f'{camera}: metadata frame count {expected} differs from CSV {len(rows)}')
            start = self.recording.get('start_unix_ns')
            if rows and any(r['read_timestamp_ns'] - r['elapsed_ns'] != start for r in rows):
                self.issues.append(f'{camera}: timestamps disagree with recording start_unix_ns')
        settings = self.recording.get('camera_settings', {})
        if settings.get('cam0') is None or settings.get('cam1') is None or settings.get('cam0') == settings.get('cam1'):
            self.issues.append('Camera device identities missing or identical')
        self.pairs = []
        timing_ok = all(self.rows.values()) and all(
            [r['frame_index'] for r in rows] == list(range(len(rows))) and
            all(a['elapsed_ns'] < b['elapsed_ns'] for a, b in zip(rows, rows[1:]))
            for rows in self.rows.values())
        if timing_ok:
            times = [[r['elapsed_ns'] for r in self.rows[c]] for c in CAMERAS]
            self.pairs = _all_frame_pairs(*times)
        self.reused = {camera: Counter(p[i] for p in self.pairs) for i, camera in enumerate(CAMERAS)}
        if strict and self.issues:
            raise ValueError('\n'.join(self.issues))

    def _fingerprint(self, path):
        if path.is_file():
            self.sources[str(path.relative_to(self.path))] = file_identity(path)
        else:
            self.issues.append(f'Missing file {path.relative_to(self.path)}')

    def _json(self, relative):
        path = self.path / relative
        self._fingerprint(path)
        try:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except (OSError, ValueError) as exc:
            self.issues.append(f'{relative}: {exc}')
            return {}

    def pair_info(self, index):
        """Return both original timestamps, indices and reuse for a candidate pair."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.pairs):
            raise ValueError('Unknown pair index')
        frames = self.pairs[index]
        rows = [self.rows[c][i] for c, i in zip(CAMERAS, frames)]
        return dict(pair_index=index, frame_indices=list(frames),
                    cameras={c: dict(self.rows[c][i], device=self.recording['camera_settings'].get(c),
                                     pair_uses=self.reused[c][i]) for c, i in zip(CAMERAS, frames)},
                    host_gap_ms=abs(rows[1]['read_timestamp_ns']-rows[0]['read_timestamp_ns'])/1e6,
                    representative_elapsed_seconds=(rows[0]['elapsed_ns']+rows[1]['elapsed_ns'])/2e9,
                    exposure_time_uncertainty_seconds=None)

    def frame(self, camera, index):
        """Decode sequentially from frame zero so CSV indices agree with pixels."""
        if camera not in CAMERAS or not isinstance(index, int) or not 0 <= index < len(self.rows[camera]):
            raise ValueError('Unknown camera or frame index')
        cap = cv2.VideoCapture(str(self.path / 'video' / f'{camera}.avi'))
        try:
            frame = None
            for _ in range(index + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'{camera} cannot decode frame {index}')
            size = getattr(self.cal, f'size{camera[-1]}')
            if frame.shape[:2] != (size[1], size[0]):
                raise ValueError('Video dimensions differ from calibration')
            return frame
        finally:
            cap.release()

    def assert_calibration_current(self):
        """Refuse geometry if a session snapshot changes after server startup."""
        for name, digest in self.cal.hashes.items():
            if sha256_file(self.path / 'calibration' / name) != digest:
                raise ValueError(f'Calibration changed after loading: {name}')

    def calibration_audit(self):
        """Expose geometry checks without certifying physical camera identity."""
        if self.cal is None:
            return None
        try:
            relative = compare_relative_geometry(self.cal)
        except (ValueError, KeyError, TypeError) as exc:
            relative = dict(error=str(exc))
        return dict(hashes=self.cal.hashes, checks=self.cal.checks,
                    relative_geometry=relative,
                    board_to_tank_identity=self.cal.tank_transform_is_identity,
                    tank_axes_verified=False, physical_up_verified=False,
                    camera_identity_status='Recorded device indices only; serial identity unavailable')
