"""Non-destructive feasibility reports for recorded stereo wave sessions."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from record_stereo_waves import CAMERAS, TIMING_WARNING
from validate_stereo_3d import read_json
from .session_io import WaveSession, distribution, timestamp_summary


def json_text(value):
    """Serialize diagnostics with strict JSON nulls for unavailable values."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(v) for key, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.generic):
            return item.item()
        return item
    return json.dumps(clean(value), indent=2, allow_nan=False) + '\n'


def output_path(session, output):
    """Require an analysis location under outputs, outside the source trial."""
    path = Path(output).resolve()
    source = Path(session).resolve()
    if 'outputs' not in path.parts or path == source or source in path.parents or path in source.parents:
        raise ValueError('Choose a separate directory under outputs, outside the original session')
    return path


def code_version():
    """Record the repository revision and whether source changes are present."""
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)
    files = [root / 'camera' / 'analyze_wave_trial.py']
    files.extend(sorted((root / 'camera' / 'wave_analysis').rglob('*')))
    digests = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in files if p.is_file() and '__pycache__' not in p.parts}
    return dict(commit=sha, dirty=dirty, implementation_sha256=digests)


def write_csv(path, rows, columns):
    """Write a new CSV with explicit columns, including for empty data."""
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def scan_video(path, selected, expected_size):
    """Decode the full video, count frames, and flag byte-identical images.

    Identical decoded pixels can also occur in a stationary scene. They are
    reported as candidates, never classified as proven repeated exposures.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return dict(error='Video cannot be opened', decoded_frames=0), {}
    header_fps = cap.get(cv2.CAP_PROP_FPS)
    header_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames, seen, repeats, bad_sizes = {}, {}, [], []
    count = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if list(image.shape[1::-1]) != list(expected_size or []):
                bad_sizes.append(count)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                repeats.append([seen[digest], count])
            else:
                seen[digest] = count
            if count in selected:
                frames[count] = image
            count += 1
    finally:
        cap.release()
    return dict(decoded_frames=count, header_fps=header_fps, header_frame_count=header_count,
                dimension_mismatch_indices=bad_sizes, identical_pixel_candidates=repeats), frames


def audit_exported_pairs(session):
    """Cross-check old exported pairs against original frame timestamps and hashes."""
    issues, pairs, uses = [], [], {c: Counter() for c in CAMERAS}
    for directory in sorted((session.path / 'pairs').glob('pair_*')):
        if not directory.is_dir():
            continue
        try:
            meta = read_json(directory / 'pair.json')
            indices = meta['video_frame_indices']
            if len(indices) != 2:
                raise ValueError('Expected two video frame indices')
            for camera, index in zip(CAMERAS, indices):
                if type(index) is not int or index < 0:
                    raise ValueError('Invalid frame index')
                row = session.rows[camera][index]
                uses[camera][index] += 1
                if meta[camera]['read_timestamp_ns'] != row['read_timestamp_ns']:
                    issues.append(f'{directory.name}: {camera} timestamp mismatch')
                image = cv2.imread(str(directory / f'{camera}.png'))
                if image is None or tuple(image.shape[1::-1]) != getattr(session.cal, f'size{camera[-1]}'):
                    issues.append(f'{directory.name}: {camera} image missing or dimensions disagree')
            if session.cal and meta.get('calibration_hashes') != session.cal.hashes:
                issues.append(f'{directory.name}: calibration hash mismatch')
            pairs.append(tuple(indices))
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            issues.append(f'{directory.name}: {exc}')
    return dict(count=len(pairs), issues=issues,
                duplicate_pairs=[list(p) for p, n in Counter(pairs).items() if n > 1],
                reuse_counts={c: {str(i): n for i, n in counts.items() if n > 1} for c, counts in uses.items()})


def inspect_trial(session_path, output_dir, *, samples=9):
    """Audit real files and export representative raw views for human inspection."""
    if type(samples) is not int or not 1 <= samples <= 30:
        raise ValueError('samples must be between 1 and 30')
    session = WaveSession(session_path, strict=False)
    output = output_path(session.path, output_dir)
    output.mkdir(parents=True, exist_ok=False)
    review = output / 'review'
    review.mkdir()
    pairs = [session.pair_info(i) for i in range(len(session.pairs))]
    if pairs:
        times = np.array([p['representative_elapsed_seconds'] for p in pairs])
        selected = sorted({int(np.argmin(abs(times-t))) for t in np.linspace(times[0], times[-1], samples)})
    else:
        selected = []
    videos, images, timing = {}, {}, []
    for camera in CAMERAS:
        camera_index = int(camera[-1])
        indices = {session.pairs[i][camera_index] for i in selected}
        expected_size = getattr(session.cal, f'size{camera_index}', None)
        videos[camera], images[camera] = scan_video(session.path / 'video' / f'{camera}.avi', indices, expected_size)
        videos[camera].update(timestamp_summary(session.rows[camera]))
        recorded_size = session.recording.get('camera_settings', {}).get(f'actual_size_{camera}')
        if recorded_size is not None and list(recorded_size) != list(expected_size or []):
            session.issues.append(f'{camera}: recorded camera dimensions differ from calibration')
        if videos[camera]['decoded_frames'] != len(session.rows[camera]):
            session.issues.append(f'{camera}: decoded video count differs from timestamp count')
        if videos[camera].get('dimension_mismatch_indices'):
            session.issues.append(f'{camera}: decoded dimensions differ from calibration')
        previous = None
        for row in session.rows[camera]:
            timing.append(dict(camera=camera, **row, interval_ms=None if previous is None else
                               (row['elapsed_ns']-previous)/1e6,
                               pair_uses=session.reused[camera][row['frame_index']]))
            previous = row['elapsed_ns']
    contact_rows, representatives = [], []
    for i in selected:
        pair = pairs[i]
        cells = []
        for camera, index in zip(CAMERAS, pair['frame_indices']):
            image = images[camera].get(index)
            if image is None:
                continue
            filename = f'pair_{i:04d}_{camera}_frame_{index:04d}.png'
            if not cv2.imwrite(str(review / filename), image):
                raise OSError(f'Could not write {filename}')
            pair[camera + '_image'] = 'review/' + filename
            cell = cv2.resize(image, (640, 360))
            cv2.rectangle(cell, (0, 0), (640, 36), (20, 20, 20), -1)
            label = f'{camera} frame {index} | {pair["cameras"][camera]["elapsed_ns"]/1e9:.3f}s | gap {pair["host_gap_ms"]:.1f}ms'
            cv2.putText(cell, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cells.append(cell)
        if len(cells) == 2:
            contact_rows.append(np.hstack(cells))
        representatives.append(pair)
    if contact_rows:
        cv2.imwrite(str(review / 'contact_sheet.jpg'), np.vstack(contact_rows))
    exported = audit_exported_pairs(session)
    session.issues.extend(exported['issues'])
    report = dict(schema_version=1, source_session=str(session.path), created_utc=datetime.now(timezone.utc).isoformat(),
                  code=code_version(), source_files=session.sources, issues=session.issues,
                  cameras=session.recording.get('camera_settings', {}), streams=videos,
                  calibration=session.calibration_audit(), exported_pairs=exported,
                  candidate_pair_count=len(pairs), host_gap_ms=distribution([p['host_gap_ms'] for p in pairs]),
                  candidate_reuse={c: {str(i): n for i, n in counts.items() if n > 1} for c, counts in session.reused.items()},
                  timing_warning=TIMING_WARNING, representative_pairs=representatives,
                  surface_assessment=dict(status='requires visual review', stereo_feasibility='unknown',
                                          physical_accuracy='unvalidated', recommended_method='Manual point matches and independent image traces'))
    (output / 'quality_report.json').write_text(json_text(report))
    (output / 'analysis_manifest.json').write_text(json_text({k: report[k] for k in
        ('schema_version', 'source_session', 'created_utc', 'code', 'source_files')}))
    write_csv(output / 'session_timing.csv', timing,
              ['camera', *('frame_index', 'elapsed_ns', 'read_timestamp_ns'), 'interval_ms', 'pair_uses'])
    write_csv(output / 'pair_timing.csv', [dict(pair_index=p['pair_index'], cam0_frame=p['frame_indices'][0],
              cam1_frame=p['frame_indices'][1], elapsed_seconds=p['representative_elapsed_seconds'], host_gap_ms=p['host_gap_ms'],
              cam0_uses=p['cameras']['cam0']['pair_uses'], cam1_uses=p['cameras']['cam1']['pair_uses']) for p in pairs],
              ['pair_index', 'cam0_frame', 'cam1_frame', 'elapsed_seconds', 'host_gap_ms', 'cam0_uses', 'cam1_uses'])
    lines = ['# Wave trial inspection', '', f'Source `{session.path}`', '',
             '| Camera | Decoded frames | Observed fps | Median interval ms |', '| --- | --- | --- | --- |']
    for c, data in videos.items():
        lines.append(f'| {c} | {data["decoded_frames"]} | {data["observed_fps"]} | {data["intervals_ms"]["median"]} |')
    lines += ['', f'Candidate pairs {len(pairs)}. Host read gaps in ms `{report["host_gap_ms"]}`.', '',
              TIMING_WARNING, '', '## Visual review', '',
              'Inspect the full-resolution views in `review/` and record findings in a separate `surface_review.md`.', '',
              '- Identify the actual water surface in each camera and state which regions remain ambiguous.',
              '- Check whether the same physical surface features are visible in both views.',
              '- Record glare, transparency, wall reflections and occlusion.',
              '- Assess longitudinal coverage and distinguish image traces from verified stereo pairs.',
              '- A wall waterline may support a separately calibrated single-camera measurement after verifying its plane and optical path.',
              '- Begin with the representative pairs. Save uncertain points as uncertain or unmatched.', '',
              '## Integrity findings', '', *(f'- {issue}' for issue in session.issues), '',
              'Detailed checks and calibration geometry are in `quality_report.json`. Camera serial identity, exposure skew, physical up and tank axes require experimental verification.']
    (output / 'feasibility_report.md').write_text('\n'.join(lines)+'\n')
    return report
