"""Versioned surface observations stored separately from the original trial."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .reports import code_version, json_text, output_path


class RevisionConflict(ValueError):
    """Another reviewer saved a newer version of this pair."""


def atomic_create(path, value):
    """Publish one complete JSON file atomically without replacing an old file."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def pixel(value, size):
    """Validate a full-resolution observation without rounding or snapping."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('A pixel needs two coordinates')
    u, v = [number(x, 'pixel') for x in value]
    if not 0 <= u < size[0] or not 0 <= v < size[1]:
        raise ValueError('Pixel is outside the original image')


def validate_document(data, session):
    """Validate independent traces, verified matches and breaking evidence."""
    if not isinstance(data, dict):
        raise ValueError('Expected an annotation object')
    ids = set()
    for group in ('points', 'traces', 'masks'):
        items = data.get(group)
        if not isinstance(items, list) or len(items) > 5000:
            raise ValueError(f'{group} must be a list with at most 5000 entries')
        for item in items:
            if not isinstance(item, dict):
                raise ValueError('Annotations must be objects')
            identifier = item.get('id')
            if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier) or identifier in ids:
                raise ValueError('Annotation IDs must be unique simple identifiers')
            ids.add(identifier)
            if item.get('origin') not in ('human', 'machine'):
                raise ValueError('Record annotation origin as human or machine')
            confidence = number(item.get('confidence'), 'confidence')
            if not 0 <= confidence <= 1:
                raise ValueError('confidence must be between 0 and 1')
            for key in ('label', 'notes'):
                if not isinstance(item.get(key, ''), str) or len(item.get(key, '')) > 4000:
                    raise ValueError(f'Invalid {key}')
            if group == 'points':
                if item.get('status') not in ('verified', 'uncertain', 'unmatched'):
                    raise ValueError('Choose verified, uncertain, or unmatched')
                if type(item.get('order')) is not int or item['order'] < 0:
                    raise ValueError('Point order must be a nonnegative integer')
                present = [camera for camera in ('cam0', 'cam1') if item.get(camera) is not None]
                if not present or (item['status'] == 'verified' and len(present) != 2):
                    raise ValueError('Verified points need both camera observations')
                if item['status'] == 'unmatched' and len(present) != 1:
                    raise ValueError('Unmatched points need exactly one camera observation')
                if item['origin'] == 'machine' and item['status'] == 'verified' and item.get('human_reviewed') is not True:
                    raise ValueError('Machine suggestions require human review')
                for camera in present:
                    pixel(item[camera], getattr(session.cal, f'size{camera[-1]}'))
            else:
                camera = item.get('camera')
                if camera not in ('cam0', 'cam1'):
                    raise ValueError('Unknown camera')
                vertices = item.get('vertices_px')
                minimum = 3 if group == 'masks' else 2
                if not isinstance(vertices, list) or not minimum <= len(vertices) <= 10000:
                    raise ValueError(f'{group} require {minimum} to 10000 vertices')
                for vertex in vertices:
                    pixel(vertex, getattr(session.cal, f'size{camera[-1]}'))
                if group == 'masks' and item.get('kind') not in ('reflection', 'occlusion', 'glare'):
                    raise ValueError('Unknown mask kind')
                if group == 'traces' and item.get('direction') not in ('along_propagation', 'unknown'):
                    raise ValueError('Record trace direction explicitly')
    breaking = data.get('breaking')
    if not isinstance(breaking, dict) or breaking.get('observed') not in ('yes', 'no', 'unknown'):
        raise ValueError('Breaking must be yes, no, or unknown')
    if type(breaking.get('observation_complete')) is not bool:
        raise ValueError('Record whether the complete outcome was observable')
    if breaking['observed'] == 'no' and not breaking['observation_complete']:
        raise ValueError('Use unknown when a wave leaves view before its outcome is known')
    confidence = number(breaking.get('confidence'), 'breaking confidence')
    if not 0 <= confidence <= 1:
        raise ValueError('Breaking confidence must be between 0 and 1')
    for field in ('wave_id', 'notes', 'criterion'):
        if not isinstance(breaking.get(field), str) or len(breaking[field]) > 4000:
            raise ValueError(f'Invalid breaking {field}')
    camera = breaking.get('onset_camera')
    if breaking['observed'] == 'yes':
        if camera not in ('cam0', 'cam1') or not breaking['criterion'].strip():
            raise ValueError('Select the camera and criterion for first visible breaking')
        if breaking.get('onset_pixel') is not None:
            pixel(breaking['onset_pixel'], getattr(session.cal, f'size{camera[-1]}'))
    elif camera is not None or breaking.get('onset_pixel') is not None:
        raise ValueError('Only a yes outcome may have an onset observation')
    if not isinstance(data.get('notes', ''), str) or len(data.get('notes', '')) > 10000:
        raise ValueError('Invalid pair notes')


def empty_document():
    """Return an explicitly unreviewed pair with unknown breaking outcome."""
    return dict(revision=0, points=[], traces=[], masks=[], notes='', breaking=dict(
        observed='unknown', wave_id='', onset_camera=None, onset_pixel=None,
        observation_complete=False, confidence=0.5, notes='',
        criterion='First visible crest overturning or spilling foam attached to the wave, checked in adjacent frames'))


class AnnotationStore:
    """Append-only revisions with optimistic concurrency across browser sessions."""

    def __init__(self, session, output_dir):
        self.session = session
        self.path = output_path(session.path, output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / 'annotation_manifest.json'
        identity = dict(schema_version=1, source_session=str(session.path), source_files=session.sources,
                        calibration_hashes=session.cal.hashes)
        if not manifest.exists():
            if any(self.path.iterdir()):
                raise ValueError('Annotation directory contains files without a manifest')
            try:
                atomic_create(manifest, identity)
            except FileExistsError:
                pass
        existing = json.loads(manifest.read_text())
        if existing.get('source_files') != identity['source_files'] or existing.get('schema_version') != 1:
            raise ValueError('Annotation directory belongs to different source files')

    def load(self, index):
        """Read the latest complete revision, or return a blank document."""
        pair = self.session.pair_info(index)
        files = sorted((self.path / f'pair_{index:05d}').glob('revision_*.json'))
        if files:
            document = json.loads(files[-1].read_text())
            expected_revision = int(files[-1].stem.split('_')[1])
            if (document.get('schema_version') != 1 or document.get('revision') != expected_revision
                    or document.get('pair') != pair
                    or document.get('calibration_hashes') != self.session.cal.hashes):
                raise ValueError('Saved annotation provenance or revision does not match this session')
            validate_document(document, self.session)
            return document
        return dict(empty_document(), pair=pair)

    def save(self, index, data, expected_revision):
        """Append a validated revision and preserve stale edits on conflict."""
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError('A nonnegative base revision is required')
        self.session.assert_calibration_current()
        validate_document(data, self.session)
        current = self.load(index)
        if current['revision'] != expected_revision:
            raise RevisionConflict('A newer revision exists. Reopen this pair before saving your edit.')
        document = {key: deepcopy(data[key]) for key in ('points', 'traces', 'masks', 'breaking')}
        document.update(schema_version=1, revision=expected_revision+1, notes=data.get('notes', ''),
                        pair=self.session.pair_info(index), calibration_hashes=self.session.cal.hashes,
                        source_session=str(self.session.path), code=code_version(),
                        saved_utc=datetime.now(timezone.utc).isoformat())
        event = document['breaking']
        event['onset_observation'] = None
        if event['observed'] == 'yes':
            event['onset_observation'] = document['pair']['cameras'][event['onset_camera']]
        folder = self.path / f'pair_{index:05d}'
        folder.mkdir(exist_ok=True)
        try:
            atomic_create(folder / f'revision_{document["revision"]:06d}.json', document)
        except FileExistsError as exc:
            raise RevisionConflict('Another reviewer saved concurrently. Reopen before saving.') from exc
        return document
