"""Interpretable edge/patch proposals. No suggestion is physical ground truth."""
import uuid
import cv2
import numpy as np
from correspondence_guide import predict_correspondence
from validate_stereo_3d import triangulate_point


def polyline_distance(points, vertices):
    """Euclidean raw-pixel distance to a polyline, including its endpoints."""
    points, vertices = np.asarray(points, float), np.asarray(vertices, float)
    if points.ndim != 2 or points.shape[1] != 2 or vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
        raise ValueError('Need Nx2 points and at least two polyline vertices')
    a, delta = vertices[:-1], np.diff(vertices, axis=0)
    diff = points[:, None, :]-a
    fraction = np.clip(np.sum(diff*delta, axis=2)/np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
    return np.min(np.linalg.norm(diff-fraction[..., None]*delta, axis=2), axis=1)


def suggest_trace(image, seed, masks=(), *, band_px=12, step_px=4, min_gradient=20, min_margin=0.15):
    """Refine a rough x-monotonic trace using isolated vertical intensity edges.

    Missing/ambiguous columns split traces. No interpolation spans those gaps.
    The search band is a user-selected region, not a water-surface classifier.
    """
    vertices = np.asarray(seed, float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError('Seed needs at least two finite raw pixel vertices')
    if np.all(np.diff(vertices[:, 0]) < 0):
        vertices = vertices[::-1]
    if np.any(np.diff(vertices[:, 0]) <= 0):
        raise ValueError('Trace must run left-to-right or right-to-left without folding in image X')
    height, width = image.shape[:2]
    if np.any(vertices < 0) or np.any(vertices >= [width, height]):
        raise ValueError('Seed outside image')
    if type(band_px) is not int or not 2 <= band_px <= 100 or type(step_px) is not int or not 1 <= step_px <= 32:
        raise ValueError('Band must be 2..100 px; sampling step 1..32 px')
    if not np.isfinite([min_gradient, min_margin]).all() or min_gradient <= 0 or not 0 <= min_margin <= 1:
        raise ValueError('Invalid gradient/margin gates')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gradient = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 0, 1))
    excluded = np.zeros((height, width), np.uint8)
    for polygon in masks:
        cv2.fillPoly(excluded, [np.asarray(polygon, np.int32)], 1)
    xs = np.arange(int(np.ceil(vertices[0, 0])), int(vertices[-1, 0])+1, step_px)
    segments, run, missing, confidence = [], [], [], []
    for x in xs:
        center = np.interp(x, vertices[:, 0], vertices[:, 1])
        ys = np.arange(max(0, int(center)-band_px), min(height, int(center)+band_px+1))
        scores = gradient[ys, x].copy(); scores[excluded[ys, x] != 0] = 0
        best = int(np.argmax(scores)); value = float(scores[best])
        rival = scores.copy(); rival[max(0, best-2):best+3] = 0
        margin = (value-float(rival.max()))/max(value, 1e-12)
        if value < min_gradient or margin < min_margin:
            if len(run) >= 2: segments.append(run)
            run = []; missing.append(int(x)); continue
        run.append([int(x), int(ys[best])]); confidence.append(margin)
    if len(run) >= 2: segments.append(run)
    retained = sum(map(len, segments))
    return dict(segments=segments, coverage_fraction=retained/max(len(xs), 1), missing_columns=missing,
                confidence=float(np.mean(confidence)) if confidence else 0,
                localization_sigma_px=None, uncertainty_note='Pixel localization uncertainty requires held-out manual evaluation',
                settings=dict(band_px=band_px, step_px=step_px, min_gradient=min_gradient, min_margin=min_margin))


def _patch(image, p, radius=4):
    x, y = np.rint(p).astype(int); h, w = image.shape[:2]
    if min(x, y) < radius or x+radius >= w or y+radius >= h:
        return None
    values = image[y-radius:y+radius+1, x-radius:x+radius+1].astype(float)
    values -= values.mean()
    norm = np.linalg.norm(values)
    return values/norm if norm > 10 else None


def suggest_matches(images, traces, cal, *, min_ncc=0.8, uniqueness=0.1, curve_tolerance_px=2):
    """Propose mutually unique patch matches near the saved epipolar curve.

    Try upright and 180-degree patches for the opposed views. Perspective and
    glare can defeat this deliberately simple matcher; rejection is expected.
    Existing triangulation validity limits are applied without modification.
    """
    samples = [np.asarray([p for segment in traces[c] for p in segment], float).reshape(-1, 2) for c in ('cam0', 'cam1')]
    # Cap work on large 1080p images while retaining original pixel positions.
    samples = [v[::max(1, int(np.ceil(len(v)/200)))] for v in samples]
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images]
    patches = [[_patch(im, p) for p in ps] for im, ps in zip(gray, samples)]
    scores = np.full((len(samples[0]), len(samples[1])), -np.inf)
    for i, p in enumerate(samples[0]):
        if patches[0][i] is None or not len(samples[1]): continue
        curves = predict_correspondence(cal, 0, p.tolist())['curve_segments_px']
        curves = [c for c in curves if len(c) >= 2]
        if not curves: continue
        distance = np.min([polyline_distance(samples[1], c) for c in curves if len(c) >= 2], axis=0)
        for j in np.flatnonzero(distance <= curve_tolerance_px):
            if patches[1][j] is not None:
                scores[i, j] = max(float(np.sum(patches[0][i]*p2)) for p2 in (patches[1][j], patches[1][j][::-1, ::-1]))
    def unique(values, index, positions):
        rivals = values.copy(); rivals[np.linalg.norm(positions-positions[index], axis=1) <= 5] = -np.inf
        return values[index] >= min_ncc and values[index]-np.max(rivals, initial=-np.inf) >= uniqueness
    result = []
    for i, row in enumerate(scores):
        if not len(row): continue
        j = int(np.argmax(row))
        if not unique(row, j, samples[1]) or int(np.argmax(scores[:, j])) != i or not unique(scores[:, j], i, samples[0]): continue
        geometry = triangulate_point(samples[0][i], samples[1][j], cal)
        result.append(dict(id=uuid.uuid4().hex, label='Suggested match', notes='Patch NCC + epipolar constraint; inspect physical identity',
            origin='machine', human_reviewed=False, confidence=float(np.clip(row[j], 0, 1)), order=len(result),
            status='uncertain', cam0=samples[0][i].tolist(), cam1=samples[1][j].tolist(),
            suggestion_geometry=geometry, positional_uncertainty_mm=None))
    return result


def suggest_document(images, document, cal, band_px=12):
    """Return new proposals separately; never replace/save human annotations."""
    traces, diagnostics, segments = [], {}, {}
    for camera, image in zip(('cam0', 'cam1'), images):
        seeds = [t for t in document['traces'] if t['camera'] == camera and t['origin'] == 'human']
        if len(seeds) != 1:
            raise ValueError('Draw exactly one rough human Image trace per camera as the detection search band')
        masks = [m['vertices_px'] for m in document['masks'] if m['camera'] == camera]
        found = suggest_trace(image, seeds[0]['vertices_px'], masks, band_px=band_px)
        segments[camera] = found['segments']; diagnostics[camera] = found
        for vertices in found['segments']:
            traces.append(dict(id=uuid.uuid4().hex, label='Suggested waterline', notes='Edge candidate; human review required',
                origin='machine', human_reviewed=False, confidence=found['confidence'], camera=camera,
                vertices_px=vertices, suggested_vertices_px=vertices, direction='unknown'))
    return dict(traces=traces, points=suggest_matches(images, segments, cal), diagnostics=diagnostics,
                note='Accept/edit/reject each suggestion. Independent traces are not stereo correspondences.')


def evaluate_traces(reference, proposed, tolerance_px=3):
    """Measure symmetric trace errors/coverage on held-out human reference traces."""
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('Tolerance must be positive')
    def dense(lines):
        points = []
        for line in lines:
            for a, b in zip(line[:-1], line[1:]):
                n = max(2, int(np.ceil(np.linalg.norm(np.subtract(b, a))))+1)
                points.extend(np.linspace(a, b, n))
        return np.asarray(points).reshape(-1, 2)
    r, p = dense(reference), dense(proposed)
    def distances(points, lines):
        return np.min([polyline_distance(points, line) for line in lines], axis=0)
    recall = distances(r, proposed) if len(r) and len(p) else None
    precision = distances(p, reference) if len(r) and len(p) else None
    return dict(reference_samples=len(r), proposed_samples=len(p), tolerance_px=tolerance_px,
        coverage_within_tolerance=float(np.mean(recall <= tolerance_px)) if recall is not None else 0,
        precision_within_tolerance=float(np.mean(precision <= tolerance_px)) if precision is not None else 0,
        mean_error_px=float(np.mean(precision)) if precision is not None else None,
        p95_error_px=float(np.percentile(precision, 95)) if precision is not None else None)
