"""Sparse measured profiles with explicit gaps, coordinates and uncertainty."""
import numpy as np

from validate_stereo_3d import cam0_to_board, triangulate_point


def point_covariance(point, cal, rotation, pixel_sigma):
    """First-order covariance from independent raw-pixel noise only (mm²).

    Excludes calibration, refractive model and timing bias. Those must never be
    presented as covered by the pixel-noise calculation.
    """
    if pixel_sigma is None:
        return None
    uv = np.asarray(point['cam0'] + point['cam1'], float)
    J = np.zeros((3, 4))
    for i in range(4):
        delta = np.zeros(4)
        delta[i] = .01
        points = []
        for sample in (uv + delta, uv - delta):
            tri = triangulate_point(sample[:2], sample[2:], cal)
            points.append(rotation @ cam0_to_board(np.asarray(tri['X_cam0_mm']), cal))
        J[:, i] = (points[0] - points[1]) / .02
    cov = (pixel_sigma ** 2) * J @ J.T
    return cov if np.isfinite(cov).all() else None


def build_profiles(documents, config, cal=None):
    """Convert approved board points into sparse tank profiles, in SI units.

    Sorts by tank Y and splits at configured gaps. No extrapolation, rescaling,
    snapping or replacement of missing points. Coincident Y observations are
    excluded as ambiguous rather than silently averaged into a single surface.
    """
    matrix = np.asarray(config['board_to_tank_mm'], float)
    R, t = matrix[:3, :3], matrix[:3, 3]
    profiles, exclusions = [], []
    for record in documents:
        doc = record['annotations']
        info = doc['pair']
        rows = []
        for point in record['reconstructed']:
            geometry = point.get('geometry') or {}
            reason = None
            xyz = geometry.get('X_board_mm')
            if point['status'] != 'verified' or not geometry.get('valid'):
                reason = 'unverified_or_invalid_geometry'
            elif point['mask_ids']:
                reason = 'masked'
            elif info['host_gap_ms'] > config['max_host_gap_ms']:
                reason = 'host_gap_exceeds_configured_limit'
            elif xyz is None or not np.isfinite(xyz).all():
                reason = 'nonfinite_geometry'
            if reason is None:
                tank = R @ np.asarray(xyz, float) + t
                if abs(tank[0] - config['slice_x_mm']) > config['slice_halfwidth_mm']:
                    reason = 'outside_x_slice'
                elif not config['y_range_mm'][0] <= tank[1] <= config['y_range_mm'][1]:
                    reason = 'outside_y_range'
            if reason:
                exclusions.append(dict(pair_index=info['pair_index'], point_id=point['id'], reason=reason))
                continue
            cov = point_covariance(point, cal, R, config['pixel_sigma_px']) if cal is not None else None
            sys = config['systematic_position_sigma_mm']
            ref = config['still_water_sigma_mm']
            total = cov + np.eye(3)*sys**2 if cov is not None and sys is not None else None
            rows.append(dict(point_id=point['id'], x_m=tank[0]/1000, y_m=tank[1]/1000,
                z_m=tank[2]/1000, eta_m=(tank[2]-config['still_water_z_mm'])/1000,
                pixel_noise_sigma_z_m=None if cov is None else np.sqrt(cov[2,2])/1000,
                sigma_y_m=None if total is None else np.sqrt(total[1,1])/1000,
                sigma_eta_m=None if total is None or ref is None else np.sqrt(total[2,2]+ref**2)/1000))
        rows.sort(key=lambda p: p['y_m'])
        ambiguous = set()
        for a, b in zip(rows, rows[1:]):
            if b['y_m']-a['y_m'] <= 1e-9:
                ambiguous.update([a['point_id'], b['point_id']])
        exclusions.extend(dict(pair_index=info['pair_index'], point_id=i, reason='ambiguous_duplicate_y') for i in sorted(ambiguous))
        rows = [r for r in rows if r['point_id'] not in ambiguous]
        segment = 0
        for i, row in enumerate(rows):
            if i and (row['y_m']-rows[i-1]['y_m'])*1000 > config['max_gap_mm']:
                segment += 1
            row['segment'] = segment
        covered = sum(b['y_m']-a['y_m'] for a,b in zip(rows,rows[1:]) if a['segment']==b['segment'])
        span = (config['y_range_mm'][1]-config['y_range_mm'][0])/1000
        profiles.append(dict(pair_index=info['pair_index'], annotation_revision=doc['revision'],
            pair=info, time_s=info['representative_elapsed_seconds'], points=rows,
            coverage_fraction=covered/span, covered_length_m=covered,
            status='sparse_profile' if len(rows)>=2 else 'insufficient_points',
            experimental_validation=config['experimental_validation'],
            uncertainty_note='Conditional on pixel/systematic/reference assumptions; exposure skew and optical bias are not inferred.'))
    return profiles, exclusions
