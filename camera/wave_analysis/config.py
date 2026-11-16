"""Explicit, session-bound physical conventions; no inferred axis flips or scale."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def template():
    """Return an intentionally incomplete laboratory measurement configuration."""
    return dict(schema_version=1, trial_id=None, calibration_hashes=None,
                frame_verified=False, frame_evidence='', board_to_tank_mm=None,
                still_water_z_mm=None, still_water_sigma_mm=None,
                slice_x_mm=None, slice_halfwidth_mm=None, y_range_mm=None,
                max_gap_mm=None, max_host_gap_ms=None, propagation_direction=None,
                pixel_sigma_px=None, systematic_position_sigma_mm=None,
                exposure_time_sigma_s=None, uncertainty_evidence='',
                min_wave_height_mm=None, min_samples_per_wave=9,
                smoothing_window=1, shape_reference='local_wave_mean', max_track_gap_s=None,
                paddle_sweep_angle_deg=None, paddle_amplitude=None,
                paddle_amplitude_units=None, paddle_amplitude_kind=None,
                driving_frequency_hz=None, still_water_depth_mm=None,
                ramp_start_y_mm=None, bottom_profile_yz_mm=None,
                experimental_validation=dict(verified=False, evidence=''), notes='')


def finite(value, name, *, positive=False, nonnegative=False):
    """Reject missing, boolean and nonfinite numerical settings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite measured/configured number')
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f'{name} has an invalid sign')
    return float(value)


def validate_config(data, calibration_hashes=None):
    """Validate a proper rigid mm transform with tank +Z physically upward.

    Verification and evidence are researcher declarations, not automatic proof.
    Calibration hashes bind the declaration to the actual recording snapshot.
    """
    if not isinstance(data, dict):
        raise ValueError('Configuration must be an object')
    unknown = set(data) - set(template())
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = dict(template(), **deepcopy(data))
    if type(config['schema_version']) is not int or config['schema_version'] != 1 or not isinstance(config['trial_id'], str) or not config['trial_id'].strip():
        raise ValueError('Set schema_version=1 and a trial_id')
    for key in ('frame_evidence','uncertainty_evidence','notes'):
        if not isinstance(config[key],str):
            raise ValueError(f'{key} must be text')
    if config['frame_verified'] is not True or not config['frame_evidence'].strip():
        raise ValueError('Measured physical-up/tank frame and frame_evidence are required')
    hashes = config['calibration_hashes']
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Supply the recording calibration_hashes')
    if calibration_hashes is not None and hashes != calibration_hashes:
        raise ValueError('Configuration calibration hashes differ from this recording')
    matrix = np.asarray(config['board_to_tank_mm'], dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('board_to_tank_mm must be a finite 4x4 rigid transform')
    R = matrix[:3, :3]
    if not (np.allclose(matrix[3], [0, 0, 0, 1]) and np.allclose(R.T @ R, np.eye(3), atol=1e-7)
            and np.isclose(np.linalg.det(R), 1, atol=1e-7)):
        raise ValueError('Use a proper rigid rotation, without scaling or reflection')
    for key in ('still_water_z_mm', 'slice_x_mm'):
        finite(config[key], key)
    for key in ('slice_halfwidth_mm', 'max_gap_mm', 'max_host_gap_ms', 'min_wave_height_mm', 'max_track_gap_s'):
        finite(config[key], key, positive=True)
    interval = config['y_range_mm']
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError('y_range_mm needs two increasing endpoints')
    if finite(interval[1], 'y max') <= finite(interval[0], 'y min'):
        raise ValueError('y_range_mm must increase')
    if type(config['propagation_direction']) is not int or config['propagation_direction'] not in (-1, 1):
        raise ValueError('propagation_direction must be +1 or -1 along tank Y')
    for key in ('still_water_sigma_mm', 'pixel_sigma_px', 'systematic_position_sigma_mm', 'exposure_time_sigma_s'):
        if config[key] is not None:
            finite(config[key], key, nonnegative=True)
    if any(config[key] is not None for key in ('pixel_sigma_px', 'systematic_position_sigma_mm', 'exposure_time_sigma_s')) and not config['uncertainty_evidence'].strip():
        raise ValueError('Document uncertainty assumptions in uncertainty_evidence')
    for key, minimum in (('min_samples_per_wave', 5), ('smoothing_window', 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if config['smoothing_window'] % 2 != 1:
        raise ValueError('smoothing_window must be odd')
    if config['shape_reference'] not in ('local_wave_mean','still_water'):
        raise ValueError('shape_reference must be local_wave_mean or still_water')
    for key in ('still_water_depth_mm', 'driving_frequency_hz'):
        if config[key] is not None:
            finite(config[key], key, positive=True)
    for key in ('paddle_sweep_angle_deg', 'paddle_amplitude', 'ramp_start_y_mm'):
        if config[key] is not None:
            finite(config[key], key)
    if config['paddle_amplitude'] is not None and (not config['paddle_amplitude_units'] or config['paddle_amplitude_kind'] not in ('commanded', 'measured')):
        raise ValueError('Paddle amplitude requires units and commanded/measured kind')
    bottom = config['bottom_profile_yz_mm']
    if bottom is not None:
        b = np.asarray(bottom, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2 or len(b) < 2 or not np.isfinite(b).all() or np.any(np.diff(b[:, 0]) <= 0):
            raise ValueError('bottom_profile_yz_mm needs increasing measured Y,Z pairs')
    validation = config['experimental_validation']
    if not isinstance(validation, dict) or type(validation.get('verified')) is not bool or not isinstance(validation.get('evidence'), str):
        raise ValueError('experimental_validation needs verified and evidence')
    if validation['verified'] and not validation['evidence'].strip():
        raise ValueError('Experimental validation needs evidence')
    return config


def load_config(path, calibration_hashes=None):
    """Load and validate without changing the source configuration."""
    return validate_config(json.loads(Path(path).read_text()), calibration_hashes)
