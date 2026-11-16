#!/usr/bin/env python3
"""Exercise phases 4/5/7 on artificial signals and labels, not tank evidence."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from demo_wave_analysis import run as profiles_demo
from wave_analysis.auto_tracking import propose_tracks
from wave_analysis.suggestions import suggest_trace, evaluate_traces
from wave_analysis.prediction import train_prediction
from wave_analysis.pipeline import table, digest
from wave_analysis.reports import json_text, output_path


def run(output_dir):
    """Write synthetic profiles, tracks, image proposals and predictive artifacts."""
    output = output_path('/synthetic_demo_source', output_dir)
    output.mkdir(parents=True, exist_ok=False)
    profiles_demo(output/'profiles')
    data = json.loads((output/'profiles/measurements/measurements.json').read_text())
    result = propose_tracks(data['profiles'], data['features'], data['config'], max_speed_m_s=.3)
    (output/'automatic_tracks.json').write_text(json_text(result))
    image = np.zeros((120, 320, 3), np.uint8)
    truth = []
    for x in range(320):
        y = round(60+8*np.sin(x/30)); image[y:, x] = 200; truth.append([x, y])
    suggestion = suggest_trace(image, [[0,60], [319,60]], band_px=15)
    for line in suggestion['segments']:
        cv2.polylines(image, [np.asarray(line, np.int32)], False, (0,0,255), 1)
    cv2.imwrite(str(output/'synthetic_edge_proposals.png'), image)
    (output/'synthetic_detection.json').write_text(json_text(dict(suggestion=suggestion,
        evaluation=evaluate_traces([truth], suggestion['segments']))))
    rows, locations = [], []
    rng = np.random.default_rng(1729)
    for split in ('train', 'validation', 'test'):
        for i in range(12):
            label = i%2; trial = f'SYNTHETIC_{split}_{i}'
            rows.append(dict(trial_id=trial, wave_id='w', split=split, experimentally_validated=False,
                breaking_observed='yes' if label else 'no', onset_time_s=2 if label else None,
                prebreaking_by_recorded_times=True, latest_frame_time_s=.5, exposure_time_sigma_s=.002,
                steepness=.12+.18*label+rng.normal(0,.03), kh=1+i*.1, H_over_h=.1+.03*label,
                skewness=.1*label, asymmetry=-.1*label, causal_phase_speed_m_s=.15,
                paddle_sweep_angle_deg=15+10*label, driving_frequency_hz=1, still_water_depth_mm=150))
            if label: locations.append(dict(trial_id=trial, wave_id='w', y_m=.8+.01*i,
                verified=True, evidence='Artificial label, not a measured onset'))
    source=output/'toy_dataset'; source.mkdir()
    table(source/'research_dataset.csv', rows, [])
    (source/'analysis_manifest.json').write_text(json_text(dict(data_kind='synthetic',
        dataset_sha256=digest(source/'research_dataset.csv'),
        note='Artificial class-dependent feature distributions; no hydrodynamic breaking model')))
    locations_path=output/'toy_locations.json'; locations_path.write_text(json_text(locations))
    # Model outputs must be separate from the input dataset.
    prediction=train_prediction(source, output/'toy_models', observation_time_s=1,
                                locations_path=locations_path, synthetic=True)
    return dict(output_dir=str(output), automatic_tracks=len(result['tracks']),
                prediction=prediction, status='Synthetic software exercise only')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    print(run(parser.parse_args().output_dir))
