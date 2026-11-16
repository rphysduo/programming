"""Command integration for tracking, reviewed detection and prediction."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from .reports import json_text, output_path, code_version
from .pipeline import digest, table


def add_commands(commands):
    """Register offline commands without changing the capture/calibration CLI."""
    track = commands.add_parser('track', help='Propose crest or packet identities for review')
    track.add_argument('--measurements', type=Path, required=True)
    track.add_argument('--output-dir', type=Path, required=True)
    track.add_argument('--max-speed-m-s', type=float, required=True)
    track.add_argument('--position-gate-m', type=float, default=.01)
    track.add_argument('--kind', choices=['crest', 'packet'], default='crest')
    suggest = commands.add_parser('suggest', help='Batch edge/match proposals in a new review folder')
    suggest.add_argument('--session', type=Path, required=True)
    suggest.add_argument('--annotations', type=Path, required=True)
    suggest.add_argument('--pairs', type=int, nargs='+', required=True)
    suggest.add_argument('--output-dir', type=Path, required=True)
    suggest.add_argument('--band-px', type=int, default=12)
    evaluate = commands.add_parser('evaluate-suggestions', help='Evaluate untouched machine traces against held-out human references')
    for name in ('session', 'reference', 'suggestions', 'output-dir'):
        evaluate.add_argument('--'+name, type=Path, required=True)
    evaluate.add_argument('--pairs', type=int, nargs='+', required=True)
    evaluate.add_argument('--tuning-pairs', type=int, nargs='+', required=True)
    evaluate.add_argument('--tolerance-px', type=float, default=3)
    train = commands.add_parser('train-predictor', help='Fit simple models using independent whole-trial splits')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--observation-time-s', type=float, required=True)
    train.add_argument('--lead-time-s', type=float, default=.1)
    train.add_argument('--locations', type=Path)
    train.add_argument('--synthetic-demo', action='store_true')
    predict = commands.add_parser('predict', help='Apply a saved model to early physical-feature CSV rows')
    predict.add_argument('--model-dir', type=Path, required=True)
    predict.add_argument('--features', type=Path, required=True)
    predict.add_argument('--output-dir', type=Path, required=True)


def run(args):
    """Dispatch extended commands; return None for the original CLI commands."""
    if args.command == 'track':
        from .auto_tracking import propose_tracks
        from .plots import plot_series
        source = args.measurements.resolve()
        manifest = json.loads((source/'analysis_manifest.json').read_text())
        if digest(source/'measurements.json') != manifest['measurements_sha256']:
            raise ValueError('Measurement file changed; regenerate it')
        data = json.loads((source/'measurements.json').read_text())
        # Without reviewed wave IDs, stopping all candidates at onset is conservative.
        breaks = [int(e['elapsed_ns'])/1e9 for e in data['breaking_events'] if e['observed']=='yes' and e.get('elapsed_ns') not in (None, '')]
        result = propose_tracks(data['profiles'], data['features'], data['config'], kind=args.kind,
            max_speed_m_s=args.max_speed_m_s, position_gate_m=args.position_gate_m, breaking_times=breaks)
        output = output_path(source, args.output_dir); output.mkdir(parents=True, exist_ok=False)
        (output/'track_links.json').write_text(json_text(result['tracks']))
        (output/'tracking_report.json').write_text(json_text(dict(result, source_sha256=digest(source/'measurements.json'), code=code_version())))
        plot_series(output/'trajectories.svg', [[(o['time_s'], o['y_m']) for o in t['observations']] for t in result['tracks']],
                    'Machine track proposals: human review required', 'Actual time (s)', 'Tank Y (m)')
        return dict(output_dir=str(output), proposed_tracks=len(result['tracks']), reviewed=False)
    if args.command in ('suggest', 'evaluate-suggestions'):
        from .session_io import WaveSession
        from .annotations import AnnotationStore
        from .suggestions import suggest_document, evaluate_traces
        session = WaveSession(args.session)
        source_dirs = [args.annotations] if args.command == 'suggest' else [args.reference, args.suggestions]
        for folder in source_dirs:
            if not (folder/'annotation_manifest.json').is_file(): raise ValueError('Input annotation manifest missing')
        stores = [AnnotationStore(session, folder) for folder in source_dirs]
        indices = sorted(set(args.pairs))
        if args.command == 'evaluate-suggestions':
            used = [set(), set()]
            for i in args.tuning_pairs:
                for c, frame in enumerate(session.pair_info(i)['frame_indices']): used[c].add(frame)
            for i in indices:
                if any(frame in used[c] for c, frame in enumerate(session.pair_info(i)['frame_indices'])):
                    raise ValueError('Held-out/tuning pairs share a camera frame; choose independent frames')
        prepared = []
        for index in indices:
            doc = stores[0].load(index)
            if args.command == 'suggest':
                images = [session.frame(c, doc['pair']['cameras'][c]['frame_index']) for c in ('cam0', 'cam1')]
                proposal = suggest_document(images, doc, session.cal, args.band_px)
                doc['points'].extend(proposal['points']); doc['traces'].extend(proposal['traces'])
                doc['suggestion_diagnostics'] = proposal['diagnostics']
                prepared.append((index, doc, images))
            else:
                proposed = stores[1].load(index)
                for camera in ('cam0', 'cam1'):
                    truth = [t['vertices_px'] for t in doc['traces'] if t['camera']==camera and t['origin']=='human']
                    machine = [t for t in proposed['traces'] if t['camera']==camera and t['origin']=='machine']
                    if any(t.get('human_reviewed') or t.get('machine_edited') or t.get('suggested_vertices_px') != t['vertices_px'] for t in machine): raise ValueError('Evaluate untouched proposals before human correction')
                    if not truth: raise ValueError('Every held-out view needs a manual reference trace')
                    prepared.append(dict(pair_index=index, camera=camera, reference_revision=doc['revision'],
                        suggestion_revision=proposed['revision'], **evaluate_traces(truth, [t['vertices_px'] for t in machine], args.tolerance_px)))
        output = output_path(session.path, args.output_dir)
        for folder in source_dirs: output_path(folder, output)
        output.mkdir(parents=True, exist_ok=False)
        if args.command == 'suggest':
            destination = AnnotationStore(session, output)
            for index, doc, images in prepared:
                destination.save(index, doc, 0)
                for camera, image in zip(('cam0', 'cam1'), images):
                    for trace in doc['traces']:
                        if trace['camera']==camera and trace['origin']=='machine':
                            cv2.polylines(image, [np.asarray(trace['vertices_px'], np.int32)], False, (0, 255, 255), 1)
                    if not cv2.imwrite(str(output/f'preview_{index:05d}_{camera}.png'), image): raise OSError('Cannot save suggestion preview')
            return dict(output_dir=str(output), pairs=len(prepared), next_step='Open annotate with this output folder and review suggestions')
        table(output/'heldout_errors.csv', prepared, [])
        (output/'evaluation.json').write_text(json_text(dict(results=prepared, tuning_pairs=args.tuning_pairs,
            heldout_pairs=indices, code=code_version(), source_files=session.sources,
            annotation_inputs={str(p):digest(p) for folder in source_dirs for p in folder.rglob('*.json')},
            limitation='Conditional on manually supplied search bands. Trace agreement is not stereo physical accuracy.')))
        return dict(output_dir=str(output), evaluated_views=len(prepared))
    if args.command == 'train-predictor':
        from .prediction import train_prediction
        return train_prediction(args.dataset, args.output_dir, observation_time_s=args.observation_time_s,
            lead_time_s=args.lead_time_s, locations_path=args.locations, synthetic=args.synthetic_demo)
    if args.command == 'predict':
        from .prediction import infer, numeric
        path = args.model_dir/'model.json'; manifest = json.loads((args.model_dir/'manifest.json').read_text())
        if digest(path) != manifest['model_sha256']: raise ValueError('Saved model fingerprint changed')
        model = json.loads(path.read_text())
        with args.features.open(newline='') as stream: rows = list(csv.DictReader(stream))
        for row in rows:
            uncertainty = numeric(row.get('exposure_time_sigma_s'))
            if not np.isfinite(uncertainty) or uncertainty < 0 or not np.isfinite(numeric(row.get('latest_frame_time_s'))) or numeric(row['latest_frame_time_s'])+2*uncertainty > model['observation_time_s']:
                raise ValueError('Predict inputs must be early snapshots at/before the model cutoff')
        uses_amplitude = 'paddle_amplitude' in model['classification']['features'] or (model['location'] is not None and 'paddle_amplitude' in model['location']['features'])
        if uses_amplitude:
            expected = model.get('amplitude_units')
            if not expected or any([r.get('paddle_amplitude_units'), r.get('paddle_amplitude_kind')] != expected[0] for r in rows if r.get('paddle_amplitude') not in ('', None)):
                raise ValueError('Prediction paddle amplitude units/kind do not match training')
        probabilities = infer(rows, model['classification'])
        locations = [None]*len(rows) if model['location'] is None else infer(rows, model['location'])
        output = output_path(args.model_dir, args.output_dir)
        output_path(args.features, output); output.mkdir(parents=True, exist_ok=False)
        table(output/'predictions.csv', [dict(trial_id=r.get('trial_id'), wave_id=r.get('wave_id'),
            breaking_probability=p, conditional_breaking_y_m=y, synthetic_model=model['synthetic']) for r, p, y in zip(rows, probabilities, locations)], [])
        (output/'manifest.json').write_text(json_text(dict(model_sha256=digest(path), features_sha256=digest(args.features), code=code_version())))
        return dict(output_dir=str(output), predictions=len(rows))
    return None
