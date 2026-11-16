#!/usr/bin/env python3
"""Inspect recorded wave trials and annotate their original image coordinates."""
import argparse
from pathlib import Path
import sys

from wave_analysis.reports import inspect_trial


def main(argv=None):
    """Run offline analysis commands with explicit session and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='Read-only timing, video and calibration audit')
    inspect.add_argument('--session', type=Path, required=True)
    inspect.add_argument('--output-dir', type=Path, required=True)
    inspect.add_argument('--samples', type=int, default=9)
    annotate = commands.add_parser('annotate', help='Review points, traces, masks and breaking events')
    annotate.add_argument('--session', type=Path, required=True)
    annotate.add_argument('--output-dir', type=Path, required=True)
    annotate.add_argument('--host', default='127.0.0.1')
    annotate.add_argument('--port', type=int, default=5005)
    extract = commands.add_parser('extract', help='Export diagnostic geometry from saved manual annotations')
    extract.add_argument('--session', type=Path, required=True)
    extract.add_argument('--from-annotations', type=Path, required=True)
    extract.add_argument('--output-dir', type=Path, required=True)
    extract.add_argument('--config', type=Path, help='Measured physical frame and analysis settings')
    configure = commands.add_parser('init-config', help='Write an incomplete session-bound lab configuration')
    configure.add_argument('--session', type=Path, required=True)
    configure.add_argument('--output', type=Path, required=True)
    measure = commands.add_parser('measure', help='Measure physical profiles and optionally reviewed tracks')
    measure.add_argument('--analysis-dir', type=Path, required=True)
    measure.add_argument('--output-dir', type=Path)
    measure.add_argument('--tracks', type=Path, help='Reviewed crest or packet track links JSON')
    dataset = commands.add_parser('dataset', help='Join trials and report research readiness; does not train models')
    dataset.add_argument('--analyses', type=Path, nargs='+', required=True)
    dataset.add_argument('--output-dir', type=Path, required=True)
    dataset.add_argument('--split-policy', type=Path)
    from wave_analysis.automation_cli import add_commands, run
    add_commands(commands)
    args = parser.parse_args(argv)
    try:
        extended = run(args)
        if extended is not None:
            print(extended)
            return 0
        if args.command == 'init-config':
            from wave_analysis.config import template
            from wave_analysis.session_io import WaveSession
            from wave_analysis.reports import output_path, json_text
            session = WaveSession(args.session)
            destination = output_path(session.path, args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = template()
            data.update(trial_id=session.path.name, calibration_hashes=session.cal.hashes)
            with destination.open('x') as stream:
                stream.write(json_text(data))
            print(f'Fill measured settings in {destination}; this incomplete template cannot produce physical profiles yet.')
            return 0
        if args.command == 'measure':
            from wave_analysis.pipeline import measure_analysis
            print(measure_analysis(args.analysis_dir, args.output_dir, args.tracks))
            return 0
        if args.command == 'dataset':
            import json
            from wave_analysis.dataset import build_dataset
            policy = None if args.split_policy is None else json.loads(args.split_policy.read_text())
            print(build_dataset(args.analyses, args.output_dir, policy))
            return 0
        if args.command == 'annotate':
            from wave_analysis.annotation_web import create_app
            app = create_app(args.session, args.output_dir)
            print(f'Open http://{args.host}:{args.port} in your browser')
            app.run(host=args.host, port=args.port, threaded=True, debug=False)
            return 0
        if args.command == 'extract':
            from wave_analysis.annotations import AnnotationStore
            from wave_analysis.session_io import WaveSession
            from wave_analysis.stereo import extract_annotations
            session = WaveSession(args.session)
            if not (args.from_annotations / 'annotation_manifest.json').is_file():
                raise ValueError('No saved annotation manifest at --from-annotations')
            from wave_analysis.config import load_config
            config = None if args.config is None else load_config(args.config, session.cal.hashes)
            result = extract_annotations(session, AnnotationStore(session, args.from_annotations), args.output_dir, config)
            print(result)
            return 0
        report = inspect_trial(args.session, args.output_dir, samples=args.samples)
        for camera, data in report['streams'].items():
            print(f'{camera}: {data["decoded_frames"]} decoded frames, {data["observed_fps"]} observed fps')
        print(f'Report saved to {args.output_dir}')
        print(f'Integrity findings: {len(report["issues"])}')
        return 1 if report['issues'] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
