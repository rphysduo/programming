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
    args = parser.parse_args(argv)
    try:
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
            result = extract_annotations(session, AnnotationStore(session, args.from_annotations), args.output_dir)
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
