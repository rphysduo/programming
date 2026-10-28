#!/usr/bin/env python3
import argparse
import math
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a traveling 3D wave, stream it to the backend, and save a test snapshot."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Backend base URL")
    parser.add_argument("--username", default="user", help="Backend username")
    parser.add_argument("--password", default="123", help="Backend password")
    parser.add_argument("--test-name", default="jetson-3d-wave", help="Saved test name")
    parser.add_argument("--surface-prefix", default="surface", help="Prefix for generated sensor ids")
    parser.add_argument("--unit", default="meters", help="Sensor unit")
    parser.add_argument("--variance", type=float, default=0.000025, help="Sensor variance")
    parser.add_argument("--scale", type=float, default=1.0, help="Sensor scale")
    parser.add_argument("--offset", type=float, default=0.0, help="Sensor offset")
    parser.add_argument("--minimum-value", type=float, default=-1.0, help="Minimum z value")
    parser.add_argument("--maximum-value", type=float, default=1.0, help="Maximum z value")
    parser.add_argument("--active-variance", type=float, default=0.01, help="Pipeline active variance")
    parser.add_argument("--initial-variance", type=float, default=1.0, help="Pipeline initial variance")
    parser.add_argument("--grid-width", type=int, default=6, help="Number of points across x")
    parser.add_argument("--grid-height", type=int, default=5, help="Number of points across y")
    parser.add_argument("--spacing", type=float, default=0.2, help="Distance between adjacent points")
    parser.add_argument(
        "--length-width-ratio",
        type=float,
        default=6.0,
        help="Scale factor applied to x span relative to y span",
    )
    parser.add_argument("--count", type=int, default=120, help="Number of timestamps to generate")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between timestamps")
    parser.add_argument("--amplitude", type=float, default=0.12, help="Wave amplitude")
    parser.add_argument("--baseline", type=float, default=0.0, help="Wave baseline offset")
    parser.add_argument("--frequency", type=float, default=0.35, help="Temporal frequency in Hz")
    parser.add_argument("--x-phase-step", type=float, default=0.55, help="Phase delta per x step")
    parser.add_argument("--y-phase-step", type=float, default=0.35, help="Phase delta per y step")
    parser.add_argument("--radial-phase", type=float, default=0.18, help="Extra phase by radial distance from origin")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If > 0, save an intermediate snapshot every N timestamps",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Do not reset pipeline state before registering sensors and streaming points",
    )
    return parser.parse_args()


def api_request(session, method, url, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload


def login(session, base_url, username, password):
    payload = {"username": username, "password": password}
    return api_request(session, "POST", f"{base_url}/api/auth/login", json=payload)


def reset_pipeline(session, base_url, active_variance, initial_variance):
    payload = {
        "active_variance": active_variance,
        "initial_variance": initial_variance,
    }
    return api_request(session, "POST", f"{base_url}/api/reset", json=payload)


def save_snapshot(session, base_url, name):
    return api_request(session, "POST", f"{base_url}/api/tests", json={"name": name})


def stream_coordinate_point(session, base_url, point, timestamp, args):
    payload = {
        "sensor_id": point["sensor_id"],
        "x": point["x"],
        "y": point["y"],
        "unit": args.unit,
        "variance": args.variance,
        "scale": args.scale,
        "offset": args.offset,
        "minimum_value": args.minimum_value,
        "maximum_value": args.maximum_value,
        "rows": [{"time": timestamp, "z": point["z"]}],
    }
    return api_request(session, "POST", f"{base_url}/api/coordinate-data", json=payload)


def build_grid(args):
    points = []
    x_spacing = args.spacing * args.length_width_ratio
    y_spacing = args.spacing
    x_offset = (args.grid_width - 1) * x_spacing / 2
    y_offset = (args.grid_height - 1) * y_spacing / 2
    for y_index in range(args.grid_height):
        for x_index in range(args.grid_width):
            x = x_index * x_spacing - x_offset
            y = y_index * y_spacing - y_offset
            radial_distance = math.hypot(x, y)
            phase = (
                x_index * args.x_phase_step
                + y_index * args.y_phase_step
                + radial_distance * args.radial_phase
            )
            points.append(
                {
                    "sensor_id": f"{args.surface_prefix}_{y_index}_{x_index}",
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "phase": phase,
                }
            )
    return points


def build_wave_frame(points, timestamp, args):
    omega_t = 2 * math.pi * args.frequency * timestamp
    frame = []
    for point in points:
        z = args.baseline + args.amplitude * math.sin(omega_t + point["phase"])
        frame.append({**point, "z": z})
    return frame


def main():
    args = parse_args()
    session = requests.Session()
    base_url = args.base_url.rstrip("/")

    login(session, base_url, args.username, args.password)
    if not args.skip_reset:
        reset_pipeline(session, base_url, args.active_variance, args.initial_variance)

    points = build_grid(args)
    started_at = time.monotonic()

    for index in range(args.count):
        timestamp = round(index * args.interval, 6)
        frame = build_wave_frame(points, timestamp, args)
        for point in frame:
            stream_coordinate_point(session, base_url, point, timestamp, args)

        frame_number = index + 1
        if args.save_every and frame_number % args.save_every == 0:
            save_snapshot(session, base_url, f"{args.test_name}-{frame_number:05d}")

        target_elapsed = frame_number * args.interval
        actual_elapsed = time.monotonic() - started_at
        delay = target_elapsed - actual_elapsed
        if delay > 0:
            time.sleep(delay)

    result = save_snapshot(session, base_url, args.test_name)
    print(
        f"Saved test id={result['id']} name={result['name']} after streaming "
        f"{args.count} frames across {len(points)} coordinate sensors."
    )


if __name__ == "__main__":
    main()
