"""Read-only audit and headless output helpers for the dry-board diagnostic."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np

from validate_stereo_3d import load_calibration, sha256_file


def json_safe(value):
    """Serialize missing geometry as null, never nonstandard JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_calibration(paths, spec):
    """Record source files, schemas and conventions without fitting any poses."""
    paths = {k: Path(v) for k, v in paths.items()}
    cal = load_calibration(paths)
    files = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            files[name] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                           "size_bytes": path.stat().st_size,
                           "schema": {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)}
                                      for k in data.files}}
    tank = cal.tank
    for key, expected in (("board_size", list(spec.grid)), ("square_length_mm", spec.square_mm),
                          ("marker_length_mm", spec.marker_mm)):
        if key not in tank or not np.allclose(tank[key], expected):
            raise ValueError(f"world file {key} does not match the selected original board")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=Path(__file__).parent,
                               capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, []
    return {
        "files": files, "checks": cal.checks,
        "software": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__, "git_head": revision, "modified_tracked_files": dirty,
                     "diagnostic_source_sha256": {
                         name: sha256_file(Path(__file__).with_name(name)) for name in (
                             "charuco_geometry_diagnostic.py", "charuco_diagnostic_report.py",
                             "charuco_common.py", "validate_stereo_3d.py")}},
        "image_size": list(cal.size0),
        "board": {"name": spec.name, "grid": list(spec.grid), "square_mm": spec.square_mm,
                  "marker_mm": spec.marker_mm, "dictionary": "DICT_5X5_250", "legacy_pattern": False,
                  "corner_count": len(spec.board.getChessboardCorners()),
                  "origin": "original untrimmed outer corner; never inferred from remaining strip"},
        "tank_transform": cal.T_board_to_tank.tolist(),
        "tank_transform_identity_placeholder": cal.tank_transform_is_identity,
        "tank_transform_verified": False,
        "stereo_scale": str(tank.get("stereo_scale", "unknown")),
        "conventions": {
            "stereo": "X_cam1 = R @ X_cam0 + T; T in mm in cam1 axes",
            "world": "X_cami = R_board_to_cami @ X_board + t_board_to_cami",
            "inverse_world": "X_board = R_board_to_cam0.T @ (X_cam0 - t_board_to_cam0)",
            "camera_axes": "OpenCV x right, y down, z forward; proper right-handed rotations",
            "pixels": "raw original resolution, top-left origin; no rotate/flip/crop/rectify",
            "board_axes": "+X short edge, +Y long edge; +Z is board normal, not verified tank-up",
            "residuals": "estimated/projected minus known/observed",
            "ray_plane": "only independent ray intersections use Z=0; free stereo is unchanged",
            "camera_identity": "file indices must match physical camera serials/devices; hashes cannot prove wiring",
        },
    }


def error_statistics(rows):
    """Bias/RMSE/max and longitude bins over all analyzed rows, including invalids.

    Only undefined nonfinite quantities are omitted; their count is explicit.
    No robust trimming is applied to geometric errors.
    """
    def stats(values):
        a = np.asarray(values, float)
        finite = a[np.isfinite(a)]
        return {"finite_count": len(finite), "undefined_count": int(a.size - finite.size),
                "mean": float(np.mean(finite)) if finite.size else None,
                "rmse": float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
                "max_abs": float(np.max(np.abs(finite))) if finite.size else None}

    result = {key: stats([r[key] for r in rows]) for key in (
        "cam0_world_reproj_error_px", "cam1_world_reproj_error_px",
        "cam0_stereo_reproj_error_px", "cam1_stereo_reproj_error_px", "stereo_error_mm")}
    result["signed_stereo_axes_mm"] = {
        axis: stats([r["stereo_error_vector_mm"][i] for r in rows]) for i, axis in enumerate("xyz")}
    result["invalid_geometric_count"] = sum(not r["valid"] for r in rows)
    # Original long-board Y rows expose longitudinal/seam structure without fitting corrections.
    groups = {}
    for row in rows:
        y = float(row["known_board_mm"][1])
        groups.setdefault(y, []).append(row)
    result["by_board_y_mm"] = [{"y_mm": y, "count": len(group),
                                "signed_dz_mm": stats([r["stereo_error_vector_mm"][2] for r in group]),
                                "world_cam0_px": stats([r["cam0_world_reproj_error_px"] for r in group]),
                                "world_cam1_px": stats([r["cam1_world_reproj_error_px"] for r in group])}
                               for y, group in sorted(groups.items())]
    return result


def annotate_image(image, camera, points, duplicates, roi, result=None):
    """Raw-size overlay: detected circles, projected crosses and residual arrows."""
    out = image.copy()
    if roi is not None:
        cv2.polylines(out, [np.rint(roi).astype(np.int32)], True, (255, 255, 0), 2)
    candidates = {r["corner_id"]: r for r in (result or {}).get("selection", {}).get("candidates", [])}
    analyzed = {r["corner_id"]: r for r in (result or {}).get("corners", [])}
    for cid, pixel in points.items():
        p = tuple(np.rint(pixel).astype(int))
        entry = candidates.get(cid, {})
        color = (0, 210, 0) if entry.get("selected") else (0, 160, 255)
        cv2.circle(out, p, 4, color, 1)
        row = analyzed.get(cid)
        label = str(cid)
        if row:
            projected = row[f"cam{camera}_known_projected_px"]
            if np.all(np.isfinite(projected)) and np.max(np.abs(projected)) < 1e6:
                q = tuple(np.rint(projected).astype(int))
                cv2.drawMarker(out, q, (255, 0, 255), cv2.MARKER_CROSS, 9, 1)
                cv2.arrowedLine(out, p, q, (255, 0, 255), 1, tipLength=0.2)
            label += f" dz={row['stereo_error_vector_mm'][2]:+.1f}mm"
            if row["flags"]:
                label += " !"
        elif entry.get("exclusion"):
            label += " " + entry["exclusion"]
        cv2.putText(out, label, (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 85), (20, 20, 20), -1)
    for j, line in enumerate((f"cam{camera} RAW pixels | ROI must enclose DIRECT board only; review before interpreting",
                              "circle=detected; magenta cross=known point via saved world pose; dz=FREE stereo error",
                              f"green=analyzed; orange=excluded/unmatched; duplicate IDs excluded: {duplicates}")):
        cv2.putText(out, line, (15, 22 + 24*j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def write_visuals(result, image0, image1, detector, output):
    """Write native-size annotated evidence and optional headless scientific plots."""
    from charuco_geometry_diagnostic import detect_charuco_points
    output = Path(output)
    for i, image in enumerate((image0, image1)):
        roi = result["selection"][f"roi_cam{i}"]
        points, duplicates = detect_charuco_points(image, detector, roi)
        out = annotate_image(image, i, points, duplicates, roi, result)
        if not cv2.imwrite(str(output / f"cam{i}_annotated.png"), out):
            raise OSError("could not write annotated image")
    # Optional on the Jetson: the numerical diagnostic and OpenCV overlays need no Matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Matplotlib unavailable: skipped error_pattern.png and board_3d.png"
    rows = result["corners"]
    known = np.array([r["known_board_mm"] for r in rows])
    stereo = np.array([r["stereo_board_via_cam0_mm"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for i, axis in enumerate("xyz"):
        axes[0].scatter(known[:, 1], (stereo-known)[:, i], label=f"d{axis}", s=18)
    axes[0].axhline(0, color="gray", linewidth=0.7)
    axes[0].set(xlabel="Known board Y (mm)", ylabel="Signed stereo error (mm)", title="All analyzed corners; invalid results retained")
    axes[0].legend()
    for i in (0, 1):
        axes[1].scatter(known[:, 1], [r[f"cam{i}_world_reproj_error_px"] for r in rows], label=f"cam{i} world")
    axes[1].set(xlabel="Known board Y (mm)", ylabel="Reprojection error (px)")
    axes[1].legend()
    fig.savefig(output / "error_pattern.png", dpi=140)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*known.T, c="black", marker="+", label="Known board")
    finite = np.all(np.isfinite(stereo), axis=1)
    ax.scatter(*stereo[finite].T, c="crimson", label="Free stereo (including invalid)")
    for a, b in zip(known[finite], stereo[finite]):
        ax.plot(*np.stack([a, b]).T, color="gray", linewidth=0.6)
    data = np.vstack([known, stereo[finite]])
    center = (data.min(0) + data.max(0)) / 2
    half = max(float(np.ptp(data, axis=0).max()) / 2, 1) * 1.05
    for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(center[i] - half, center[i] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="Board X (mm)", ylabel="Board Y (mm)", zlabel="Board Z (mm)", title="Equal metric axes; board +Z is not verified physical up")
    ax.legend()
    fig.savefig(output / "board_3d.png", dpi=140)
    plt.close(fig)
    return None
