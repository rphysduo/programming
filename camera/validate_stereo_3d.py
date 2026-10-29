#!/usr/bin/env python3
"""Dry 3-D stereo validation for the wave-tank cameras.

This is a **measurement and validation** tool.  It never writes to
``cam0_intrinsics.npz``, ``cam1_intrinsics.npz``, ``stereo_params.npz`` or
``tank_alignment.npz`` and it never re-optimizes or rescales a calibration.

What it does
------------
1. Captures paired, full-resolution C920 frames in a browser (or headless
   ``--auto N``) and stores them under a validation session directory.  The
   live previews mark a preview origin (default: +4 board rows along +Y) and
   its +X/+Y/+Z axes; the overlay is not written into the saved frames.
2. Lets you click the same physical reference points in both images in a
   browser, with per-camera zoom, panning and a pixel magnifier.
   Correspondences are saved as raw full-resolution pixel coordinates, with
   derived pixel-space x/y/disparity and triangulated 3-D coordinates.
3. Triangulates the clicked correspondences with the saved stereo ``R``/``T``
   and intrinsics, using ``X_cam1 = R @ X_cam0 + T``.
4. Reports each point in Camera-0, big-ChArUco-board and, only when explicitly
   marked verified, tank coordinates.
5. Compares reconstructed lengths/heights/positions with ground-truth values
   from a user-edited measurement plan, with per-capture and summary statistics.
6. Writes annotated images, an equal-scale 3-D plot, JSON/CSV results and a
   readable summary into the session's ``results/`` directory.
7. Serves a browser results viewer at ``/results`` from the labeller, including
   tolerance status, point coordinates, diagnostics, images and downloads.
8. Serves a browser prediction-vs-actual page at ``/compare``; entering actual
   board XYZ values writes position tests to ``measurement_plan.json`` and
   compares them live with the triangulated predictions.

Typical use
-----------
::

    # 1. Capture pairs in a browser (open the printed URL; Ctrl+C stops)
    python validate_stereo_3d.py capture

    #    Headless smoke test, no browser:
    python validate_stereo_3d.py capture --auto 1 --interval 1.0

    # 2. Label correspondences in the browser (open the printed URL)
    python validate_stereo_3d.py label

    # 3. Generate a measurement-plan template, edit truth values, then run
    python validate_stereo_3d.py plan
    python validate_stereo_3d.py measure

    #    Or run the built-in synthetic end-to-end test:
    python validate_stereo_3d.py selftest

Session layout
--------------
::

    validation/
      session.json
      calibration/             # copies + hashes (originals are never modified)
      pairs/pair_0001_.../     # cam0.png, cam1.png, pair.json, correspondences.json
      measurement_plan.json
      results/                 # annotated PNGs, 3-D plot, JSON, CSV, summary.txt

Requirements implemented
------------------------
* Full-resolution raw frames are saved; previews are only downscaled.
* Browser-based correspondence selection with zoom/pan, magnifier, edit,
  delete-last and raw-pixel coordinate display.
* Labels save pixel-space x/y/disparity plus triangulated Camera-0 and board
  x/y/z coordinates; raw clicks remain authoritative.
* Browser results viewer for tolerance status, measurements, 3-D points,
  statistics, diagnostics, images and result-file downloads.
* Correct undistortion before triangulation; raw-pixel reprojection diagnostics.
* Front-of-camera, finite-value, epipolar, parallax and condition diagnostics.
* Known horizontal/vertical/3-D lengths and heights above the board/floor.
* Repeated captures are reported individually and as statistics.
* No automatic baseline change, rescaling or calibration adjustment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

try:  # Reuse the repository's verified acquisition settings when available.
    from charuco_common import open_camera
except Exception:  # pragma: no cover - only relevant if run outside the repo
    open_camera = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SESSION = "validation"
CALIB_NAMES = (
    "cam0_intrinsics.npz",
    "cam1_intrinsics.npz",
    "stereo_params.npz",
    "tank_alignment.npz",
)
DEFAULT_CALIB = {name: SCRIPT_DIR / name for name in CALIB_NAMES}

# Geometric thresholds are for point validity/diagnostics, not pass/fail.
# A real pass/fail tolerance must be supplied by the experimenter with
# --tolerance-mm.
MIN_PARALLAX_DEG = 0.20          # below this, depth is poorly conditioned
REJECT_PARALLAX_DEG = 0.01       # essentially parallel rays
HIGH_REPROJ_PX = 3.0             # flagged points are excluded from measurements
HIGH_EPIPOLAR_PX = 2.0           # flagged points are excluded from measurements
SCALE_WARN_PCT = 2.0


# ============================================================
# SMALL UTILITIES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_ready(obj: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, dict):
        return {key: _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(value) for value in obj]
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(data), fh, indent=2, sort_keys=False,
                  allow_nan=False)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path.resolve())


def check(checks: list[dict], level: str, code: str, message: str) -> None:
    checks.append({"level": level, "code": code, "message": message})


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def point_name(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    name = ""
    n = int(index)
    while True:
        name = chr(ord("A") + (n % 26)) + name
        n = n // 26 - 1
        if n < 0:
            return name


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


# ============================================================
# CALIBRATION
# ============================================================

@dataclass
class Calibration:
    paths: dict[str, Path]
    hashes: dict[str, str]
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray | None
    F: np.ndarray | None
    size0: tuple[int, int]
    size1: tuple[int, int]
    tank: dict[str, Any]
    checks: list[dict] = field(default_factory=list)

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def R_board_to_cam0(self) -> np.ndarray:
        return np.asarray(self.tank["R_board_to_cam0"], np.float64)

    @property
    def t_board_to_cam0(self) -> np.ndarray:
        return np.asarray(self.tank["t_board_to_cam0"], np.float64).reshape(3)

    @property
    def R_board_to_cam1(self) -> np.ndarray:
        return np.asarray(self.tank["R_board_to_cam1"], np.float64)

    @property
    def t_board_to_cam1(self) -> np.ndarray:
        return np.asarray(self.tank["t_board_to_cam1"], np.float64).reshape(3)

    @property
    def T_board_to_tank(self) -> np.ndarray:
        return np.asarray(self.tank["T_world_board"], np.float64)

    @property
    def tank_transform_is_identity(self) -> bool:
        T = self.tank.get("T_world_board")
        if T is None:
            return False
        T = np.asarray(T, np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return False
        return bool(np.allclose(T, np.eye(4), atol=1e-9))

    @property
    def errors(self) -> list[dict]:
        return [c for c in self.checks if c["level"] == "error"]

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.checks if c["level"] == "warn"]


def _validate_calibration(cal: Calibration) -> None:
    c = cal.checks

    # --- intrinsics / stereo cross-file consistency -----------------------
    intr0 = _load_npz(cal.paths["cam0_intrinsics.npz"])
    intr1 = _load_npz(cal.paths["cam1_intrinsics.npz"])

    for name, K, D, size in (
        ("cam0", cal.K0, cal.D0, cal.size0),
        ("cam1", cal.K1, cal.D1, cal.size1),
    ):
        if K.shape != (3, 3):
            check(c, "error", f"{name}_K_shape", f"camera matrix must be 3x3, got {K.shape}")
        if D.ndim != 1 or D.size not in (4, 5, 8, 12, 14):
            check(c, "warn", f"{name}_D_shape",
                  f"distortion coefficient count {D.size} is unusual")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            check(c, "error", f"{name}_nonfinite", "camera matrix or distortion is non-finite")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            check(c, "error", f"{name}_focal", "fx/fy must be positive")
        if size[0] <= 0 or size[1] <= 0:
            check(c, "error", f"{name}_size", f"invalid image size {size}")

    if cal.size0 != cal.size1:
        check(c, "error", "image_size_mismatch",
              f"intrinsic image sizes differ: {cal.size0} vs {cal.size1}")
    else:
        check(c, "ok", "image_size", f"both intrinsic files use {cal.size0[0]}x{cal.size1[1]}")

    if intr0["camera_matrix"].shape == (3, 3):
        dk = float(np.max(np.abs(intr0["camera_matrix"] - cal.K0)))
        dd = float(np.max(np.abs(intr0["dist_coeffs"].reshape(-1) - cal.D0)))
        if dk > 1e-9 or dd > 1e-9:
            check(c, "warn", "stereo_K0_mismatch",
                  f"stereo K0/D0 differs from cam0_intrinsics (max K {dk:.3g}, D {dd:.3g})")
        else:
            check(c, "ok", "stereo_K0", "stereo K0/D0 matches cam0_intrinsics")
    if intr1["camera_matrix"].shape == (3, 3):
        dk = float(np.max(np.abs(intr1["camera_matrix"] - cal.K1)))
        dd = float(np.max(np.abs(intr1["dist_coeffs"].reshape(-1) - cal.D1)))
        if dk > 1e-9 or dd > 1e-9:
            check(c, "warn", "stereo_K1_mismatch",
                  f"stereo K1/D1 differs from cam1_intrinsics (max K {dk:.3g}, D {dd:.3g})")
        else:
            check(c, "ok", "stereo_K1", "stereo K1/D1 matches cam1_intrinsics")

    # --- rigid transform sanity -------------------------------------------
    if cal.R.shape != (3, 3) or cal.T.size != 3:
        check(c, "error", "stereo_shape", f"R/T shapes are {cal.R.shape}/{cal.T.shape}")
    else:
        det = float(np.linalg.det(cal.R))
        orth = float(np.linalg.norm(cal.R @ cal.R.T - np.eye(3)))
        if not np.all(np.isfinite(cal.R)):
            check(c, "error", "stereo_R_nonfinite",
                  "stereo rotation matrix contains non-finite values")
        elif abs(det - 1.0) > 1e-4 or orth > 1e-4:
            check(c, "error", "stereo_R",
                  f"R is not a proper rotation (det={det:.8f}, orth_err={orth:.3g})")
        else:
            check(c, "ok", "stereo_R", f"R is a proper rotation (det={det:.8f})")
        if not np.all(np.isfinite(cal.T)) or cal.baseline_mm <= 0:
            check(c, "error", "stereo_T", "stereo translation is non-finite or zero")
        else:
            check(c, "ok", "stereo_T", f"|T| = {cal.baseline_mm:.3f} mm")
        if cal.E is not None and cal.E.shape != (3, 3):
            check(c, "warn", "stereo_E_shape", f"E shape {cal.E.shape}, expected (3,3)")
        if cal.F is not None and cal.F.shape != (3, 3):
            check(c, "warn", "stereo_F_shape", f"F shape {cal.F.shape}, expected (3,3)")
        # Verify that the saved E/F are consistent with the saved R/T and K.
        Tx = np.array([[0.0, -cal.T[2], cal.T[1]],
                       [cal.T[2], 0.0, -cal.T[0]],
                       [-cal.T[1], cal.T[0], 0.0]])
        E_calc = Tx @ cal.R
        F_calc = np.linalg.inv(cal.K1).T @ E_calc @ np.linalg.inv(cal.K0)
        if cal.E is not None and cal.E.shape == (3, 3) and np.linalg.norm(cal.E) > 0:
            e_rel = float(np.linalg.norm(cal.E / np.linalg.norm(cal.E)
                                         - E_calc / np.linalg.norm(E_calc)))
            if e_rel > 1e-5:
                check(c, "warn", "stereo_E_inconsistent",
                      f"saved E differs from [T]x R by {e_rel:.3g} (relative)")
            else:
                check(c, "ok", "stereo_E", "saved E is consistent with R/T")
        if cal.F is not None and cal.F.shape == (3, 3) and np.linalg.norm(cal.F) > 0:
            f_rel = float(np.linalg.norm(cal.F / np.linalg.norm(cal.F)
                                         - F_calc / np.linalg.norm(F_calc)))
            if f_rel > 1e-5:
                check(c, "warn", "stereo_F_inconsistent",
                      f"saved F differs from K1^-T [T]x R K0^-1 by {f_rel:.3g} "
                      "(relative)")
            else:
                check(c, "ok", "stereo_F",
                      "saved F is consistent with K0/K1/R/T")

    # --- tank / world file -------------------------------------------------
    req = ("R_board_to_cam0", "t_board_to_cam0", "R_board_to_cam1",
           "t_board_to_cam1", "T_world_board")
    missing = [k for k in req if k not in cal.tank]
    if missing:
        check(c, "error", "tank_missing", f"tank_alignment.npz missing {missing}")
    else:
        for key in ("R_board_to_cam0", "R_board_to_cam1"):
            R = np.asarray(cal.tank[key], np.float64)
            if (R.shape != (3, 3) or not np.all(np.isfinite(R))
                    or abs(np.linalg.det(R) - 1.0) > 1e-4):
                check(c, "error", f"{key}_bad",
                      f"{key} is not a finite proper rotation")
        for key in ("t_board_to_cam0", "t_board_to_cam1"):
            t = np.asarray(cal.tank[key], np.float64)
            if t.size != 3 or not np.all(np.isfinite(t)):
                check(c, "error", f"{key}_bad",
                      f"{key} is not a finite length-3 vector")
        T = cal.T_board_to_tank
        if T.shape != (4, 4):
            check(c, "error", "T_world_board_shape",
                  f"T_world_board shape {T.shape}, expected (4,4)")
        elif not np.all(np.isfinite(T)):
            check(c, "error", "T_world_board_nonfinite",
                  "T_world_board contains non-finite values")
        elif abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-4:
            check(c, "error", "T_world_board_rot",
                  "T_world_board rotation is not proper")
        elif not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            check(c, "error", "T_world_board_homogeneous",
                  "T_world_board last row is not [0, 0, 0, 1]")
        tank_ready = (
            T.shape == (4, 4) and np.all(np.isfinite(T))
            and abs(np.linalg.det(T[:3, :3]) - 1.0) <= 1e-4
            and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9))
        if tank_ready:
            if cal.tank_transform_is_identity:
                check(c, "warn", "tank_identity",
                      "T_world_board is the identity default: board coordinates "
                      "are NOT verified tank coordinates")
            else:
                check(c, "ok", "tank_transform",
                      "T_world_board is non-identity; verify it is a measured "
                      "transform")
        if (T.shape == (4, 4) and "R_board_to_tank" in cal.tank
                and "t_board_to_tank" in cal.tank):
            try:
                aux = np.eye(4)
                aux[:3, :3] = np.asarray(cal.tank["R_board_to_tank"],
                                         np.float64)
                aux[:3, 3] = np.asarray(cal.tank["t_board_to_tank"],
                                        np.float64).reshape(3)
                if not np.allclose(aux, T, atol=1e-8):
                    check(c, "warn", "tank_transform_aux_mismatch",
                          "R_board_to_tank/t_board_to_tank disagree with "
                          "T_world_board.  This tool uses T_world_board and does "
                          "not silently substitute the auxiliary arrays.")
            except (TypeError, ValueError):
                check(c, "warn", "tank_transform_aux_invalid",
                      "R_board_to_tank/t_board_to_tank are malformed; ignored.")

    tank_size = cal.tank.get("image_size")
    if tank_size is not None:
        tank_size = tuple(int(v) for v in np.asarray(tank_size).reshape(-1))
        if tank_size != cal.size0:
            check(c, "error", "tank_image_size",
                  f"tank_alignment image_size {tank_size} != intrinsics {cal.size0}")
        else:
            check(c, "ok", "tank_image_size", f"tank alignment image size {tank_size}")

    if "board_size" in cal.tank:
        check(c, "ok", "board_geometry",
              f"board {cal.tank.get('board', '?')} size "
              f"{np.asarray(cal.tank['board_size']).tolist()} with "
              f"{float(np.asarray(cal.tank['square_length_mm'])):g} mm squares")

    if "stereo_scale" in cal.tank:
        scale_value = np.asarray(cal.tank["stereo_scale"]).tolist()
        provisional = bool(np.asarray(
            cal.tank.get("stereo_scale_provisional", False)).tolist())
        if provisional:
            check(c, "warn", "stereo_scale_provisional",
                  f"tank_alignment records stereo_scale={scale_value!r} and "
                  "marks the world-derived metric scale as provisional.  Do not "
                  "treat it as final.")
        else:
            check(c, "info", "stereo_scale",
                  f"tank_alignment records stereo_scale={scale_value!r}")

    # --- world vs stereo consistency (diagnostic only) --------------------
    world_shapes_ok = (
        np.asarray(cal.tank.get("R_board_to_cam0")).shape == (3, 3)
        and np.asarray(cal.tank.get("R_board_to_cam1")).shape == (3, 3)
        and np.asarray(cal.tank.get("t_board_to_cam0")).size == 3
        and np.asarray(cal.tank.get("t_board_to_cam1")).size == 3)
    if (not missing and cal.R.shape == (3, 3)
            and cal.T.shape == (3,) and world_shapes_ok):
        R_c = cal.R @ cal.R_board_to_cam0
        R_err = rotation_angle_deg(R_c, cal.R_board_to_cam1)
        t_pred = cal.R @ cal.t_board_to_cam0 + cal.T
        t_err = float(np.linalg.norm(t_pred - cal.t_board_to_cam1))
        check(c, "info", "world_stereo_rotation",
              f"world/stereo rotation consistency: {R_err:.3f} deg")
        check(c, "info", "world_stereo_translation",
              f"world/stereo translation consistency: {t_err:.3f} mm")
        centers0 = -cal.R_board_to_cam0.T @ cal.t_board_to_cam0
        centers1 = -cal.R_board_to_cam1.T @ cal.t_board_to_cam1
        world_b = float(np.linalg.norm(centers0 - centers1))
        check(c, "info", "baseline_compare",
              f"stereo |T| = {cal.baseline_mm:.3f} mm; independent world "
              f"camera-centre baseline = {world_b:.3f} mm; difference = "
              f"{world_b - cal.baseline_mm:+.3f} mm")
        if abs(world_b - cal.baseline_mm) > max(2.0, 0.05 * cal.baseline_mm):
            check(c, "warn", "baseline_disagreement",
                  "stereo baseline and independently fitted world baseline differ; "
                  "possible metric-scale/geometry error, board movement, or camera "
                  "movement after calibration")
        if R_err > 1.0:
            check(c, "warn", "rotation_disagreement",
                  "world/stereo rotations differ by more than 1 degree")
        if t_err > max(5.0, 0.10 * cal.baseline_mm):
            check(c, "warn", "translation_disagreement",
                  "world/stereo translations are inconsistent; the two "
                  "calibrations may not describe the same camera rig")


def load_calibration(paths: dict[str, Path]) -> Calibration:
    intr0 = _load_npz(paths["cam0_intrinsics.npz"])
    intr1 = _load_npz(paths["cam1_intrinsics.npz"])
    stereo = _load_npz(paths["stereo_params.npz"])
    tank = _load_npz(paths["tank_alignment.npz"])

    for name, data in (
        ("cam0_intrinsics.npz", intr0),
        ("cam1_intrinsics.npz", intr1),
        ("stereo_params.npz", stereo),
        ("tank_alignment.npz", tank),
    ):
        if not data:
            raise RuntimeError(f"{name} is empty")

    try:
        K0 = np.asarray(stereo["K0"], np.float64)
        D0 = np.asarray(stereo["D0"], np.float64).reshape(-1)
        K1 = np.asarray(stereo["K1"], np.float64)
        D1 = np.asarray(stereo["D1"], np.float64).reshape(-1)
        R = np.asarray(stereo["R"], np.float64)
        T = np.asarray(stereo["T"], np.float64).reshape(3)
    except KeyError as exc:
        raise RuntimeError(f"stereo_params.npz missing key {exc}") from None

    # The intrinsic files are authoritative for image size and are cross-checked
    # against stereo K/D above.
    size0 = tuple(int(v) for v in np.asarray(intr0["image_size"]).reshape(-1))
    size1 = tuple(int(v) for v in np.asarray(intr1["image_size"]).reshape(-1))

    cal = Calibration(
        paths={k: Path(v) for k, v in paths.items()},
        hashes={name: sha256_file(Path(path)) for name, path in paths.items()},
        K0=K0, D0=D0, K1=K1, D1=D1, R=R, T=T,
        E=np.asarray(stereo["E"], np.float64) if "E" in stereo else None,
        F=np.asarray(stereo["F"], np.float64) if "F" in stereo else None,
        size0=size0, size1=size1,
        tank=tank,
    )
    _validate_calibration(cal)
    return cal


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Rrel = Ra @ Rb.T
    cos = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ============================================================
# SESSION / CALIBRATION COPIES
# ============================================================

def session_dirs(session: Path) -> dict[str, Path]:
    return {
        "session": session,
        "calibration": session / "calibration",
        "pairs": session / "pairs",
        "results": session / "results",
    }


def ensure_session(session: Path, source_paths: dict[str, Path],
                   quiet: bool = False) -> dict[str, Path]:
    """Create the session and copy calibration files once.

    Existing session copies are never overwritten.  If the source file has
    changed since the session was created, a warning is printed but the measured
    session keeps its original calibration copies for reproducibility.
    """
    dirs = session_dirs(session)
    for key in ("calibration", "pairs", "results"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    manifest_path = dirs["calibration"] / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "created_iso": utc_now_iso(),
        "files": {},
    }

    for name, src in source_paths.items():
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(f"calibration file not found: {src}")
        dst = dirs["calibration"] / name
        src_hash = sha256_file(src)
        if dst.exists():
            dst_hash = sha256_file(dst)
            rec = manifest["files"].get(name)
            if rec and rec.get("sha256") != dst_hash:
                print(f"WARNING: session copy {dst} hash does not match its manifest")
            if src_hash != dst_hash:
                print(f"WARNING: {src} changed since this session was created; "
                      f"continuing with the session copy {dst}")
        else:
            shutil.copy2(src, dst)
        manifest["files"][name] = {
            "source_path": str(src.resolve()),
            "session_path": str(dst.resolve()),
            "sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size,
            "copied_iso": manifest["files"].get(name, {}).get("copied_iso",
                                                              utc_now_iso()),
        }
        if not quiet:
            print(f"Calibration: {name}  sha256={manifest['files'][name]['sha256'][:16]}…")

    manifest["updated_iso"] = utc_now_iso()
    write_json(manifest_path, manifest)
    write_json(dirs["session"] / "session.json", {
        "session": str(session.resolve()),
        "created_iso": manifest["created_iso"],
        "updated_iso": manifest["updated_iso"],
        "calibration_manifest": str(manifest_path.resolve()),
    })
    return {name: dirs["calibration"] / name for name in source_paths}


def list_pair_dirs(session: Path) -> list[Path]:
    pairs = session / "pairs"
    if not pairs.exists():
        return []
    return sorted(p for p in pairs.iterdir()
                  if p.is_dir() and (p / "cam0.png").exists()
                  and (p / "cam1.png").exists())


def _next_pair_number(session: Path) -> int:
    nums = []
    for p in list_pair_dirs(session):
        parts = p.name.split("_")
        if len(parts) >= 2 and parts[0] == "pair":
            try:
                nums.append(int(parts[1]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


# ============================================================
# RECONSTRUCTION
# ============================================================

def board_up_direction(cal: Calibration, override: str = "auto") -> dict:
    """Infer the physically upward direction in big-board coordinates.

    The board is lying on the tank floor.  Both cameras are above that plane, so
    the side of the board plane containing the camera centres is "up".  The
    current calibration happens to have +Z_board pointing *downward*, hence the
    result is -Z_board.
    """
    normal = np.array([0.0, 0.0, 1.0])
    c0 = -cal.R_board_to_cam0.T @ cal.t_board_to_cam0
    c1 = -cal.R_board_to_cam1.T @ cal.t_board_to_cam1
    z0 = float(c0 @ normal)
    z1 = float(c1 @ normal)
    mean_z = 0.5 * (z0 + z1)
    if override == "board+z":
        up = normal
        method = "user override: board +Z"
    elif override == "board-z":
        up = -normal
        method = "user override: board -Z"
    elif abs(mean_z) < 1e-6:
        up = normal
        method = "ambiguous (cameras lie on the board plane); defaulted to board +Z"
    else:
        up = normal if mean_z > 0 else -normal
        method = ("inferred from camera centres; cameras are on the "
                  f"{'positive' if mean_z > 0 else 'negative'} board-Z side")
    up = np.asarray(up, np.float64)
    up /= np.linalg.norm(up)
    return {
        "up_board": up.tolist(),
        "normal_board": normal.tolist(),
        "camera_centres_board_mm": [c0.tolist(), c1.tolist()],
        "camera_board_z_mm": [z0, z1],
        "mean_camera_board_z_mm": mean_z,
        "camera_height_above_board_mm": abs(mean_z),
        "method": method,
        "note": ("Board +Z points physically DOWN in the current calibration; "
                 "heights below are computed along the inferred up vector."
                 if up[2] < 0 else
                 "Board +Z points physically UP in the current calibration."),
    }


def triangulate_point(pt0: list[float], pt1: list[float],
                      cal: Calibration) -> dict:
    """Undistort, triangulate and diagnose one correspondence pair."""
    x0 = np.asarray(pt0, np.float64).reshape(1, 1, 2)
    x1 = np.asarray(pt1, np.float64).reshape(1, 1, 2)
    u0 = cv2.undistortPoints(x0, cal.K0, cal.D0).reshape(2)
    u1 = cv2.undistortPoints(x1, cal.K1, cal.D1).reshape(2)

    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([cal.R, cal.T.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P0, P1,
                               u0.reshape(2, 1), u1.reshape(2, 1)).reshape(4)
    w = float(Xh[3])
    finite = bool(np.all(np.isfinite(Xh)))
    X0 = np.full(3, np.nan)
    X1 = np.full(3, np.nan)
    if finite and abs(w) > 1e-15:
        X0 = Xh[:3] / w
        X1 = cal.R @ X0 + cal.T

    # Raw-pixel reprojection (projectPoints applies the distortion model).
    reproj0 = np.full(2, np.nan)
    reproj1 = np.full(2, np.nan)
    err0 = err1 = float("nan")
    if np.all(np.isfinite(X0)):
        p0, _ = cv2.projectPoints(X0.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                  cal.K0, cal.D0)
        p1, _ = cv2.projectPoints(X1.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                  cal.K1, cal.D1)
        reproj0 = p0.reshape(2)
        reproj1 = p1.reshape(2)
        err0 = float(np.linalg.norm(reproj0 - np.asarray(pt0, np.float64)))
        err1 = float(np.linalg.norm(reproj1 - np.asarray(pt1, np.float64)))

    # Sampson epipolar distance using the stored fundamental matrix.  F is
    # defined for undistorted pinhole pixel coordinates, not for distorted raw
    # pixels, so convert the normalized undistorted points back to ideal pixels.
    sampson = float("nan")
    if cal.F is not None:
        x0h = cal.K0 @ np.array([u0[0], u0[1], 1.0])
        x1h = cal.K1 @ np.array([u1[0], u1[1], 1.0])
        Fx0 = cal.F @ x0h
        Ftx1 = cal.F.T @ x1h
        denom = Fx0[0] ** 2 + Fx0[1] ** 2 + Ftx1[0] ** 2 + Ftx1[1] ** 2
        if denom > 1e-20:
            sampson = float(abs(x1h @ cal.F @ x0h) / math.sqrt(denom))

    # Parallax angle between the two viewing rays, expressed in camera-0.
    r0 = np.array([u0[0], u0[1], 1.0])
    r1 = cal.R.T @ np.array([u1[0], u1[1], 1.0])
    r0 /= np.linalg.norm(r0)
    r1 /= np.linalg.norm(r1)
    parallax = float(np.degrees(np.arccos(np.clip(np.dot(r0, r1), -1.0, 1.0))))

    z0 = float(X0[2]) if np.all(np.isfinite(X0)) else float("nan")
    z1 = float(X1[2]) if np.all(np.isfinite(X1)) else float("nan")
    f = 0.5 * (cal.K0[0, 0] + cal.K0[1, 1])
    depth_sensitivity = (float("nan") if not (f > 0 and cal.baseline_mm > 0 and z0 > 0)
                         else float(z0 * z0 / (f * cal.baseline_mm)))

    flags = []
    if not finite or not np.all(np.isfinite(X0)) or w == 0:
        flags.append("nonfinite")
    if not (np.isfinite(z0) and z0 > 0):
        flags.append("behind_cam0")
    if not (np.isfinite(z1) and z1 > 0):
        flags.append("behind_cam1")
    if np.isfinite(parallax) and parallax < MIN_PARALLAX_DEG:
        flags.append("low_parallax")
    if np.isfinite(err0) and err0 > HIGH_REPROJ_PX:
        flags.append("high_reproj_cam0")
    if np.isfinite(err1) and err1 > HIGH_REPROJ_PX:
        flags.append("high_reproj_cam1")
    if np.isfinite(sampson) and sampson > HIGH_EPIPOLAR_PX:
        flags.append("high_epipolar")

    # A point that fails a geometric consistency check is not silently used in
    # a measurement: it is retained for diagnosis but excluded from tests.
    valid = not any(f in flags for f in
                    ("nonfinite", "behind_cam0", "behind_cam1",
                     "high_reproj_cam0", "high_reproj_cam1", "high_epipolar"))
    if np.isfinite(parallax) and parallax < REJECT_PARALLAX_DEG:
        valid = False
        flags.append("rejected_parallel_rays")

    return {
        "X_cam0_mm": X0.tolist(),
        "X_cam1_mm": X1.tolist(),
        "valid": bool(valid),
        "flags": flags,
        "reproj_cam0_px": reproj0.tolist(),
        "reproj_cam1_px": reproj1.tolist(),
        "reproj_error_cam0_px": err0,
        "reproj_error_cam1_px": err1,
        "epipolar_sampson_px": sampson,
        "parallax_deg": parallax,
        "depth_z_cam0_mm": z0,
        "depth_z_cam1_mm": z1,
        "depth_sensitivity_mm_per_px": depth_sensitivity,
    }


def board_to_cam0(P_board: np.ndarray, cal: Calibration) -> np.ndarray:
    return cal.R_board_to_cam0 @ P_board + cal.t_board_to_cam0


def cam0_to_board(P_cam0: np.ndarray, cal: Calibration) -> np.ndarray:
    return cal.R_board_to_cam0.T @ (P_cam0 - cal.t_board_to_cam0)


def board_to_tank(P_board: np.ndarray, cal: Calibration) -> np.ndarray:
    T = cal.T_board_to_tank
    return T[:3, :3] @ P_board + T[:3, 3]


def _project_camera_point_px(point: np.ndarray, K: np.ndarray,
                             D: np.ndarray) -> list[float] | None:
    """Project a camera-frame point to raw, distorted pixels."""
    point = np.asarray(point, np.float64).reshape(3)
    if not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    try:
        px, _ = cv2.projectPoints(
            point.reshape(1, 1, 3), np.zeros((3, 1)),
            np.zeros((3, 1)), K, D)
    except cv2.error:
        return None
    uv = np.asarray(px, np.float64).reshape(2)
    return uv.tolist() if np.all(np.isfinite(uv)) else None


def project_board_frame_px(
        cal: Calibration,
        axis_length_mm: float = 50.0,
        origin_board_mm: np.ndarray | None = None
) -> dict[str, dict[str, Any]]:
    """Project a board-frame origin and its +X/+Y/+Z axes into each camera.

    The world calibration stores ``X_cam = R_board_to_cam @ X_board +
    t_board_to_cam``.  Returns raw distorted pixels because the live previews
    are raw camera frames, not undistorted images.
    """
    length = float(axis_length_mm)
    if not np.isfinite(length) or length <= 0.0:
        length = 50.0
    board_origin = (np.zeros(3, np.float64) if origin_board_mm is None
                    else np.asarray(origin_board_mm, np.float64).reshape(3))
    result: dict[str, dict[str, Any]] = {}
    cameras = (
        ("cam0", cal.K0, cal.D0, cal.R_board_to_cam0, cal.t_board_to_cam0),
        ("cam1", cal.K1, cal.D1, cal.R_board_to_cam1, cal.t_board_to_cam1),
    )
    for key, K, D, R, t in cameras:
        R = np.asarray(R, np.float64)
        origin = np.asarray(t, np.float64).reshape(3) + R @ board_origin
        axes: dict[str, list[float] | None] = {}
        for name, axis in (("X", (1.0, 0.0, 0.0)),
                           ("Y", (0.0, 1.0, 0.0)),
                           ("Z", (0.0, 0.0, 1.0))):
            endpoint = origin + R @ np.asarray(axis) * length
            axes[name] = _project_camera_point_px(endpoint, K, D)
        result[key] = {
            "origin": _project_camera_point_px(origin, K, D),
            "axes": axes,
        }
    return result


def draw_board_axes(frame: np.ndarray, origin: list[float] | None,
                    axes: dict[str, list[float] | None] | None,
                    labels_only: bool = False) -> None:
    """Draw the projected board +X/+Y/+Z axes and short labels in place."""
    if origin is None or len(origin) != 2 or not axes:
        return
    try:
        ox, oy = float(origin[0]), float(origin[1])
    except (TypeError, ValueError):
        return
    if not (np.isfinite(ox) and np.isfinite(oy)):
        return
    height, width = frame.shape[:2]
    if ox < 0 or ox > width - 1 or oy < 0 or oy > height - 1:
        return
    start = (int(round(ox)), int(round(oy)))
    colors = {
        "X": (0, 0, 255),
        "Y": (0, 220, 0),
        "Z": (255, 120, 0),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    for name in ("X", "Y", "Z"):
        end = axes.get(name)
        if end is None or len(end) != 2:
            continue
        try:
            ex, ey = float(end[0]), float(end[1])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(ex) and np.isfinite(ey)):
            continue
        tip = (int(round(min(max(ex, 2), width - 3))),
               int(round(min(max(ey, 2), height - 3))))
        color = colors[name]
        if not labels_only:
            cv2.arrowedLine(frame, start, tip, (0, 0, 0), 7,
                            cv2.LINE_AA, tipLength=0.20)
            cv2.arrowedLine(frame, start, tip, color, 3,
                            cv2.LINE_AA, tipLength=0.20)

        label = "+" + name
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        dx, dy = ex - ox, ey - oy
        distance = math.hypot(dx, dy)
        label_x, label_y = ex, ey
        if 0.0 < distance < 55.0:
            label_x = ox + dx / distance * 55.0
            label_y = oy + dy / distance * 55.0
        if label_x > width * 0.72:
            tx = label_x - tw - 10
        elif label_x < width * 0.28:
            tx = label_x + 10
        else:
            tx = label_x - tw / 2
        if label_y < height / 2:
            ty = label_y - 10
        else:
            ty = label_y + th + 10
        tx = int(min(max(tx, 4), width - 4 - tw))
        ty = int(min(max(ty, th + 4), height - 4))
        cv2.putText(frame, label, (tx, ty), font, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), font, scale,
                    color, thickness, cv2.LINE_AA)


def draw_board_origin(frame: np.ndarray, uv: list[float] | None,
                      label: str = "board origin (0,0,0)") -> None:
    """Draw a crosshair and label at the projected board origin in place."""
    if uv is None or len(uv) != 2:
        return
    try:
        u, v = float(uv[0]), float(uv[1])
    except (TypeError, ValueError):
        return
    if not (np.isfinite(u) and np.isfinite(v)):
        return
    height, width = frame.shape[:2]
    if u < 0 or u > width - 1 or v < 0 or v > height - 1:
        return
    x = int(round(u))
    y = int(round(v))

    color = (0, 215, 255)
    cv2.circle(frame, (x, y), 18, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 18, color, 3, cv2.LINE_AA)
    cv2.line(frame, (x - 27, y), (x + 27, y), (0, 0, 0), 6, cv2.LINE_AA)
    cv2.line(frame, (x, y - 27), (x, y + 27), (0, 0, 0), 6, cv2.LINE_AA)
    cv2.line(frame, (x - 27, y), (x + 27, y), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x, y - 27), (x, y + 27), color, 3, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 7
    gap = 12
    if x + 27 + gap + tw + 2 * pad <= width - 4:
        box_x = x + 27 + gap
    else:
        box_x = x - 27 - gap - tw - 2 * pad
    box_x = int(min(max(box_x, 4), width - 4 - tw - 2 * pad))
    if y - 27 - gap - th - 2 * pad >= 4:
        box_y = y - 27 - gap - th - 2 * pad
    else:
        box_y = y + 27 + gap
    box_y = int(min(max(box_y, 4), height - 4 - th - 2 * pad))

    cv2.rectangle(frame, (box_x, box_y),
                  (box_x + tw + 2 * pad, box_y + th + 2 * pad),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x, box_y),
                  (box_x + tw + 2 * pad, box_y + th + 2 * pad),
                  color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (box_x + pad, box_y + pad + th),
                font, scale, color, thickness, cv2.LINE_AA)


def fit_plane(points: np.ndarray) -> dict | None:
    """Least-squares plane fit.  Returns normal/centroid/offset in board frame."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    _, s, vh = np.linalg.svd(pts - centroid)
    if s.size < 3:
        return None
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return {
        "centroid_mm": centroid.tolist(),
        "normal": normal.tolist(),
        "offset_mm": float(centroid @ normal),
        "singular_values": s.tolist(),
    }


# ============================================================
# CORRESPONDENCES / PLAN
# ============================================================

def correspondence_path(pair_dir: Path) -> Path:
    return pair_dir / "correspondences.json"


def load_correspondences(pair_dir: Path) -> list[dict]:
    path = correspondence_path(pair_dir)
    if not path.exists():
        return []
    data = read_json(path)
    return list(data.get("points", []))


def validate_correspondences(points: list[dict],
                             size0: tuple[int, int],
                             size1: tuple[int, int]) -> list[str]:
    errors = []
    seen_ids = set()
    seen_names = set()
    seen_observations: dict[tuple[float, float, float, float], str] = {}
    for i, p in enumerate(points):
        name = str(p.get("name") or p.get("label") or f"P{i}")
        pid = p.get("id", i)
        if pid in seen_ids:
            errors.append(f"duplicate point id {pid}")
        if name in seen_names:
            errors.append(f"duplicate point name {name!r}")
        seen_ids.add(pid)
        seen_names.add(name)
        parsed = {}
        for cam, size in (("cam0", size0), ("cam1", size1)):
            xy = p.get(cam)
            if xy is None or len(xy) != 2:
                errors.append(f"{name}: missing {cam} observation")
                continue
            x, y = float(xy[0]), float(xy[1])
            parsed[cam] = (x, y)
            if not (math.isfinite(x) and math.isfinite(y)):
                errors.append(f"{name}: non-finite {cam} observation")
            elif not (0 <= x <= size[0] - 1 and 0 <= y <= size[1] - 1):
                errors.append(f"{name}: {cam} observation ({x:.2f}, {y:.2f}) "
                              f"outside image {size}")
        if len(parsed) == 2:
            key = (round(parsed["cam0"][0], 3), round(parsed["cam0"][1], 3),
                   round(parsed["cam1"][0], 3), round(parsed["cam1"][1], 3))
            other = seen_observations.get(key)
            if other is not None:
                errors.append(f"{name}: identical observation pair to point "
                              f"{other!r}; this is an ambiguous duplicate")
            else:
                seen_observations[key] = name
    return errors


def default_plan(point_names: list[str]) -> dict:
    return {
        "_instructions": (
            "Set truth_mm for real measured dimensions.  kind may be "
            "'euclidean', 'horizontal' or 'vertical'.  Height tests use the "
            "inferred up direction and either the board surface or a named "
            "reference point.  Set board_thickness_mm to the board's measured "
            "top-surface height above the tank floor.  Optionally set "
            "physical_baseline_mm for a comparison only; it never rescales "
            "the reconstruction.  Set "
            "tank_transform_verified=true only if the board->tank transform in "
            "tank_alignment.npz is a measured, verified transform.  "
            "floor_reference_points are used only for a least-squares "
            "plane-consistency check and the 3-D plot; they do not redefine the "
            "height zero (which is the board surface plus board_thickness_mm).  "
            "Use identical point and test names across repeated captures so "
            "per-test repeatability can be assessed."
        ),
        "board_thickness_mm": None,
        "physical_baseline_mm": None,
        "tank_transform_verified": False,
        "up_direction": "auto",
        "floor_reference_points": [],
        "length_tests": [],
        "height_tests": [],
        "position_tests": [],
        "available_points": point_names,
        "point_name_examples": {
            "length": {"name": "horizontal_100mm", "p1": "A", "p2": "B",
                       "kind": "horizontal", "truth_mm": 100.0},
            "vertical_length": {"name": "vertical_40mm", "p1": "C", "p2": "D",
                                "kind": "vertical", "truth_mm": 40.0},
            "height": {"name": "height_20mm", "point": "C",
                       "reference_point": None, "truth_mm": 20.0},
            "position": {"name": "known_board_position", "point": "A",
                         "truth_board_mm": [44.0, 650.0, 0.0]},
        },
    }


def load_or_create_plan(session: Path, point_names: list[str]) -> tuple[dict, Path, bool]:
    path = session / "measurement_plan.json"
    if path.exists():
        return read_json(path), path, False
    plan = default_plan(point_names)
    write_json(path, plan)
    return plan, path, True


def enabled(entry: dict) -> bool:
    value = entry.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def validate_plan(plan: dict) -> None:
    """Reject plan entries that would silently merge or mis-type results."""
    errors = []
    up_override = plan.get("up_direction", "auto")
    if up_override not in ("auto", "board+z", "board-z"):
        errors.append("up_direction must be auto, board+z or board-z")
    thickness = plan.get("board_thickness_mm")
    if thickness is not None:
        try:
            if not np.isfinite(float(thickness)) or float(thickness) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("board_thickness_mm must be null or a finite "
                          "non-negative value in millimetres")
    if not isinstance(plan.get("tank_transform_verified", False), bool):
        errors.append("tank_transform_verified must be true or false")
    physical_baseline = plan.get("physical_baseline_mm")
    if physical_baseline is not None:
        try:
            if (not np.isfinite(float(physical_baseline))
                    or float(physical_baseline) <= 0):
                raise ValueError
        except (TypeError, ValueError):
            errors.append("physical_baseline_mm must be null or a finite "
                          "positive value in millimetres")
    floor_names = plan.get("floor_reference_points", [])
    if not isinstance(floor_names, list):
        errors.append("floor_reference_points must be a list")
    elif any(not isinstance(n, str) or not n.strip() for n in floor_names):
        errors.append("floor_reference_points must contain non-empty names")
    names: set[str] = set()
    for key, test_type in (("length_tests", "length"),
                           ("height_tests", "height"),
                           ("position_tests", "position")):
        entries = plan.get(key, [])
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            errors.append(f"{key} must be a list")
            continue
        for index, test in enumerate(entries):
            if not isinstance(test, dict):
                errors.append(f"{key}[{index}] must be an object")
                continue
            if not enabled(test):
                continue
            name = test.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{key}[{index}] needs a non-empty name")
            elif name in names:
                errors.append(f"duplicate enabled test name {name!r}")
            else:
                names.add(name)
            if test_type == "length":
                for field in ("p1", "p2"):
                    if not test.get(field):
                        errors.append(f"{key}[{index}] needs {field}")
                if test.get("p1") and test.get("p1") == test.get("p2"):
                    errors.append(f"{key}[{index}] p1 and p2 must differ")
                kind = test.get("kind", "euclidean")
                if kind not in ("euclidean", "horizontal", "vertical"):
                    errors.append(
                        f"{key}[{index}] kind must be euclidean, horizontal or "
                        f"vertical, not {kind!r}")
                truth = test.get("truth_mm")
                try:
                    if truth is None or not np.isfinite(float(truth)) \
                            or float(truth) <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"{key}[{index}] truth_mm must be a finite "
                                  "positive length")
            elif test_type == "height":
                if not test.get("point"):
                    errors.append(f"{key}[{index}] needs point")
                truth = test.get("truth_mm")
                try:
                    if truth is None or not np.isfinite(float(truth)):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"{key}[{index}] truth_mm must be finite")
            else:
                if not test.get("point"):
                    errors.append(f"{key}[{index}] needs point")
                truth = test.get("truth_board_mm")
                if not isinstance(truth, (list, tuple)) or len(truth) != 3:
                    errors.append(
                        f"{key}[{index}] needs truth_board_mm with three values")
                else:
                    try:
                        if not np.all(np.isfinite(np.asarray(truth, np.float64))):
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(
                            f"{key}[{index}] truth_board_mm values must be finite")
    if errors:
        raise ValueError("Invalid measurement plan:\n  " + "\n  ".join(errors))


# ============================================================
# STATISTICS
# ============================================================

def summarize_errors(errors: list[float | None]) -> dict:
    arr = np.asarray([float(e) for e in errors
                      if e is not None and np.isfinite(float(e))], np.float64)
    if arr.size == 0:
        return {"n": 0}
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean_signed_mm": float(arr.mean()),
        "mean_abs_mm": float(np.abs(arr).mean()),
        "rmse_mm": float(np.sqrt(np.mean(arr ** 2))),
        "max_abs_mm": float(np.abs(arr).max()),
        "std_signed_mm": sd,
        "min_signed_mm": float(arr.min()),
        "max_signed_mm": float(arr.max()),
    }


def summarize_values(values: list[float | None]) -> dict:
    arr = np.asarray([float(v) for v in values
                      if v is not None and np.isfinite(float(v))], np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def scale_fit(truths: list[float | None], recons: list[float | None]) -> dict:
    pairs = [(float(t), float(r)) for t, r in zip(truths, recons)
             if t is not None and r is not None]
    t = np.asarray([p[0] for p in pairs], np.float64)
    r = np.asarray([p[1] for p in pairs], np.float64)
    mask = np.isfinite(t) & np.isfinite(r)
    t, r = t[mask], r[mask]
    out = {"n": int(t.size)}
    if t.size == 0:
        return out
    ratio_mask = np.abs(t) > 1e-12
    ratios = r[ratio_mask] / t[ratio_mask]
    if ratios.size:
        out["mean_ratio"] = float(ratios.mean())
        out["ratio_std"] = (float(np.std(ratios, ddof=1))
                            if ratios.size > 1 else 0.0)
        out["mean_scale_bias_pct"] = float((ratios.mean() - 1.0) * 100.0)
    if t.size >= 2 and np.var(t) > 1e-12:
        slope, intercept = np.polyfit(t, r, 1)
        pred = slope * t + intercept
        ss_res = float(np.sum((r - pred) ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        out.update({
            "linear_slope": float(slope),
            "linear_intercept_mm": float(intercept),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        })
    return out


# ============================================================
# MEASUREMENT ENGINE
# ============================================================

def _load_pair_meta(pair_dir: Path) -> dict:
    p = pair_dir / "pair.json"
    return read_json(p) if p.exists() else {}


def _test_value(test: dict, test_type: str, pts_board: dict[str, np.ndarray],
                up: np.ndarray, board_thickness: float | None) -> dict:
    kind = test.get("kind", "euclidean" if test_type == "length" else "height")
    if test_type == "length":
        p1, p2 = test["p1"], test["p2"]
        P1, P2 = pts_board[p1], pts_board[p2]
        d = P2 - P1
        signed_up = float(d @ up)
        if kind == "horizontal":
            value = float(np.linalg.norm(d - signed_up * up))
        elif kind == "vertical":
            value = abs(signed_up)
        else:
            value = float(np.linalg.norm(d))
        return {
            "value_mm": value,
            "kind": kind,
            "signed_up_component_mm": signed_up,
            "components_board_mm": d.tolist(),
            "p1": p1,
            "p2": p2,
        }

    point = test["point"]
    P = pts_board[point]
    ref_name = test.get("reference_point")
    if ref_name:
        if ref_name not in pts_board:
            raise KeyError(f"height test {test.get('name')!r} references missing "
                           f"point {ref_name!r}")
        signed = float((P - pts_board[ref_name]) @ up)
        basis = f"reference point {ref_name}"
    else:
        signed = float(P @ up)
        if board_thickness is not None:
            signed += float(board_thickness)
            basis = "board surface + board_thickness_mm (tank-floor reference)"
        else:
            basis = "board surface (board_thickness_mm not supplied)"
    return {
        "value_mm": signed,
        "kind": kind,
        "signed_up_component_mm": signed,
        "components_board_mm": P.tolist(),
        "p1": point,
        "p2": ref_name or "floor",
        "height_basis": basis,
    }


def _reconstruct_pair(pair_dir: Path, cal: Calibration, up: np.ndarray,
                      plan: dict) -> dict:
    points = load_correspondences(pair_dir)
    errs = validate_correspondences(points, cal.size0, cal.size1)
    if errs:
        raise ValueError(f"{pair_dir.name}: invalid correspondences:\n  " +
                         "\n  ".join(errs))
    if not points:
        raise ValueError(f"{pair_dir.name}: no correspondences saved")

    meta = _load_pair_meta(pair_dir)
    img0 = cv2.imread(str(pair_dir / "cam0.png"), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(pair_dir / "cam1.png"), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        raise ValueError(f"{pair_dir.name}: could not read cam0.png/cam1.png")
    if (img0.shape[1], img0.shape[0]) != cal.size0:
        raise ValueError(f"{pair_dir.name}: cam0 image size "
                         f"{img0.shape[1]}x{img0.shape[0]} != calibration {cal.size0}")
    if (img1.shape[1], img1.shape[0]) != cal.size1:
        raise ValueError(f"{pair_dir.name}: cam1 image size "
                         f"{img1.shape[1]}x{img1.shape[0]} != calibration {cal.size1}")

    point_results = {}
    pts_board = {}
    for point_index, p in enumerate(points):
        name = str(p.get("name") or p.get("label") or f"P{point_index}")
        rec = triangulate_point(p["cam0"], p["cam1"], cal)
        X_cam0 = np.asarray(rec["X_cam0_mm"], np.float64)
        if np.all(np.isfinite(X_cam0)):
            X_board = cam0_to_board(X_cam0, cal)
            if rec["valid"]:
                pts_board[name] = X_board
            rec["X_board_mm"] = X_board.tolist()
            rec["height_above_board_mm"] = float(X_board @ up)
            rec["height_above_tank_floor_mm"] = (
                None if plan.get("board_thickness_mm") is None
                else float(X_board @ up + float(plan["board_thickness_mm"])))
            tank_verified = (bool(plan.get("tank_transform_verified", False))
                             and not cal.tank_transform_is_identity)
            rec["X_tank_mm"] = (board_to_tank(X_board, cal).tolist()
                                if tank_verified else None)
            rec["X_tank_verified"] = tank_verified
        else:
            rec["X_board_mm"] = [float("nan")] * 3
            rec["height_above_board_mm"] = float("nan")
            rec["height_above_tank_floor_mm"] = None
            rec["X_tank_mm"] = None
            rec["X_tank_verified"] = False
        rec["name"] = name
        rec["cam0_raw_px"] = list(map(float, p["cam0"]))
        rec["cam1_raw_px"] = list(map(float, p["cam1"]))
        rec["pixel_xyz"] = [
            float(p["cam0"][0]),
            float(p["cam0"][1]),
            float(p["cam0"][0]) - float(p["cam1"][0]),
        ]
        point_results[name] = rec

    # A configured test must be measurable in every processed pair.  Silently
    # dropping a missing or invalid point would hide a failed correspondence.
    measurements = []
    for test_type, key in (("length", "length_tests"), ("height", "height_tests")):
        for test in plan.get(key, []):
            if not enabled(test):
                continue
            name = test.get("name") or f"{key}_{len(measurements)}"
            truth = test.get("truth_mm")
            if truth is None:
                raise ValueError(
                    f"measurement plan test {name!r} has no truth_mm; enter the "
                    "physically measured value before running measure")
            needed = ([test.get("p1"), test.get("p2")]
                      if test_type == "length"
                      else [test.get("point"), test.get("reference_point")])
            missing = [n for n in needed if n and n not in pts_board]
            if missing:
                details = []
                for point_name in missing:
                    p = point_results.get(point_name)
                    if p is not None:
                        details.append(f"{point_name} flags={p.get('flags', [])}")
                suffix = ("; " + "; ".join(details)) if details else ""
                raise KeyError(f"{pair_dir.name}: test {name!r} needs point(s) "
                               f"{missing} that are absent, non-finite, behind "
                               f"a camera, or otherwise invalid{suffix}")
            val = _test_value(test, test_type, pts_board, up,
                              plan.get("board_thickness_mm"))
            truth = float(truth) if truth is not None else None
            err = (val["value_mm"] - truth) if truth is not None else None
            pct = (100.0 * err / truth) if (err is not None and truth not in
                                            (None, 0)) else None
            measurements.append({
                "pair": pair_dir.name,
                "test": name,
                "type": test_type,
                "kind": val.get("kind", test.get("kind", "")),
                "p1": val["p1"],
                "p2": val["p2"],
                "truth_mm": truth,
                "reconstructed_mm": float(val["value_mm"]),
                "signed_error_mm": (None if err is None else float(err)),
                "abs_error_mm": (None if err is None else abs(float(err))),
                "pct_error": (None if pct is None else float(pct)),
                "signed_up_component_mm": val.get("signed_up_component_mm"),
                "components_board_mm": val.get("components_board_mm"),
                "height_basis": val.get("height_basis"),
            })

    # Optional known absolute positions in board coordinates.
    for test in plan.get("position_tests", []):
        if not enabled(test):
            continue
        name = test.get("name") or f"position_tests_{len(measurements)}"
        point = test.get("point")
        if not point or point not in pts_board:
            raise KeyError(f"{pair_dir.name}: position test {name!r} needs "
                           f"missing/invalid point {point!r}")
        truth = test.get("truth_board_mm")
        if truth is None or len(truth) != 3:
            raise ValueError(f"position test {name!r} needs truth_board_mm "
                             "with three values [x, y, z]")
        truth = np.asarray(truth, np.float64)
        P = pts_board[point]
        err = P - truth
        measurements.append({
            "pair": pair_dir.name,
            "test": name,
            "type": "position",
            "kind": "position",
            "p1": point,
            "p2": "truth_board",
            "truth_mm": None,
            "reconstructed_mm": None,
            "position_error_mm": float(np.linalg.norm(err)),
            "signed_error_mm": None,
            "abs_error_mm": float(np.linalg.norm(err)),
            "pct_error": None,
            "truth_board_mm": truth.tolist(),
            "reconstructed_board_mm": P.tolist(),
            "error_x_mm": float(err[0]),
            "error_y_mm": float(err[1]),
            "error_z_mm": float(err[2]),
        })

    valid_points = [r for r in point_results.values() if r["valid"]]
    reproj_sq = [r["reproj_error_cam0_px"] ** 2 + r["reproj_error_cam1_px"] ** 2
                 for r in valid_points
                 if np.isfinite(r["reproj_error_cam0_px"])
                 and np.isfinite(r["reproj_error_cam1_px"])]
    rms = float(np.sqrt(np.mean(reproj_sq))) if reproj_sq else float("nan")
    pair_result = {
        "pair": pair_dir.name,
        "pair_meta": meta,
        "paths": {
            "cam0": str((pair_dir / "cam0.png").resolve()),
            "cam1": str((pair_dir / "cam1.png").resolve()),
            "correspondences": str(correspondence_path(pair_dir).resolve()),
            "pair_meta": str((pair_dir / "pair.json").resolve()),
        },
        "image_size": [cal.size0[0], cal.size0[1]],
        "points": point_results,
        "measurements": measurements,
        "n_points": len(points),
        "n_valid": len(valid_points),
        "combined_reproj_rms_px": rms,
        "max_reproj_px": float(max(
            [max(r["reproj_error_cam0_px"], r["reproj_error_cam1_px"])
             for r in valid_points], default=float("nan"))),
        "min_parallax_deg": float(min(
            [r["parallax_deg"] for r in valid_points], default=float("nan"))),
        "images": {"cam0": img0, "cam1": img1},
    }
    return pair_result


def build_statistics(measurements: list[dict]) -> dict:
    by_test: dict[str, list[dict]] = {}
    for m in measurements:
        by_test.setdefault(m["test"], []).append(m)

    by_test_out = {}
    for name, rows in by_test.items():
        rows_sorted = sorted(rows, key=lambda r: r["pair"])
        vals = [r["reconstructed_mm"] for r in rows_sorted]
        errs = [r["signed_error_mm"] for r in rows_sorted
                if r["signed_error_mm"] is not None]
        by_test_out[name] = {
            "type": rows_sorted[0]["type"],
            "kind": rows_sorted[0]["kind"],
            "truth_mm": rows_sorted[0]["truth_mm"],
            "reconstructed": summarize_values(vals),
            "errors": summarize_errors(errs),
            "per_pair": [
                {"pair": r["pair"],
                 "reconstructed_mm": r["reconstructed_mm"],
                 "signed_error_mm": r["signed_error_mm"],
                 "abs_error_mm": r["abs_error_mm"],
                 "pct_error": r["pct_error"]}
                for r in rows_sorted
            ],
        }

    def group(pred):
        return [m for m in measurements if pred(m)]

    categories = {
        "all": group(lambda m: m["type"] != "position"),
        "length_euclidean": group(lambda m: m["type"] == "length" and m["kind"] not in
                                  ("horizontal", "vertical")),
        "horizontal_length": group(lambda m: m["type"] == "length" and
                                   m["kind"] == "horizontal"),
        "vertical_length": group(lambda m: m["type"] == "length" and
                                 m["kind"] == "vertical"),
        "height": group(lambda m: m["type"] == "height"),
    }
    categories_out = {}
    for name, rows in categories.items():
        errs = [r["signed_error_mm"] for r in rows if r["signed_error_mm"] is not None]
        if not errs:
            continue
        truths = [r["truth_mm"] for r in rows if r["truth_mm"] is not None]
        recons = [r["reconstructed_mm"] for r in rows if r["truth_mm"] is not None]
        categories_out[name] = {
            "errors": summarize_errors(errs),
            "scale": scale_fit(truths, recons),
        }

    truths = [m["truth_mm"] for m in measurements if m["truth_mm"] is not None]
    recons = [m["reconstructed_mm"] for m in measurements if m["truth_mm"] is not None]
    position_rows = [m for m in measurements if m["type"] == "position"]
    position = None
    if position_rows:
        position = {
            "n": len(position_rows),
            "position_error_mm": summarize_values(
                [m["abs_error_mm"] for m in position_rows]),
            "abs_x_error_mm": summarize_values(
                [abs(m["error_x_mm"]) for m in position_rows]),
            "abs_y_error_mm": summarize_values(
                [abs(m["error_y_mm"]) for m in position_rows]),
            "abs_z_error_mm": summarize_values(
                [abs(m["error_z_mm"]) for m in position_rows]),
        }
    return {
        "by_test": by_test_out,
        "categories": categories_out,
        "position": position,
        "overall_scale": scale_fit(truths, recons),
    }


def build_diagnostics(cal: Calibration, result: dict, plan: dict,
                      up_info: dict) -> tuple[list[dict], list[str]]:
    diagnostics: list[dict] = []
    warnings: list[str] = []

    def add(level, code, message):
        diagnostics.append({"level": level, "code": code, "message": message})
        if level == "warn":
            warnings.append(message)

    # Calibration-file checks.
    for c in cal.checks:
        if c["level"] in ("warn", "error"):
            add(c["level"] if c["level"] == "error" else "warn",
                c["code"], c["message"])

    pairs = result["pairs"]
    if not pairs:
        add("warn", "no_pairs", "No image pairs with correspondences were processed.")
        return diagnostics, warnings

    all_points = [p for pair in pairs for p in pair["points"].values()]
    valid = [p for p in all_points if p["valid"]]
    if not valid:
        add("error", "no_valid_points", "No triangulated point passed validity checks.")
        return diagnostics, warnings

    reproj_values = [max(p["reproj_error_cam0_px"],
                         p["reproj_error_cam1_px"])
                     for p in all_points
                     if np.isfinite(p["reproj_error_cam0_px"])
                     and np.isfinite(p["reproj_error_cam1_px"])]
    max_reproj = max(reproj_values, default=float("nan"))
    max_epi = max((p["epipolar_sampson_px"] for p in all_points
                   if np.isfinite(p["epipolar_sampson_px"])), default=float("nan"))
    min_par = min((p["parallax_deg"] for p in valid), default=float("nan"))

    add("info", "triangulation_quality",
        f"max reprojection error {max_reproj:.3f} px; max Sampson epipolar "
        f"error {max_epi:.3f} px; minimum parallax {min_par:.3f} deg")
    if max_reproj > HIGH_REPROJ_PX:
        add("warn", "high_reprojection",
            f"At least one correspondence reprojects with {max_reproj:.2f} px "
            "error; re-check the clicked point pair or camera stability.")
    if np.isfinite(max_epi) and max_epi > HIGH_EPIPOLAR_PX:
        add("warn", "high_epipolar",
            f"At least one correspondence has {max_epi:.2f} px epipolar error; "
            "the clicked points may not be the same physical point.")
    if min_par < MIN_PARALLAX_DEG:
        add("warn", "low_parallax",
            f"Minimum ray parallax is {min_par:.2f} deg.  Depth and vertical "
            "measurements are poorly conditioned; horizontal image-plane "
            "measurements are less affected.")

    # Depth sensitivity is a geometric fact, not a calibration error.
    sens = [p["depth_sensitivity_mm_per_px"] for p in valid
            if np.isfinite(p["depth_sensitivity_mm_per_px"])]
    median_sens = float(np.median(sens)) if sens else float("nan")
    if sens:
        add("info", "depth_sensitivity",
            f"Approximate depth sensitivity: {median_sens:.2f} mm per "
            "pixel of disparity at these points (not an observed error).")

    # Is each test large enough to be meaningful given click/depth sensitivity?
    if result["measurements"]:
        zvals = [p["depth_z_cam0_mm"] for p in valid
                 if np.isfinite(p["depth_z_cam0_mm"])]
        f_mean = 0.5 * (cal.K0[0, 0] + cal.K0[1, 1])
        lateral_unc = (0.5 * float(np.median(zvals)) / f_mean
                       if zvals and f_mean > 0 else float("nan"))
        seen = {}
        for m in result["measurements"]:
            seen.setdefault(m["test"], m)
        for name, m in seen.items():
            truth = m["truth_mm"]
            if truth is None or abs(truth) < 1e-12:
                continue
            vertical_test = (m["type"] == "height" or
                             m["kind"] == "vertical")
            unc = (0.5 * median_sens if vertical_test and np.isfinite(median_sens)
                   else lateral_unc)
            if not np.isfinite(unc):
                continue
            if abs(truth) < 2.0 * unc:
                add("warn", "test_geometry_weak",
                    f"Test {name!r} has a {abs(truth):.2f} mm reference span, "
                    f"below about twice the approximate 1-sigma click/depth "
                    f"sensitivity ({unc:.2f} mm).  It cannot establish "
                    "millimetre-scale accuracy reliably; use a larger span or "
                    "repeat many captures.")
        add("info", "test_geometry",
            f"Approximate 1-sigma lateral click sensitivity: {lateral_unc:.2f} "
            f"mm; conservative vertical/depth sensitivity: "
            f"{0.5 * median_sens:.2f} mm (at these distances).")

    # Error patterns.
    stats = result["statistics"]
    for test_name, data in stats["by_test"].items():
        rv = data.get("reconstructed", {})
        if rv.get("n", 0) < 2:
            continue
        mean_value = abs(float(rv.get("mean", 0.0)))
        allowed = max(1.0, 0.02 * mean_value)
        if float(rv.get("range", 0.0)) > allowed:
            add("warn", "repeatability",
                f"Repeated captures of test {test_name!r} vary over "
                f"{rv['range']:.2f} mm (std {rv['std']:.2f} mm).  If the target "
                "and cameras were stationary, this indicates click/geometry "
                "instability, target movement, or camera movement; do not "
                "average it away without identifying the cause.")
    length_ratios = [m["reconstructed_mm"] / m["truth_mm"]
                     for m in result["measurements"]
                     if m["truth_mm"] not in (None, 0)
                     and m["type"] == "length"]
    if len(length_ratios) >= 3:
        ratios = np.asarray(length_ratios)
        mean_ratio = float(ratios.mean())
        spread = float(np.std(ratios, ddof=1))
        if abs(mean_ratio - 1.0) * 100 > SCALE_WARN_PCT and spread < 0.02:
            add("warn", "possible_metric_scale",
                f"Known lengths reconstruct with a consistent mean ratio of "
                f"{mean_ratio:.4f} ({100 * (mean_ratio - 1):+.2f}%).  This pattern "
                "is consistent with a stereo metric-scale error, but do not "
                "rescale automatically; verify the physical baseline independently.")

    height_rows = [m for m in result["measurements"] if m["type"] == "height"
                   and m["signed_error_mm"] is not None]
    if height_rows:
        herrs = np.asarray([m["signed_error_mm"] for m in height_rows])
        if abs(float(herrs.mean())) > 2.0 and (len(herrs) < 3
                                               or float(np.std(herrs, ddof=1)) < 2.0):
            add("warn", "height_offset",
                f"Height errors have a near-constant offset of "
                f"{herrs.mean():+.2f} mm.  Check the board-surface/tank-floor "
                "reference and board_thickness_mm before blaming triangulation.")
        truths = np.asarray([m["truth_mm"] for m in height_rows])
        if (len(height_rows) >= 3 and np.std(truths) > 1e-9
                and float(np.max(np.abs(herrs))) >= 0.5
                and float(np.ptp(truths)) >= 5.0):
            corr = float(np.corrcoef(truths, herrs)[0, 1])
            if abs(corr) > 0.8:
                add("warn", "height_trend",
                    f"Height error correlates with height (r={corr:.2f}).  "
                    "Check the inferred up direction, plane tilt, or vertical scale.")

    # World/stereo consistency and camera movement.
    world_b = None
    for c in cal.checks:
        if c["code"] == "baseline_compare":
            add("info", "baseline_compare", c["message"])
    c0 = -cal.R_board_to_cam0.T @ cal.t_board_to_cam0
    c1 = -cal.R_board_to_cam1.T @ cal.t_board_to_cam1
    world_b = float(np.linalg.norm(c0 - c1))
    if abs(world_b - cal.baseline_mm) > max(2.0, 0.05 * cal.baseline_mm):
        add("warn", "baseline_mismatch_diagnosis",
            "The independently fitted world baseline and stereo |T| disagree.  "
            "Possible causes: stereo metric scale, board/world alignment error, "
            "or camera movement after calibration.  The stereo R/T remains the "
            "triangulation input; no correction is applied.")
    physical_b = plan.get("physical_baseline_mm")
    if physical_b is not None:
        try:
            physical_b = float(physical_b)
            if physical_b <= 0:
                raise ValueError("baseline must be positive")
            add("info", "physical_baseline",
                f"physical baseline {physical_b:.3f} mm; stereo |T| "
                f"{cal.baseline_mm:.3f} mm ({100*(cal.baseline_mm/physical_b-1):+.2f}%); "
                f"world-derived {world_b:.3f} mm "
                f"({100*(world_b/physical_b-1):+.2f}%)")
            if abs(world_b - physical_b) < max(1.0, 0.03 * physical_b) and \
                    abs(cal.baseline_mm - physical_b) > max(1.0, 0.03 * physical_b):
                add("warn", "metric_scale_supported",
                    "The independent world baseline agrees with the physical "
                    "baseline while stereo |T| does not.  This supports a stereo "
                    "metric-scale problem; it does not by itself prove that "
                    "rescaling is the right correction.  Do not rescale without "
                    "further validation.")
        except (TypeError, ValueError):
            add("warn", "physical_baseline_invalid",
                f"physical_baseline_mm is not numeric: {physical_b!r}")
    if cal.tank_transform_is_identity or not plan.get("tank_transform_verified", False):
        add("warn", "tank_not_verified",
            "Tank coordinates are not claimed: T_world_board is the identity "
            "default and/or the plan does not mark it verified.  Board-frame "
            "coordinates are reported instead.")
    if plan.get("tank_transform_verified", False) and cal.tank_transform_is_identity:
        add("warn", "tank_identity_claimed",
            "The plan marks the tank transform verified, but T_world_board is "
            "still the identity.  Confirm that the board origin and axes really "
            "coincide with the tank origin and axes.")

    floor_heights = [m for m in result["measurements"]
                     if m["type"] == "height" and m["p2"] == "floor"]
    if floor_heights and plan.get("board_thickness_mm") is None:
        add("warn", "board_thickness_missing",
            "Height tests use the board surface as the floor reference, but "
            "board_thickness_mm is null.  Absolute tank-floor heights therefore "
            "include an unknown board/paper thickness offset.")

    # Floor-plane consistency and fit quality, when the user supplied points.
    floor_names = [str(n) for n in plan.get("floor_reference_points", [])]
    fitted_plane = "least-squares" in str(result["floor_plane"].get("source", ""))
    if floor_names:
        present = set()
        for pair in result["pairs"]:
            for name in floor_names:
                p = pair["points"].get(name)
                if p is not None and p.get("valid"):
                    present.add(name)
        missing = [n for n in floor_names if n not in present]
        if missing:
            add("warn", "floor_reference_missing",
                f"floor_reference_points {missing} are missing or invalid; the "
                "floor-plane fit uses only the available points.")
    if fitted_plane:
        angle = result["floor_plane"].get("angle_to_up_deg")
        ratio = result["floor_plane"].get("planarity_ratio")
        if angle is not None and angle > 2.0:
            add("warn", "floor_plane_tilt",
                f"The fitted floor plane is tilted {angle:.2f} deg from the "
                "inferred up direction.  Check the clicked floor points, the "
                "board/floor relationship, or the inferred up direction.")
        if ratio is None:
            add("warn", "floor_plane_degenerate",
                "The supplied floor reference points are nearly collinear; the "
                "least-squares plane normal is not trustworthy.")
        elif ratio > 0.10:
            add("warn", "floor_plane_noisy",
                f"Floor-plane residual ratio is {ratio:.3f}; the reference "
                "points are not a good plane (check clicks or the assumption).")
        add("info", "floor_plane_fit",
            f"Fitted floor plane: angle to up {angle:.2f} deg; offset along up "
            f"{result['floor_plane']['offset_along_up_mm']:+.2f} mm; "
            f"planarity ratio {ratio if ratio is not None else 'undefined'}.")
        thickness = plan.get("board_thickness_mm")
        if thickness is not None:
            measured_offset = float(result["floor_plane"]["offset_along_up_mm"])
            expected_offset = -float(thickness)
            difference = measured_offset - expected_offset
            add("info", "floor_plane_thickness",
                f"Fitted floor offset along up {measured_offset:+.2f} mm; "
                f"board_thickness_mm {float(thickness):.2f} mm predicts "
                f"{expected_offset:+.2f} mm (difference {difference:+.2f} mm).")
            if abs(difference) > max(1.0, 0.05 * float(thickness)):
                add("warn", "floor_plane_thickness_mismatch",
                    "The fitted floor points and board_thickness_mm disagree.  "
                    "Check that the reference points are really on the tank "
                    "floor, that the board/paper thickness was measured to the "
                    "printed board plane, and that the inferred up direction is "
                    "correct.")
    elif floor_names:
        add("warn", "floor_plane_not_fitted",
            "Fewer than three valid floor_reference_points were available; the "
            "board plane is used as the floor reference instead.")

    # Distortion/refraction note is deliberately not turned into a numeric flag.
    max_d = max(float(np.max(np.abs(cal.D0))), float(np.max(np.abs(cal.D1))))
    add("info", "distortion",
        f"Distortion coefficients are applied explicitly (max |D|={max_d:.3f}).  "
        "Dry validation cannot quantify refraction; it must be assessed with "
        "water present and the same working distance.")

    add("info", "no_automatic_correction",
        "No baseline, scale, plane or point was modified to improve agreement.  "
        "A PASS/FAIL judgement requires a user-supplied --tolerance-mm.")
    return diagnostics, warnings


def run_measurement(session: Path, plan: dict, cal: Calibration,
                    tolerance_mm: float | None = None,
                    pair_filter: list[str] | None = None) -> dict:
    validate_plan(plan)
    up_info = board_up_direction(cal, plan.get("up_direction", "auto"))
    up = np.asarray(up_info["up_board"], np.float64)

    pairs = list_pair_dirs(session)
    if pair_filter:
        wanted = set(pair_filter)
        pairs = [p for p in pairs if p.name in wanted or str(p) in wanted]
    if not pairs:
        raise RuntimeError(f"No image pairs found under {session / 'pairs'}")

    pair_results = []
    for pair_dir in pairs:
        if not correspondence_path(pair_dir).exists():
            print(f"  skip {pair_dir.name}: no correspondences.json")
            continue
        if not load_correspondences(pair_dir):
            print(f"  skip {pair_dir.name}: zero correspondences")
            continue
        print(f"  reconstructing {pair_dir.name} ...")
        pair_results.append(_reconstruct_pair(pair_dir, cal, up, plan))

    if not pair_results:
        raise RuntimeError("No pairs with correspondences to reconstruct")

    measurements = [m for p in pair_results for m in p["measurements"]]
    statistics = build_statistics(measurements)

    # Fitted floor-plane check, only when the user identifies reference points.
    floor_names = [str(n) for n in plan.get("floor_reference_points", [])]
    floor_points = []
    for pair in pair_results:
        for name in floor_names:
            p = pair["points"].get(name)
            if p is not None and p["valid"]:
                floor_points.append(np.asarray(p["X_board_mm"], np.float64))
    plane = fit_plane(np.asarray(floor_points)) if len(floor_points) >= 3 else None
    if plane is not None:
        n = np.asarray(plane["normal"], np.float64)
        if float(n @ up) < 0:
            n = -n
            plane["normal"] = n.tolist()
        angle = float(np.degrees(np.arccos(np.clip(float(n @ up), -1.0, 1.0))))
        plane["angle_to_up_deg"] = angle
        plane["offset_along_up_mm"] = float(
            np.asarray(plane["centroid_mm"]) @ up)
        plane["source"] = "least-squares fit to floor_reference_points"
        sv = np.asarray(plane.get("singular_values", []), np.float64)
        if sv.size >= 2:
            plane["planarity_ratio"] = (float(sv[2] / sv[1])
                                        if sv[1] > 1e-12 else None)
    else:
        plane = {
            "source": "board plane z=0 (no >=3 floor_reference_points supplied)",
            "normal": up.tolist(),
            "centroid_mm": [0.0, 0.0, 0.0],
            "offset_along_up_mm": 0.0,
            "angle_to_up_deg": 0.0,
        }

    result = {
        "generated_iso": utc_now_iso(),
        "session": str(session.resolve()),
        "calibration": {
            "files": {name: {"path": str(cal.paths[name].resolve()),
                             "sha256": cal.hashes[name]}
                      for name in CALIB_NAMES},
            "checks": cal.checks,
            "image_size": list(cal.size0),
            "stereo_baseline_mm": cal.baseline_mm,
            "transforms_used": {
                "K0": cal.K0.tolist(),
                "D0": cal.D0.tolist(),
                "K1": cal.K1.tolist(),
                "D1": cal.D1.tolist(),
                "R_stereo_cam0_to_cam1": cal.R.tolist(),
                "T_stereo_cam0_to_cam1_mm": cal.T.tolist(),
                "R_board_to_cam0": cal.R_board_to_cam0.tolist(),
                "t_board_to_cam0_mm": cal.t_board_to_cam0.tolist(),
                "R_board_to_cam1": cal.R_board_to_cam1.tolist(),
                "t_board_to_cam1_mm": cal.t_board_to_cam1.tolist(),
                "T_world_board": cal.T_board_to_tank.tolist(),
            },
        },
        "coordinate_frames": {
            "camera0": "OpenCV camera 0: X right, Y down, Z forward; millimetres",
            "board": "big ChArUco board: +X/+Y in plane, +Z out of the "
                     "object-point plane; physical up is given by "
                     "coordinate_frames.up",
            "tank": "only reported as verified if T_world_board is a measured, "
                    "verified transform and the plan marks it verified",
            "stereo": "X_cam1 = R @ X_cam0 + T",
            "board_to_cam": "X_cam = R_board_to_cam @ X_board + t_board_to_cam",
            "board_to_tank": "X_tank = T_world_board[:3,:3] @ X_board + "
                             "T_world_board[:3,3]; T_world_board is "
                             "authoritative",
            "error_sign": "signed_error_mm = reconstructed_mm - truth_mm",
            "acquisition": "raw full-resolution frames saved without software "
                           "resize, crop, rotation or mirroring; previews are "
                           "downscaled only for display",
            "up": up_info,
        },
        "tank_transform": {
            "matrix": cal.T_board_to_tank.tolist(),
            "is_identity": cal.tank_transform_is_identity,
            "verified": bool(plan.get("tank_transform_verified", False)) and
                        not cal.tank_transform_is_identity,
        },
        "plan": plan,
        "pairs": pair_results,
        "measurements": measurements,
        "statistics": statistics,
        "floor_plane": plane,
        "tolerance_mm": tolerance_mm,
    }
    diagnostics, warnings = build_diagnostics(cal, result, plan, up_info)
    result["diagnostics"] = diagnostics
    result["warnings"] = warnings

    if tolerance_mm is not None:
        failed = [m for m in measurements if m["abs_error_mm"] is not None
                  and m["abs_error_mm"] > tolerance_mm]
        result["tolerance_result"] = {
            "tolerance_mm": tolerance_mm,
            "passed": len(failed) == 0,
            "n_failed": len(failed),
            "failed_tests": [f"{m['pair']}:{m['test']}" for m in failed],
        }
    return result


# ============================================================
# OUTPUT / VISUALIZATION
# ============================================================

def _draw_text_block(img: np.ndarray, lines: list[str], origin=(20, 40),
                     scale=1.0, color=(255, 255, 255), bg=True) -> None:
    if not lines:
        return
    x, y = origin
    line_h = int(round(32 * scale))
    if bg:
        w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
                for t in lines) + 24
        cv2.rectangle(img, (x - 12, y - line_h + 8),
                      (x + w, y + line_h * (len(lines) - 1) + 10),
                      (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def annotate_pair(pair: dict, cal: Calibration, plan: dict, result: dict,
                  out_dir: Path) -> list[Path]:
    colors = [(80, 220, 80), (80, 180, 255), (255, 180, 80), (220, 120, 255),
              (255, 255, 120), (120, 255, 255), (180, 120, 255), (150, 255, 180)]
    name_colors = {}
    for i, name in enumerate(pair["points"]):
        name_colors[name] = colors[i % len(colors)]

    tests = pair["measurements"]
    paths = []
    for cam_key, cam_idx in (("cam0", 0), ("cam1", 1)):
        img = pair["images"][cam_key].copy()
        h, w = img.shape[:2]
        sc = max(0.6, min(1.2, w / 1920.0))
        # Clicked and reprojected points.
        for name, p in pair["points"].items():
            xy = p[f"{cam_key}_raw_px"]
            if not (np.all(np.isfinite(xy))):
                continue
            color = name_colors[name] if p["valid"] else (0, 0, 255)
            pt = (int(round(xy[0])), int(round(xy[1])))
            cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS, 18, 2)
            cv2.circle(img, pt, 8, color, 2)
            label = f"{name} ({xy[0]:.1f},{xy[1]:.1f})"
            if not p["valid"]:
                label += " INVALID"
            cv2.putText(img, label,
                        (pt[0] + 12, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * sc, color, 2, cv2.LINE_AA)
            rp = p[f"reproj_{cam_key}_px"]
            if np.all(np.isfinite(rp)):
                rpt = (int(round(rp[0])), int(round(rp[1])))
                cv2.drawMarker(img, rpt, (255, 255, 0), cv2.MARKER_TILTED_CROSS,
                               12, 1)
                cv2.line(img, pt, rpt, (0, 255, 255), 1, cv2.LINE_AA)

        # Length-test lines.
        for m in tests:
            if m["type"] != "length":
                continue
            p1 = pair["points"].get(m["p1"])
            p2 = pair["points"].get(m["p2"])
            if not p1 or not p2:
                continue
            a = p1[f"{cam_key}_raw_px"]
            b = p2[f"{cam_key}_raw_px"]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            pa = (int(round(a[0])), int(round(a[1])))
            pb = (int(round(b[0])), int(round(b[1])))
            cv2.line(img, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            label = f"{m['test']}: {m['reconstructed_mm']:.2f} mm"
            if m["signed_error_mm"] is not None:
                label += f" ({m['signed_error_mm']:+.2f})"
            cv2.putText(img, label, (mid[0] + 8, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (0, 255, 0), 2,
                        cv2.LINE_AA)

        lines = [
            f"Pair {pair['pair']}  {cam_key}",
            f"clicked=colored  reprojected=cyan  invalid=red  combined RMS="
            f"{fmt(pair['combined_reproj_rms_px'], 2)} px",
        ]
        for m in tests:
            if m["type"] == "position":
                lines.append(f"{m['test']}: |position error| "
                             f"{m['abs_error_mm']:.2f} mm")
            elif m["signed_error_mm"] is not None:
                pct_text = ("n/a" if m["pct_error"] is None
                            else f"{m['pct_error']:+.2f}%")
                lines.append(f"{m['test']}: {m['reconstructed_mm']:.2f} mm  "
                             f"err {m['signed_error_mm']:+.2f} mm  "
                             f"({pct_text})")
            else:
                lines.append(f"{m['test']}: {m['reconstructed_mm']:.2f} mm")
        _draw_text_block(img, lines, origin=(20, 40), scale=0.65 * sc)
        path = out_dir / f"{pair['pair']}_{cam_key}_annotated.png"
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"could not write {path}")
        paths.append(path)
    return paths


def _project_iso(x: float, y: float, z: float) -> tuple[float, float]:
    # Isometric projection with the same scale for all three axes.
    c = math.cos(math.radians(30.0))
    s = math.sin(math.radians(30.0))
    return (x - y) * c, (x + y) * s - z


def render_3d_plot(result: dict, cal: Calibration, out_path: Path) -> Path:
    width, height = 1400, 1000
    margin = 90
    canvas = np.full((height, width, 3), 24, np.uint8)
    up = np.asarray(result["coordinate_frames"]["up"]["up_board"], np.float64)

    # Plot in board X/Y and physical up coordinate.
    points = []
    colors = [(80, 220, 80), (80, 180, 255), (255, 180, 80), (220, 120, 255),
              (255, 255, 120), (120, 255, 255), (180, 120, 255)]
    for pair_idx, pair in enumerate(result["pairs"]):
        for name, p in pair["points"].items():
            if not p["valid"]:
                continue
            Xb = np.asarray(p["X_board_mm"], np.float64)
            points.append({
                "x": float(Xb[0]), "y": float(Xb[1]), "h": float(Xb @ up),
                "name": name, "pair": pair["pair"],
                "color": colors[pair_idx % len(colors)],
            })
    if not points:
        canvas[:] = 20
        _draw_text_block(canvas, ["No valid reconstructed points to plot."])
        if not cv2.imwrite(str(out_path), canvas):
            raise RuntimeError(f"could not write {out_path}")
        return out_path

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    hs = np.array([p["h"] for p in points])
    cx, cy, ch = float(xs.mean()), float(ys.mean()), float(hs.mean())
    span_x = max(float(xs.max() - xs.min()), 100.0)
    span_y = max(float(ys.max() - ys.min()), 100.0)
    span_h = max(float(hs.max() - hs.min()), 100.0)
    # Use a local plotting box around the points, not the whole 1.5 m board.
    half_x = max(span_x * 0.7, 100.0)
    half_y = max(span_y * 0.7, 100.0)

    def iso_of(p):
        return _project_iso(p["x"] - cx, p["y"] - cy, p["h"] - ch)

    projected = [iso_of(p) for p in points]
    px = np.array([q[0] for q in projected])
    py = np.array([q[1] for q in projected])
    # Add the plane/axis corners to the bounds.
    corners = []
    for x in (cx - half_x, cx + half_x):
        for y in (cy - half_y, cy + half_y):
            for h in (0.0 - ch,):
                corners.append(_project_iso(x - cx, y - cy, h))
    cx_p = np.array([q[0] for q in corners])
    cy_p = np.array([q[1] for q in corners])
    px = np.r_[px, cx_p]
    py = np.r_[py, cy_p]
    sx_min, sx_max = float(px.min()), float(px.max())
    sy_min, sy_max = float(py.min()), float(py.max())
    scale = min((width - 2 * margin) / max(sx_max - sx_min, 1e-6),
                (height - 2 * margin - 80) / max(sy_max - sy_min, 1e-6))
    ox = width / 2 - scale * (sx_min + sx_max) / 2
    oy = height / 2 + 30 - scale * (sy_min + sy_max) / 2

    def to_px(p):
        sx, sy = iso_of(p)
        return int(round(ox + scale * sx)), int(round(oy + scale * sy))

    # Board-plane grid (physical height 0; local window around points).
    step = 50.0
    x0 = math.floor((cx - half_x) / step) * step
    x1 = math.ceil((cx + half_x) / step) * step
    y0 = math.floor((cy - half_y) / step) * step
    y1 = math.ceil((cy + half_y) / step) * step
    for x in np.arange(x0, x1 + 1e-9, step):
        a = (int(round(ox + scale * _project_iso(x - cx, y0 - cy, -ch)[0])),
             int(round(oy + scale * _project_iso(x - cx, y0 - cy, -ch)[1])))
        b = (int(round(ox + scale * _project_iso(x - cx, y1 - cy, -ch)[0])),
             int(round(oy + scale * _project_iso(x - cx, y1 - cy, -ch)[1])))
        cv2.line(canvas, a, b, (55, 55, 55), 1, cv2.LINE_AA)
    for y in np.arange(y0, y1 + 1e-9, step):
        a = (int(round(ox + scale * _project_iso(x0 - cx, y - cy, -ch)[0])),
             int(round(oy + scale * _project_iso(x0 - cx, y - cy, -ch)[1])))
        b = (int(round(ox + scale * _project_iso(x1 - cx, y - cy, -ch)[0])),
             int(round(oy + scale * _project_iso(x1 - cx, y - cy, -ch)[1])))
        cv2.line(canvas, a, b, (55, 55, 55), 1, cv2.LINE_AA)

    # A fitted floor plane, when enough floor reference points were supplied.
    fp = result.get("floor_plane", {})
    if "normal" in fp and "least-squares" in str(fp.get("source", "")):
        n = np.asarray(fp["normal"], np.float64)
        n /= np.linalg.norm(n)
        e1 = np.cross(up, n)
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(np.array([1.0, 0.0, 0.0]), n)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        c = np.asarray(fp["centroid_mm"], np.float64)
        corners = [c - half_x * e1 - half_y * e2,
                   c + half_x * e1 - half_y * e2,
                   c + half_x * e1 + half_y * e2,
                   c - half_x * e1 + half_y * e2]
        pix = []
        for P in corners:
            sx, sy = _project_iso(P[0] - cx, P[1] - cy, float(P @ up) - ch)
            pix.append((int(round(ox + scale * sx)),
                        int(round(oy + scale * sy))))
        for i in range(4):
            cv2.line(canvas, pix[i], pix[(i + 1) % 4], (220, 120, 255), 2,
                     cv2.LINE_AA)
        cv2.putText(canvas, "fitted floor plane", (pix[0][0] + 6, pix[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 120, 255), 2,
                    cv2.LINE_AA)

    # Axes.
    axis_len = max(80.0, min(half_x, half_y) * 0.35)
    origin = {"x": cx, "y": cy, "h": 0.0}
    o = to_px(origin)
    for label, target in (
        ("X", {"x": cx + axis_len, "y": cy, "h": 0.0}),
        ("Y", {"x": cx, "y": cy + axis_len, "h": 0.0}),
        ("up", {"x": cx, "y": cy, "h": axis_len}),
    ):
        q = to_px(target)
        cv2.arrowedLine(canvas, o, q, (220, 220, 220), 2, cv2.LINE_AA, tipLength=0.04)
        cv2.putText(canvas, label, (q[0] + 6, q[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # Points.
    for p, q in zip(points, projected):
        pt = to_px(p)
        cv2.circle(canvas, pt, 6, p["color"], -1, cv2.LINE_AA)
        cv2.putText(canvas, p["name"], (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, p["color"], 2, cv2.LINE_AA)

    # Scale bar: same mm-per-pixel on every axis by construction.
    bar_mm = 50.0
    bar_px = int(round(scale * bar_mm))
    bx, by = width - margin - bar_px, height - 45
    cv2.line(canvas, (bx, by), (bx + bar_px, by), (230, 230, 230), 3)
    cv2.putText(canvas, f"{bar_mm:.0f} mm", (bx, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    title_lines = [
        "Reconstructed points in big-board coordinates (mm)",
        "Isometric view; identical physical scale on X, Y and up axes",
        "Grey grid = board plane (physical height 0)",
        f"up = board {result['coordinate_frames']['up']['up_board']}",
        f"points: {len(points)} from {len(result['pairs'])} pair(s)",
    ]
    fp2 = result.get("floor_plane", {})
    if "angle_to_up_deg" in fp2 and "least-squares" in str(fp2.get("source", "")):
        title_lines.append(
            f"magenta outline = fitted floor plane; angle to up "
            f"{fp2['angle_to_up_deg']:.2f} deg; offset "
            f"{fp2['offset_along_up_mm']:+.2f} mm")
    _draw_text_block(canvas, title_lines, origin=(25, 35), scale=0.75)
    if len(result["pairs"]) > 1:
        lx, ly = 30, 35 + 32 * len(title_lines) + 20
        for i, pair in enumerate(result["pairs"]):
            color = colors[i % len(colors)]
            cv2.circle(canvas, (lx + 8, ly + i * 24), 6, color, -1,
                       cv2.LINE_AA)
            cv2.putText(canvas, pair["pair"], (lx + 22, ly + i * 24 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"could not write {out_path}")
    return out_path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(session: Path, result: dict, cal: Calibration,
                  plan: dict) -> dict:
    out_dir = session / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotated images.
    annotated = []
    for pair in result["pairs"]:
        annotated.extend(annotate_pair(pair, cal, plan, result, out_dir))
    result["annotated_images"] = [str(p) for p in annotated]

    plot_path = render_3d_plot(result, cal, out_dir / "reconstruction_3d.png")
    result["three_d_plot"] = str(plot_path)

    # Measurement CSV.
    mfields = ["pair", "test", "type", "kind", "p1", "p2", "truth_mm",
               "reconstructed_mm", "signed_error_mm", "abs_error_mm",
               "pct_error", "signed_up_component_mm", "height_basis",
               "truth_board_mm", "reconstructed_board_mm", "position_error_mm",
               "error_x_mm", "error_y_mm", "error_z_mm"]
    write_csv(out_dir / "measurements.csv", result["measurements"], mfields)

    # Per-point CSV.
    prow = []
    for pair in result["pairs"]:
        for name, p in pair["points"].items():
            xt = p.get("X_tank_mm")
            if xt is None:
                xt = [None, None, None]
            prow.append({
                "pair": pair["pair"], "point": name, "valid": p["valid"],
                "cam0_x_px": p["cam0_raw_px"][0],
                "cam0_y_px": p["cam0_raw_px"][1],
                "cam1_x_px": p["cam1_raw_px"][0],
                "cam1_y_px": p["cam1_raw_px"][1],
                "pixel_x_px": p["pixel_xyz"][0],
                "pixel_y_px": p["pixel_xyz"][1],
                "pixel_z_px": p["pixel_xyz"][2],
                "X_cam0_mm": p["X_cam0_mm"][0],
                "Y_cam0_mm": p["X_cam0_mm"][1],
                "Z_cam0_mm": p["X_cam0_mm"][2],
                "X_board_mm": p["X_board_mm"][0],
                "Y_board_mm": p["X_board_mm"][1],
                "Z_board_mm": p["X_board_mm"][2],
                "height_above_board_mm": p["height_above_board_mm"],
                "height_above_tank_floor_mm": p["height_above_tank_floor_mm"],
                "X_tank_mm": xt[0],
                "Y_tank_mm": xt[1],
                "Z_tank_mm": xt[2],
                "X_tank_verified": p["X_tank_verified"],
                "reproj_error_cam0_px": p["reproj_error_cam0_px"],
                "reproj_error_cam1_px": p["reproj_error_cam1_px"],
                "epipolar_sampson_px": p["epipolar_sampson_px"],
                "parallax_deg": p["parallax_deg"],
                "depth_sensitivity_mm_per_px": p["depth_sensitivity_mm_per_px"],
                "flags": "|".join(p["flags"]),
            })
    pfields = list(prow[0].keys()) if prow else ["pair", "point"]
    write_csv(out_dir / "points.csv", prow, pfields)

    # Human-readable summary.
    lines = []
    lines.append("DRY STEREO 3-D VALIDATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Generated: {result['generated_iso']}")
    lines.append(f"Session:   {result['session']}")
    lines.append("")
    lines.append("CALIBRATION FILES")
    for name in CALIB_NAMES:
        rec = result["calibration"]["files"][name]
        lines.append(f"  {name}: {rec['path']}")
        lines.append(f"    sha256 {rec['sha256']}")
    lines.append("")
    lines.append("COORDINATE / UP CONVENTION")
    cf = result["coordinate_frames"]
    lines.append(f"  camera0: {cf['camera0']}")
    lines.append(f"  board:   {cf['board']}")
    lines.append(f"  stereo:  {cf['stereo']}")
    lines.append(f"  board->cam: {cf['board_to_cam']}")
    lines.append(f"  board->tank: {cf['board_to_tank']}")
    lines.append(f"  acquisition: {cf['acquisition']}")
    up = cf["up"]
    lines.append(f"  up in board coordinates: {up['up_board']}")
    lines.append(f"  camera height above board: "
                 f"{up['camera_height_above_board_mm']:.2f} mm")
    lines.append(f"  inference: {up['method']}")
    lines.append(f"  {up['note']}")
    tt = result["tank_transform"]
    lines.append(f"  board->tank identity: {tt['is_identity']}; "
                 f"verified tank coordinates: {tt['verified']}")
    if not tt["verified"]:
        lines.append("  Tank coordinates are NOT reported as established. "
                     "Board-frame coordinates are used.")
    lines.append("")
    lines.append("CALIBRATION CHECKS / WARNINGS")
    for c in result["calibration"]["checks"]:
        if c["level"] in ("warn", "error", "info"):
            lines.append(f"  [{c['level'].upper()}] {c['code']}: {c['message']}")
    lines.append("")
    lines.append("MEASUREMENTS (per pair)")
    lines.append("  signed error = reconstructed - truth; positive means the "
                 "reconstruction is larger")
    if result["measurements"]:
        lines.append(f"  {'pair':<24} {'test':<26} {'truth':>9} {'recon':>9} "
                     f"{'error':>9} {'abs':>8} {'pct':>8}")
        for m in result["measurements"]:
            if m["type"] == "position":
                continue
            truth = "n/a" if m["truth_mm"] is None else f"{m['truth_mm']:.2f}"
            err = "n/a" if m["signed_error_mm"] is None else f"{m['signed_error_mm']:+.2f}"
            ab = "n/a" if m["abs_error_mm"] is None else f"{m['abs_error_mm']:.2f}"
            pct = "n/a" if m["pct_error"] is None else f"{m['pct_error']:+.2f}%"
            recon = "n/a" if m["reconstructed_mm"] is None else f"{m['reconstructed_mm']:.2f}"
            lines.append(f"  {m['pair']:<24} {m['test']:<26} {truth:>9} "
                         f"{recon:>9} {err:>9} {ab:>8} {pct:>8}")
    else:
        lines.append("  No tests enabled in measurement_plan.json.")
    if result["statistics"].get("position"):
        lines.append("")
        lines.append("KNOWN-POSITION TESTS (board frame)")
        for m in result["measurements"]:
            if m["type"] != "position":
                continue
            lines.append(f"  {m['pair']} / {m['test']}: point {m['p1']}")
            lines.append(f"    reconstructed {m['reconstructed_board_mm']} mm")
            lines.append(f"    truth         {m['truth_board_mm']} mm")
            lines.append(f"    error vector  ({m['error_x_mm']:+.3f}, "
                         f"{m['error_y_mm']:+.3f}, {m['error_z_mm']:+.3f}) mm; "
                         f"|error| {m['abs_error_mm']:.3f} mm")
        ps = result["statistics"]["position"]
        lines.append(f"    mean |position error|: "
                     f"{ps['position_error_mm']['mean']:.3f} mm; "
                     f"mean |X| {ps['abs_x_error_mm']['mean']:.3f}, "
                     f"|Y| {ps['abs_y_error_mm']['mean']:.3f}, "
                     f"|Z| {ps['abs_z_error_mm']['mean']:.3f} mm")
    lines.append("")
    lines.append("SUMMARY STATISTICS")
    for cat, data in result["statistics"]["categories"].items():
        e = data["errors"]
        s = data["scale"]
        label = "all length+height" if cat == "all" else cat
        lines.append(f"  {label}: n={e['n']}  mean_signed={e['mean_signed_mm']:+.3f} mm  "
                     f"mean_abs={e['mean_abs_mm']:.3f} mm  RMSE={e['rmse_mm']:.3f} mm  "
                     f"max_abs={e['max_abs_mm']:.3f} mm  "
                     f"std={e['std_signed_mm']:.3f} mm")
        if "mean_scale_bias_pct" in s:
            lines.append(f"    scale: mean ratio={s['mean_ratio']:.4f} "
                         f"({s['mean_scale_bias_pct']:+.2f}%), "
                         f"ratio std={s['ratio_std']:.4f}")
    spread_rows = [(name, data["reconstructed"])
                   for name, data in result["statistics"]["by_test"].items()
                   if data.get("reconstructed", {}).get("n", 0) > 1]
    if spread_rows:
        lines.append("")
        lines.append("REPEATED-CAPTURE SPREAD BY TEST (reconstructed value)")
        for name, rv in spread_rows:
            lines.append(f"  {name}: n={rv['n']} mean={rv['mean']:.3f} mm "
                         f"std={rv['std']:.3f} mm range={rv['range']:.3f} mm")
    lines.append("")
    lines.append("FLOOR PLANE CHECK")
    fp = result["floor_plane"]
    lines.append(f"  source: {fp.get('source')}")
    if "normal" in fp:
        lines.append(f"  normal (board): {fp['normal']}")
    if "angle_to_up_deg" in fp:
        lines.append(f"  angle to inferred up: {fp['angle_to_up_deg']:.3f} deg")
        lines.append(f"  offset along up: {fp['offset_along_up_mm']:+.3f} mm")
    lines.append("")
    lines.append("DIAGNOSTICS")
    for d in result["diagnostics"]:
        lines.append(f"  [{d['level'].upper()}] {d['code']}: {d['message']}")
    if "tolerance_result" in result:
        tr = result["tolerance_result"]
        lines.append("")
        lines.append(f"USER TOLERANCE: {tr['tolerance_mm']:.3f} mm -> "
                     f"{'PASS' if tr['passed'] else 'FAIL'} "
                     f"({tr['n_failed']} failed test(s))")
        for ft in tr["failed_tests"]:
            lines.append(f"  failed: {ft}")
    else:
        lines.append("")
        lines.append("No --tolerance-mm supplied: no PASS/FAIL judgement is made.")
    lines.append("")
    lines.append("OUTPUT FILES")
    for p in result["annotated_images"]:
        lines.append(f"  {p}")
    lines.append(f"  {result['three_d_plot']}")
    lines.append(f"  {out_dir / 'measurements.json'}")
    lines.append(f"  {out_dir / 'measurements.csv'}")
    lines.append(f"  {out_dir / 'points.csv'}")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")

    # JSON, with raw image arrays stripped.
    jresult = dict(result)
    jresult["pairs"] = []
    for pair in result["pairs"]:
        jp = {k: v for k, v in pair.items() if k != "images"}
        jresult["pairs"].append(jp)
    write_json(out_dir / "measurements.json", jresult)
    return result


def preliminary_distances(session: Path, cal: Calibration, plan: dict,
                          pair_filter: list[str] | None = None) -> dict:
    """Report all pairwise distances when no truth values are configured yet."""
    up_info = board_up_direction(cal, plan.get("up_direction", "auto"))
    up = np.asarray(up_info["up_board"], np.float64)
    pairs = list_pair_dirs(session)
    if pair_filter:
        wanted = set(pair_filter)
        pairs = [p for p in pairs if p.name in wanted or str(p) in wanted]
    rows = []
    pair_summaries = []
    for pair_dir in pairs:
        if not correspondence_path(pair_dir).exists():
            continue
        try:
            pair = _reconstruct_pair(pair_dir, cal, up, plan)
        except ValueError as exc:
            print(f"  skip {pair_dir.name}: {exc}")
            continue
        pair_summaries.append({
            "pair": pair["pair"],
            "pair_meta": pair["pair_meta"],
            "n_points": pair["n_points"],
            "n_valid": pair["n_valid"],
            "combined_reproj_rms_px": pair["combined_reproj_rms_px"],
            "max_reproj_px": pair["max_reproj_px"],
            "min_parallax_deg": pair["min_parallax_deg"],
            "points": pair["points"],
        })
        names = sorted(pair["points"])
        for a, b in itertools.combinations(names, 2):
            pa, pb = pair["points"][a], pair["points"][b]
            if not (pa["valid"] and pb["valid"]):
                continue
            Xa = np.asarray(pa["X_board_mm"], np.float64)
            Xb = np.asarray(pb["X_board_mm"], np.float64)
            d = Xb - Xa
            rows.append({
                "pair": pair["pair"], "p1": a, "p2": b,
                "euclidean_mm": float(np.linalg.norm(d)),
                "signed_up_mm": float(d @ up),
                "X_cam0_distance_mm": float(np.linalg.norm(
                    np.asarray(pb["X_cam0_mm"]) - np.asarray(pa["X_cam0_mm"]))),
                "height_p1_mm": pa["height_above_board_mm"],
                "height_p2_mm": pb["height_above_board_mm"],
            })
    out_dir = session / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "preliminary_distances.json",
               {"generated_iso": utc_now_iso(), "up": up_info,
                "pairs": pair_summaries, "distances": rows})
    write_csv(out_dir / "preliminary_distances.csv", rows,
              ["pair", "p1", "p2", "euclidean_mm", "signed_up_mm",
               "X_cam0_distance_mm", "height_p1_mm", "height_p2_mm"])
    print("\nPRELIMINARY PAIRWISE DISTANCES (no ground truth yet):")
    print(f"  {'pair':<24} {'p1':<5} {'p2':<5} {'3-D mm':>10} {'up mm':>10}")
    for r in rows:
        print(f"  {r['pair']:<24} {r['p1']:<5} {r['p2']:<5} "
              f"{r['euclidean_mm']:>10.2f} {r['signed_up_mm']:>10.2f}")
    print(f"\nSaved preliminary distances to {out_dir}")
    return {"rows": rows, "up": up_info, "pairs": pair_summaries}


# ============================================================
# CAPTURE INTERFACE
# ============================================================

def _save_pair(session: Path, frame0: np.ndarray, frame1: np.ndarray,
               meta_extra: dict, cam_times_ns: list[int] | None = None) -> Path:
    n = _next_pair_number(session)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_dir = session / "pairs" / f"pair_{n:04d}_{stamp}"
    pair_dir.mkdir(parents=True, exist_ok=False)
    ok0 = cv2.imwrite(str(pair_dir / "cam0.png"), frame0)
    ok1 = cv2.imwrite(str(pair_dir / "cam1.png"), frame1)
    if not (ok0 and ok1):
        raise RuntimeError(f"could not write capture images to {pair_dir}")
    times = cam_times_ns or [time.time_ns(), time.time_ns()]
    meta = {
        "pair_id": pair_dir.name,
        "created_iso": utc_now_iso(),
        "timestamp_unix": time.time(),
        "cam0": {"file": "cam0.png", "shape": list(frame0.shape),
                 "read_timestamp_ns": int(times[0]),
                 "read_timestamp_iso": datetime.fromtimestamp(
                     times[0] / 1e9).astimezone().isoformat(timespec="milliseconds")},
        "cam1": {"file": "cam1.png", "shape": list(frame1.shape),
                 "read_timestamp_ns": int(times[1]),
                 "read_timestamp_iso": datetime.fromtimestamp(
                     times[1] / 1e9).astimezone().isoformat(timespec="milliseconds")},
        "inter_camera_read_gap_ms": abs(times[1] - times[0]) / 1e6,
        "warning": "C920 cameras are not hardware-synchronized; the reference "
                   "object must be completely stationary during capture.",
    }
    meta.update(meta_extra)
    write_json(pair_dir / "pair.json", meta)
    return pair_dir


class CaptureState:
    """Shared status for the browser-based stereo capture page."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.message = "Starting cameras..."
        self.saved = 0
        self.last_pair: str | None = None
        self.error: str | None = None
        self.sizes: dict[str, tuple[int, int] | None] = {"cam0": None, "cam1": None}
        self.gap_ms: float | None = None
        self.settings: dict[str, Any] = {}
        self.ready = False
        self.viewers = 0
        self.capture_event = threading.Event()
        self.stop_event = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "message": self.message,
                "saved": self.saved,
                "last_pair": self.last_pair,
                "error": self.error,
                "sizes": {key: (list(value) if value is not None else None)
                          for key, value in self.sizes.items()},
                "gap_ms": self.gap_ms,
                "settings": dict(self.settings),
                "ready": self.ready,
                "viewers": self.viewers,
            }


def _capture_actual_settings(cap0, cap1, size0, size1) -> dict:
    return {
        "actual_size_cam0": list(size0),
        "actual_size_cam1": list(size1),
        "actual_fps_cam0": float(cap0.get(cv2.CAP_PROP_FPS)),
        "actual_fps_cam1": float(cap1.get(cv2.CAP_PROP_FPS)),
        "actual_focus_cam0": float(cap0.get(cv2.CAP_PROP_FOCUS)),
        "actual_focus_cam1": float(cap1.get(cv2.CAP_PROP_FOCUS)),
        "actual_exposure_cam0": float(cap0.get(cv2.CAP_PROP_EXPOSURE)),
        "actual_exposure_cam1": float(cap1.get(cv2.CAP_PROP_EXPOSURE)),
    }


class StereoCaptureWorker(threading.Thread):
    """Own both cameras and save fresh full-resolution pairs on request."""

    def __init__(self, args: argparse.Namespace, state: CaptureState,
                 views: dict, session: Path, meta_base: dict,
                 board_px: dict[str, dict[str, Any]] | None = None):
        super().__init__(daemon=True, name="stereo-capture")
        self.args = args
        self.state = state
        self.views = views
        self.session = session
        self.meta_base = meta_base
        self.board_px = board_px or {}

    def run(self) -> None:
        from web_tags import encode_jpeg

        cap0 = cap1 = None
        try:
            cap0, size0 = open_camera(
                self.args.cam0, self.args.width, self.args.height, self.args.fps,
                focus_val=self.args.focus, exposure_val=self.args.exposure)
            cap1, size1 = open_camera(
                self.args.cam1, self.args.width, self.args.height, self.args.fps,
                focus_val=self.args.focus, exposure_val=self.args.exposure)
            cal_sizes = self.meta_base.get("calibration_image_sizes", {})
            expected0 = tuple(cal_sizes.get("cam0", ()))
            expected1 = tuple(cal_sizes.get("cam1", ()))
            mismatch = (tuple(size0) != expected0 or tuple(size1) != expected1)
            if mismatch:
                self.meta_base["size_mismatch_warning"] = (
                    f"live size cam0/cam1 {size0}/{size1} differs from "
                    f"calibration {expected0}/{expected1}; measure will reject "
                    "these pairs.")
                self.board_px = {}
            with self.state.lock:
                self.state.sizes = {"cam0": tuple(size0), "cam1": tuple(size1)}
                self.state.ready = True
                self.state.message = (
                    "Cameras ready. Keep the target completely still and press "
                    "Capture Pair.")
                if mismatch:
                    self.state.message += (
                        f"  WARNING: live size {size0}/{size1} differs from "
                        f"calibration {expected0}/{expected1}.")

            last_preview = 0.0
            while not self.state.stop_event.is_set():
                ok0, frame0 = cap0.read()
                t0 = time.time_ns()
                ok1, frame1 = cap1.read()
                t1 = time.time_ns()
                if not (ok0 and ok1):
                    with self.state.lock:
                        self.state.message = (
                            "Camera read failed; retrying. Check USB connections.")
                    time.sleep(0.05)
                    continue

                with self.state.lock:
                    self.state.gap_ms = abs(t1 - t0) / 1e6
                    self.state.settings = _capture_actual_settings(
                        cap0, cap1, size0, size1)

                # Encode previews only while a browser is connected, and cap
                # the preview rate so the capture loop is not CPU-bound.
                with self.state.lock:
                    need_preview = self.state.viewers > 0
                now = time.monotonic()
                if need_preview and now - last_preview >= (1.0 / 15.0):
                    for key, frame in (("cam0", frame0), ("cam1", frame1)):
                        board = self.board_px.get(key)
                        if board is None:
                            jpeg = encode_jpeg(frame, self.args.display_width)
                        else:
                            preview = frame
                            scale = 1.0
                            if (self.args.display_width and
                                    frame.shape[1] > self.args.display_width):
                                scale = self.args.display_width / frame.shape[1]
                                preview = cv2.resize(
                                    frame,
                                    (self.args.display_width,
                                     int(round(frame.shape[0] * scale))),
                                    interpolation=cv2.INTER_AREA)
                            else:
                                preview = frame.copy()

                            def scaled(point):
                                if point is None:
                                    return None
                                return [point[0] * scale, point[1] * scale]

                            origin = scaled(board.get("origin"))
                            axes = {
                                name: scaled(point)
                                for name, point in (board.get("axes") or {}).items()
                            }
                            draw_board_axes(preview, origin, axes)
                            draw_board_origin(
                                preview, origin,
                                board.get("label", "board origin (0,0,0)"))
                            draw_board_axes(preview, origin, axes, labels_only=True)
                            jpeg = encode_jpeg(preview)
                        if jpeg is not None:
                            self.views[key].publish(jpeg)
                    last_preview = now

                if self.state.capture_event.is_set():
                    self.state.capture_event.clear()
                    with self.state.lock:
                        self.state.error = None
                        self.state.message = "Saving pair..."
                    per_pair = dict(self.meta_base)
                    per_pair["camera_settings"] = dict(
                        self.meta_base.get("camera_settings", {}))
                    actual = _capture_actual_settings(cap0, cap1, size0, size1)
                    per_pair["camera_settings"].update(actual)
                    per_pair["capture_settings"] = actual
                    try:
                        pair_dir = _save_pair(
                            self.session, frame0, frame1, per_pair, [t0, t1])
                        with self.state.lock:
                            self.state.saved += 1
                            self.state.last_pair = pair_dir.name
                            self.state.message = (
                                f"Saved {pair_dir.name}  "
                                f"({frame0.shape[1]}x{frame0.shape[0]}, "
                                f"{frame1.shape[1]}x{frame1.shape[0]}).  "
                                "Keep the target still for the next pair.")
                        print(f"Saved {pair_dir}  (cam0 "
                              f"{frame0.shape[1]}x{frame0.shape[0]}, cam1 "
                              f"{frame1.shape[1]}x{frame1.shape[0]})")
                    except Exception as exc:  # noqa: BLE001 - show in browser
                        with self.state.lock:
                            self.state.error = str(exc)
                            self.state.message = f"ERROR saving pair: {exc}"
        except Exception as exc:  # noqa: BLE001 - show in browser
            with self.state.lock:
                self.state.error = str(exc)
                self.state.message = f"ERROR opening cameras: {exc}"
                self.state.ready = False
        finally:
            if cap0 is not None:
                cap0.release()
            if cap1 is not None:
                cap1.release()
            for view in self.views.values():
                view.close()


CAPTURE_WEB_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dry stereo capture</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#121212; color:#eee; font-family:system-ui,sans-serif; }
  header { padding:14px 18px; background:#181818; border-bottom:1px solid #2c2c2c; }
  h1 { margin:0 0 6px; font-size:1.15rem; font-weight:600; }
  .sub { color:#aaa; font-size:.85rem; }
  main { padding:16px; }
  .warning { background:#3a2b08; border:1px solid #8a6514; color:#ffd479;
             border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:.9rem; }
  .hint { color:#bbb; font-size:.82rem; line-height:1.45; margin:0 0 14px 2px; }
  .hint b { color:#ffd479; }
  .panels { display:flex; gap:14px; flex-wrap:wrap; }
  .cam { flex:1 1 480px; background:#1b1b1b; border:1px solid #2c2c2c;
         border-radius:10px; padding:10px; }
  .cam h2 { font-size:.9rem; margin:0 0 8px; color:#9ad; font-weight:600; }
  .cam img { display:block; width:100%; height:auto; background:#000;
             border-radius:6px; }
  .controls { position:sticky; bottom:0; margin-top:16px; padding:12px 14px;
              background:#181818; border:1px solid #2c2c2c; border-radius:10px;
              display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  button { background:#2b5fa8; color:#fff; border:0; border-radius:6px;
           padding:11px 18px; font-size:1rem; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  .kv { color:#bbb; font-size:.86rem; }
  .kv b { color:#fff; }
  #error { color:#ff6b6b; white-space:pre-wrap; margin-top:8px; }
  #message { color:#ffd479; margin-top:8px; font-size:.9rem; }
  #meta { color:#999; margin-top:8px; font-size:.82rem; }
</style>
</head>
<body>
<header>
  <h1>Dry stereo capture &mdash; full-resolution C920 pairs</h1>
  <div class="sub" id="session">session: ...</div>
</header>
<main>
  <div class="warning">
    The two C920s are <b>not hardware-synchronized</b>. Keep the reference
    object completely still before and during each capture. The browser stream
    may be downscaled; saved PNGs remain full resolution.
  </div>
  <div class="hint">
    The orange crosshair is the live preview origin. It is shifted along board
    <b>+Y</b> by <code>--preview-origin-y-rows</code> (default <b>+4 rows</b>,
    +176&nbsp;mm) from the true pattern origin; saved board coordinates still
    use <b>(0,0,0)</b>. The arrows are board <b>+X</b> (red), <b>+Y</b>
    (green) and <b>+Z</b> (blue), drawn 50&nbsp;mm long; <b>+Z</b> points
    physically downward. The overlay is preview-only, is not written into
    captured PNGs, and does not live-track a board that has been moved.
  </div>
  <div class="panels">
    <div class="cam">
      <h2>cam0</h2>
      <img src="/video/cam0" alt="cam0 live view">
    </div>
    <div class="cam">
      <h2>cam1</h2>
      <img src="/video/cam1" alt="cam1 live view">
    </div>
  </div>
  <div class="controls">
    <button id="captureButton" disabled>Capture Pair</button>
    <div class="kv">saved <b id="saved">0</b></div>
    <div class="kv">last <b id="lastPair">-</b></div>
    <div class="kv">read gap <b id="gap">-</b></div>
  </div>
  <div id="message">waiting for cameras...</div>
  <div id="error"></div>
  <div id="meta"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let busy = false;

async function tick() {
  try {
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();
    $('session').textContent = 'session: ' + (s.session || '');
    $('saved').textContent = String(s.saved || 0);
    $('lastPair').textContent = s.last_pair || '-';
    $('gap').textContent = (s.gap_ms == null) ? '-' : s.gap_ms.toFixed(1) + ' ms';
    $('message').textContent = s.message || '';
    $('error').textContent = s.error || '';
    $('captureButton').disabled = !s.ready || busy;
    const sizes = s.sizes || {};
    const parts = [];
    if (sizes.cam0) parts.push('cam0 ' + sizes.cam0[0] + 'x' + sizes.cam0[1]);
    if (sizes.cam1) parts.push('cam1 ' + sizes.cam1[0] + 'x' + sizes.cam1[1]);
    if (s.settings && s.settings.actual_fps_cam0) {
      parts.push('fps ' + s.settings.actual_fps_cam0.toFixed(1));
    }
    $('meta').textContent = parts.join(' · ');
  } catch (e) {
    $('message').textContent = 'Cannot reach capture server: ' + e;
  }
}

async function capture() {
  if (busy) return;
  busy = true;
  $('captureButton').disabled = true;
  $('message').textContent = 'Capture requested...';
  try {
    await fetch('/capture', {method:'POST'});
  } catch (e) {
    $('message').textContent = 'Capture request failed: ' + e;
  } finally {
    busy = false;
  }
}

$('captureButton').addEventListener('click', capture);
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""


def create_capture_web_app(state: CaptureState, views: dict, session: Path):
    """Create the browser app for live capture."""
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index():
        return CAPTURE_WEB_TEMPLATE

    @app.route("/video/<key>")
    def video(key):
        view = views.get(key)
        if view is None:
            return ("unknown camera", 404)

        def stream():
            with state.lock:
                state.viewers += 1
            try:
                yield from view.frames()
            finally:
                with state.lock:
                    state.viewers = max(0, state.viewers - 1)

        return Response(stream(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        data = state.snapshot()
        data["session"] = str(session)
        return jsonify(data)

    @app.route("/capture", methods=["POST"])
    def capture():
        with state.lock:
            state.message = "Capture requested; waiting for the next frame pair."
        state.capture_event.set()
        return jsonify({"ok": True})

    return app


def _print_web_url(title: str, host: str, port: int) -> None:
    local_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"\n{title}: http://{local_host}:{port}/")
    if host in ("0.0.0.0", "::"):
        print(f"  From another device: http://<this-machine-ip>:{port}/")
    print("  Press Ctrl+C to stop the web server.")


def cmd_capture(args: argparse.Namespace) -> int:
    if open_camera is None:
        print("ERROR: charuco_common.open_camera is not importable. "
              "Run this script from the camera repository.", file=sys.stderr)
        return 2
    session = Path(args.session)
    cal_paths = ensure_session(session, args.calibration_paths())
    cal = load_calibration(cal_paths)
    if cal.errors:
        print("Calibration errors must be resolved before capturing:", file=sys.stderr)
        for c in cal.errors:
            print(f"  {c['code']}: {c['message']}", file=sys.stderr)
        return 2

    meta_base = {
        "calibration_hashes": cal.hashes,
        "calibration_image_size": list(cal.size0),
        "calibration_image_sizes": {
            "cam0": list(cal.size0), "cam1": list(cal.size1),
        },
        "calibration_warnings": [
            {"code": c["code"], "message": c["message"]}
            for c in cal.checks if c["level"] == "warn"
        ],
        "camera_settings": {
            "cam0": args.cam0, "cam1": args.cam1,
            "requested_width": args.width, "requested_height": args.height,
            "requested_fps": args.fps, "requested_focus": args.focus,
            "requested_exposure": args.exposure,
        },
    }

    if args.auto:
        cap0, size0 = open_camera(args.cam0, args.width, args.height, args.fps,
                                  focus_val=args.focus,
                                  exposure_val=args.exposure)
        cap1, size1 = open_camera(args.cam1, args.width, args.height, args.fps,
                                  focus_val=args.focus,
                                  exposure_val=args.exposure)
        if size0 != cal.size0 or size1 != cal.size1:
            print(f"WARNING: live size {size0}/{size1} differs from calibration "
                  f"{cal.size0}/{cal.size1}.  Measurements require matching sizes.")
        actual = _capture_actual_settings(cap0, cap1, size0, size1)
        meta_extra = dict(meta_base)
        meta_extra["camera_settings"] = dict(meta_base["camera_settings"])
        meta_extra["camera_settings"].update(actual)
        if size0 != cal.size0 or size1 != cal.size1:
            meta_extra["size_mismatch_warning"] = (
                f"live size cam0/cam1 {size0}/{size1} differs from calibration "
                f"{cal.size0}/{cal.size1}; measure will reject these pairs.")

        def grab_and_save() -> Path | None:
            ok0, a = cap0.read()
            t0 = time.time_ns()
            ok1, b = cap1.read()
            t1 = time.time_ns()
            if not (ok0 and ok1):
                print("Capture failed: one camera returned no frame.")
                return None
            # Save exactly what the sensor delivered; no crop/resize/rotation.
            per_pair = dict(meta_extra)
            per_pair["camera_settings"] = dict(meta_extra["camera_settings"])
            per_pair["capture_settings"] = _capture_actual_settings(
                cap0, cap1, size0, size1)
            pair_dir = _save_pair(session, a, b, per_pair, [t0, t1])
            print(f"Saved {pair_dir}  (cam0 {a.shape[1]}x{a.shape[0]}, "
                  f"cam1 {b.shape[1]}x{b.shape[0]})")
            return pair_dir

        try:
            print(f"Headless capture: {args.auto} pair(s), interval "
                  f"{args.interval:.2f} s.  Keep the target stationary.")
            saved = 0
            for _ in range(args.auto):
                time.sleep(max(0.0, args.interval))
                # Discard a few queued frames so both images are fresh.
                for _ in range(2):
                    cap0.grab()
                    cap1.grab()
                if grab_and_save() is not None:
                    saved += 1
            if saved != args.auto:
                print(f"ERROR: saved {saved} of {args.auto} requested pair(s).",
                      file=sys.stderr)
                return 1
            return 0
        finally:
            cap0.release()
            cap1.release()

    if args.no_gui:
        print("ERROR: interactive capture is browser-based now.  Omit "
              "--no-gui, or use --auto N for a headless capture.",
              file=sys.stderr)
        return 2

    try:
        from web_tags import LiveView
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: browser capture needs Flask/web_tags: {exc}",
              file=sys.stderr)
        return 2

    views = {"cam0": LiveView(), "cam1": LiveView()}
    state = CaptureState()
    try:
        square_mm = float(np.asarray(
            cal.tank.get("square_length_mm", 44.0)).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        square_mm = 44.0
    origin_rows = float(args.preview_origin_y_rows)
    origin_y_mm = origin_rows * square_mm
    origin_board_mm = np.array([0.0, origin_y_mm, 0.0], np.float64)
    board_px = project_board_frame_px(
        cal, origin_board_mm=origin_board_mm)
    if abs(origin_rows) < 1e-9:
        origin_label = "board origin (0,0,0)"
    else:
        origin_label = f"origin (0,{origin_y_mm:+.0f},0) mm"
    for item in board_px.values():
        item["label"] = origin_label
    for key, item in board_px.items():
        origin = item.get("origin")
        if origin is None:
            print(f"Board preview origin {key}: not visible / behind camera")
        else:
            print(f"Board preview origin {key}: "
                  f"({origin[0]:.1f}, {origin[1]:.1f}) px")
    if abs(origin_rows) >= 1e-9:
        print(f"Preview origin shifted {origin_rows:+g} board rows "
              f"({origin_y_mm:+g} mm) along +Y; saved board coordinates "
              "still use the true pattern origin.")
    worker = StereoCaptureWorker(args, state, views, session, meta_base,
                                 board_px)
    worker.start()
    app = create_capture_web_app(state, views, session)
    print(f"Capture session: {session}")
    _print_web_url("Capture view", args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, threaded=True,
                use_reloader=False)
    finally:
        state.stop_event.set()
        worker.join(timeout=3.0)
        for view in views.values():
            view.close()
    return 0

# ============================================================
# LABEL INTERFACE
# ============================================================

LABEL_WEB_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dry stereo 3-D validation — correspondences</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#121212; color:#eee; font-family:system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; padding:12px 16px;
           background:#181818; border-bottom:1px solid #2c2c2c; }
  h1 { margin:0 0 8px; font-size:1.1rem; font-weight:600; }
  h1 span { color:#9ad; font-weight:400; }
  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button, select { background:#2b5fa8; color:#fff; border:0; border-radius:6px;
                   padding:8px 11px; font-size:.86rem; cursor:pointer; }
  button.primary { background:#2e8b41; }
  button.danger { background:#8a3030; }
  button:disabled { opacity:.45; cursor:default; }
  select { background:#222; color:#eee; border:1px solid #444; }
  #saveState { color:#ffd479; font-size:.84rem; margin-left:4px; }
  #warning { color:#ffd479; background:#3a2b08; border:1px solid #8a6514;
             border-radius:6px; padding:7px 9px; margin-top:8px; font-size:.82rem; }
  #message { color:#cfe; margin-top:8px; min-height:1.2em; font-size:.86rem;
             white-space:pre-wrap; }
  #message.error { color:#ff6b6b; }
  main { padding:12px; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:1000px) { .panels { grid-template-columns:1fr; } }
  .panel { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:9px;
           padding:8px; min-width:0; }
  .panel.active { border-color:#3a8; }
  .panel h2 { margin:0 0 6px; color:#9ad; font-size:.92rem; font-weight:600; }
  .panel h2 span { color:#aaa; font-weight:400; }
  .canvas-wrap { position:relative; }
  canvas.view { width:100%; height:auto; display:block; background:#000;
                border-radius:5px; cursor:crosshair; touch-action:none; }
  canvas.mag { position:absolute; right:8px; bottom:8px; width:180px;
               height:180px; border:1px solid #aaa; background:#000;
               display:none; border-radius:4px; }
  .hint { color:#999; font-size:.76rem; margin-top:5px; }
  .points { margin-top:14px; background:#1b1b1b; border:1px solid #2c2c2c;
            border-radius:9px; padding:10px; overflow-x:auto; }
  .points h2 { font-size:.95rem; margin:0 0 8px; }
  .points table { min-width:1050px; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; }
  th, td { text-align:left; padding:5px 6px; border-bottom:1px solid #2c2c2c;
           vertical-align:middle; }
  th { color:#9ad; font-weight:600; }
  tr.editing { background:#3a2b08; }
  td button { padding:4px 7px; font-size:.75rem; margin-right:3px; }
  #coords, #geometryNote { color:#999; font-size:.8rem; margin-top:8px;
                           white-space:pre-wrap; }
  #geometryNote { color:#9ad; }
</style>
</head>
<body>
<header>
  <h1>Dry stereo correspondence labelling — <span id="pairName">...</span></h1>
  <div class="toolbar">
    <label>Pair <select id="pairSelect"></select></label>
    <button id="prevButton">Prev</button>
    <button id="nextButton">Next</button>
    <button id="saveButton" class="primary">Save</button>
    <button id="deleteLastButton">Delete last</button>
    <button id="cancelButton">Cancel pending/edit</button>
    <button id="fitButton">Fit views</button>
    <button id="resultsButton">Results</button>
    <button id="compareButton">Compare</button>
    <span id="saveState">saved</span>
  </div>
  <div id="warning">C920s are not synchronized; the reference object must stay still.</div>
  <div id="message"></div>
</header>
<main>
  <div class="panels">
    <section class="panel" id="panel0">
      <h2>cam0 <span id="meta0"></span></h2>
      <div class="canvas-wrap">
        <canvas id="view0" class="view"></canvas>
        <canvas id="mag0" class="mag" width="180" height="180"></canvas>
      </div>
      <div class="hint">left click = set/commit point · wheel = zoom · right/shift-drag = pan</div>
    </section>
    <section class="panel" id="panel1">
      <h2>cam1 <span id="meta1"></span></h2>
      <div class="canvas-wrap">
        <canvas id="view1" class="view"></canvas>
        <canvas id="mag1" class="mag" width="180" height="180"></canvas>
      </div>
      <div class="hint">left click = set/commit point · wheel = zoom · right/shift-drag = pan</div>
    </section>
  </div>
  <div class="points">
    <h2>Correspondence points</h2>
    <table>
      <thead><tr><th>Name</th><th>cam0 raw px</th><th>cam1 raw px</th><th>pixel x,y,z (px)</th><th>board X,Y,Z (mm)</th><th>valid</th><th>Actions</th></tr></thead>
      <tbody id="pointRows"><tr><td colspan="7">no points yet</td></tr></tbody>
    </table>
    <div id="coords"></div>
    <div id="geometryNote">Saved geometry: pixel x,y = cam0 raw px; pixel z = cam0.x − cam1.x (horizontal disparity, px, not metric depth). Board X,Y,Z = triangulated millimetres; recomputed when you save.</div>
  </div>
</main>
<script>
(() => {
  "use strict";
  window.addEventListener("error", (event) => {
    const el = document.getElementById("message");
    if (el) {
      el.textContent = "JavaScript error: " + (event.message || event);
      el.className = "error";
    }
  });
  window.addEventListener("unhandledrejection", (event) => {
    const el = document.getElementById("message");
    if (el) {
      el.textContent = "JavaScript rejection: " + (event.reason || event);
      el.className = "error";
    }
  });
  const DISPLAY_WIDTH = __DISPLAY_WIDTH__;
  const $ = (id) => document.getElementById(id);
  const viewCanvas = (i) => $("view" + i);
  const magCanvas = (i) => $("mag" + i);
  const state = {
    pair: null,
    pairs: [],
    points: [],
    pending: null,
    edit: null,
    dirty: false,
    images: [null, null],
    views: [null, null],
    cursor: [null, null],
    active: 0,
    token: 0,
    drag: null,
    clickStart: null,
  };
  const palette = ["#4fd16f", "#4db8ff", "#ffb84d", "#e879ff", "#fff06a",
                   "#57e6df", "#ff8d57", "#a5ffb0"];

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function pointName(index) {
    let n = index + 1;
    let out = "";
    while (n > 0) {
      n -= 1;
      out = String.fromCharCode(65 + (n % 26)) + out;
      n = Math.floor(n / 26);
    }
    return out;
  }

  function nextName() {
    const used = new Set(state.points.map((p) => p.name));
    let i = 0;
    while (used.has(pointName(i))) i += 1;
    return pointName(i);
  }

  function nextId() {
    return state.points.reduce((m, p) => Math.max(m, Number(p.id) || 0), -1) + 1;
  }

  const POINT_GEOMETRY_FIELDS = [
    "pixel_xyz", "X_cam0_mm", "X_cam1_mm", "X_board_mm",
    "height_above_board_mm", "height_above_tank_floor_mm", "X_tank_mm",
    "X_tank_verified", "valid", "flags", "reproj_error_cam0_px",
    "reproj_error_cam1_px", "epipolar_sampson_px", "parallax_deg",
    "depth_sensitivity_mm_per_px", "geometry_error"
  ];

  function clearPointGeometry(point) {
    for (const field of POINT_GEOMETRY_FIELDS) delete point[field];
  }

  function formatTriplet(value, digits) {
    if (!Array.isArray(value)) return "—";
    return value.map((item) => {
      const number = Number(item);
      return Number.isFinite(number) ? number.toFixed(digits) : "—";
    }).join(", ");
  }

  function hasChanges() {
    return state.dirty || state.pending !== null || state.edit !== null;
  }

  function setMessage(text, isError) {
    const el = $("message");
    el.textContent = text || "";
    el.className = isError ? "error" : "";
  }

  function updateSaveState() {
    const el = $("saveState");
    if (state.dirty || state.pending || state.edit) {
      el.textContent = "unsaved";
      el.style.color = "#ffd479";
    } else {
      el.textContent = "saved";
      el.style.color = "#7CFC7C";
    }
  }

  async function api(url, options) {
    const opts = Object.assign({cache: "no-store"}, options || {});
    const response = await fetch(url, opts);
    let data = null;
    try { data = await response.json(); } catch (e) { /* non-JSON error */ }
    if (!response.ok) {
      let message = "HTTP " + response.status;
      if (data && data.error) message = data.error;
      throw new Error(message);
    }
    return data || {};
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Could not load " + src));
      image.src = src;
    });
  }

  function setupCanvas(panel) {
    const image = state.images[panel];
    const canvas = viewCanvas(panel);
    const maxWidth = (DISPLAY_WIDTH && DISPLAY_WIDTH > 0) ? DISPLAY_WIDTH : 900;
    const width = Math.max(320, Math.min(maxWidth, image.naturalWidth));
    const height = Math.max(180,
      Math.round(width * image.naturalHeight / image.naturalWidth));
    canvas.width = width;
    canvas.height = height;
  }

  function fitView(panel) {
    const image = state.images[panel];
    const canvas = viewCanvas(panel);
    if (!image || !canvas.width) return;
    const scale = Math.min(canvas.width / image.naturalWidth,
                           canvas.height / image.naturalHeight);
    state.views[panel] = {
      scale: scale,
      fitScale: scale,
      offsetX: (canvas.width - image.naturalWidth * scale) / 2,
      offsetY: (canvas.height - image.naturalHeight * scale) / 2,
    };
  }

  function fitAll() {
    fitView(0);
    fitView(1);
    draw();
    updateMeta();
  }

  function rawToScreen(view, raw) {
    return [view.offsetX + raw[0] * view.scale,
            view.offsetY + raw[1] * view.scale];
  }

  function screenToRaw(view, x, y) {
    return [(x - view.offsetX) / view.scale,
            (y - view.offsetY) / view.scale];
  }

  function insideImage(panel, raw) {
    const image = state.images[panel];
    if (!image) return false;
    return raw[0] >= 0 && raw[1] >= 0 &&
           raw[0] <= image.naturalWidth - 1 &&
           raw[1] <= image.naturalHeight - 1;
  }

  function updateMeta() {
    for (let i = 0; i < 2; i += 1) {
      const image = state.images[i];
      const view = state.views[i];
      const el = $("meta" + i);
      if (!image || !view) { el.textContent = "waiting for image"; continue; }
      const zoom = view.fitScale ? (view.scale / view.fitScale) : 1;
      el.textContent = image.naturalWidth + "x" + image.naturalHeight +
                       " · zoom " + zoom.toFixed(1) + "x";
    }
  }

  function drawPoint(ctx, view, point, index, panel) {
    const raw = point["cam" + panel];
    if (!raw) return;
    const [x, y] = rawToScreen(view, raw);
    if (x < -30 || y < -30 || x > ctx.canvas.width + 30 ||
        y > ctx.canvas.height + 30) return;
    const color = palette[index % palette.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 11, y); ctx.lineTo(x + 11, y);
    ctx.moveTo(x, y - 11); ctx.lineTo(x, y + 11);
    ctx.stroke();
    ctx.font = "600 13px system-ui, sans-serif";
    const label = String(point.name || "?");
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,.72)";
    ctx.fillRect(x + 10, y - 23, tw + 8, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 14, y - 9);
  }

  function drawMagnifier(panel) {
    const mag = magCanvas(panel);
    const image = state.images[panel];
    const view = state.views[panel];
    const raw = state.cursor[panel];
    if (!image || !view || !raw || !insideImage(panel, raw)) {
      mag.style.display = "none";
      return;
    }
    mag.style.display = "block";
    const ctx = mag.getContext("2d");
    const mw = mag.width, mh = mag.height;
    const zoom = 6;
    ctx.clearRect(0, 0, mw, mh);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, mw, mh);
    ctx.imageSmoothingEnabled = false;
    const sourceW = mw / zoom;
    const sourceH = mh / zoom;
    ctx.drawImage(image,
                  raw[0] - sourceW / 2, raw[1] - sourceH / 2,
                  sourceW, sourceH, 0, 0, mw, mh);
    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mw / 2 - 14, mh / 2); ctx.lineTo(mw / 2 + 14, mh / 2);
    ctx.moveTo(mw / 2, mh / 2 - 14); ctx.lineTo(mw / 2, mh / 2 + 14);
    ctx.stroke();
    ctx.strokeStyle = "#aaa";
    ctx.strokeRect(0.5, 0.5, mw - 1, mh - 1);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "#fff";
    ctx.fillText(raw[0].toFixed(1) + ", " + raw[1].toFixed(1), 7, mh - 8);
  }

  function draw() {
    for (let panel = 0; panel < 2; panel += 1) {
      const canvas = viewCanvas(panel);
      const ctx = canvas.getContext("2d");
      const image = state.images[panel];
      const view = state.views[panel];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      $("panel" + panel).classList.toggle("active", state.active === panel);
      if (!image || !view) { drawMagnifier(panel); continue; }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(image, view.offsetX, view.offsetY,
                    image.naturalWidth * view.scale,
                    image.naturalHeight * view.scale);
      state.points.forEach((point, index) => drawPoint(ctx, view, point, index, panel));
      if (state.pending && state.pending.panel === panel) {
        const [x, y] = rawToScreen(view, state.pending.xy);
        ctx.strokeStyle = "#ffef4d";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 12, y - 12); ctx.lineTo(x + 12, y + 12);
        ctx.moveTo(x + 12, y - 12); ctx.lineTo(x - 12, y + 12);
        ctx.stroke();
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.fillStyle = "#ffef4d";
        ctx.fillText("pending", x + 14, y + 20);
      }
      if (state.edit && state.edit.panel === panel) {
        const point = state.points.find((p) => p.id === state.edit.id);
        if (point) {
          const [x, y] = rawToScreen(view, point["cam" + panel]);
          ctx.strokeStyle = "#ff8d57";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 13, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.strokeStyle = state.active === panel ? "#45b86b" : "#3a3a3a";
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
      drawMagnifier(panel);
    }
    updateMeta();
    updateCoords();
  }

  function updateCoords() {
    const c0 = state.cursor[0];
    const c1 = state.cursor[1];
    const parts = [];
    parts.push("cam0 cursor: " + (c0 ? c0[0].toFixed(2) + ", " + c0[1].toFixed(2) : "-"));
    parts.push("cam1 cursor: " + (c1 ? c1[0].toFixed(2) + ", " + c1[1].toFixed(2) : "-"));
    if (state.pending) {
      parts.push("pending: cam" + state.pending.panel + " (" +
                 state.pending.xy[0].toFixed(2) + ", " +
                 state.pending.xy[1].toFixed(2) + ")");
    }
    if (state.edit) {
      const p = state.points.find((x) => x.id === state.edit.id);
      parts.push("editing: " + (p ? p.name : "?" ) + " cam" + state.edit.panel);
    }
    $("coords").textContent = parts.join("   ·   ");
  }

  function updatePointTable() {
    const tbody = $("pointRows");
    tbody.innerHTML = "";
    if (!state.points.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "no points yet";
      tr.appendChild(td);
      tbody.appendChild(tr);
      updateSaveState();
      return;
    }
    state.points.forEach((point) => {
      const tr = document.createElement("tr");
      const nameTd = document.createElement("td");
      nameTd.textContent = point.name || "?";
      const c0Td = document.createElement("td");
      c0Td.textContent = (point.cam0 || []).map((v) => Number(v).toFixed(2)).join(", ");
      const c1Td = document.createElement("td");
      c1Td.textContent = (point.cam1 || []).map((v) => Number(v).toFixed(2)).join(", ");
      const pixelTd = document.createElement("td");
      pixelTd.textContent = formatTriplet(point.pixel_xyz, 2);
      const boardTd = document.createElement("td");
      boardTd.textContent = formatTriplet(point.X_board_mm, 3);
      const validTd = document.createElement("td");
      if (point.valid === true) {
        validTd.textContent = "yes";
      } else if (point.valid === false) {
        validTd.textContent = "no";
        if (Array.isArray(point.flags) && point.flags.length) {
          validTd.title = point.flags.join(", ");
        }
      } else {
        validTd.textContent = "save to compute";
      }
      const actions = document.createElement("td");
      for (const panel of [0, 1]) {
        const edit = document.createElement("button");
        edit.textContent = "Edit cam" + panel;
        edit.addEventListener("click", () => startEdit(point, panel));
        actions.appendChild(edit);
      }
      const del = document.createElement("button");
      del.className = "danger";
      del.textContent = "Delete";
      del.addEventListener("click", () => deletePoint(point));
      actions.appendChild(del);
      tr.appendChild(nameTd);
      tr.appendChild(c0Td);
      tr.appendChild(c1Td);
      tr.appendChild(pixelTd);
      tr.appendChild(boardTd);
      tr.appendChild(validTd);
      tr.appendChild(actions);
      if (state.edit && state.edit.id === point.id) tr.classList.add("editing");
      tbody.appendChild(tr);
    });
    updateSaveState();
  }

  function startEdit(point, panel) {
    state.pending = null;
    state.edit = {id: point.id, panel: panel};
    state.active = panel;
    setMessage("Click cam" + panel + " to move point " + point.name + ".");
    updatePointTable();
    draw();
  }

  async function deletePoint(point) {
    if (!window.confirm("Delete point " + point.name + "?")) return;
    state.points = state.points.filter((p) => p.id !== point.id);
    if (state.edit && state.edit.id === point.id) state.edit = null;
    state.dirty = true;
    setMessage("Deleted point " + point.name + ".");
    updatePointTable();
    draw();
  }

  function handleClick(panel, raw) {
    if (!insideImage(panel, raw)) {
      setMessage("Click inside the image area.", true);
      return;
    }
    if (state.edit) {
      if (panel !== state.edit.panel) {
        setMessage("Editing point in cam" + state.edit.panel +
                   "; click that camera, or cancel the edit.", true);
        return;
      }
      const point = state.points.find((p) => p.id === state.edit.id);
      if (point) {
        point["cam" + panel] = [raw[0], raw[1]];
        clearPointGeometry(point);
        state.dirty = true;
        state.edit = null;
        setMessage("Updated " + point.name + " in cam" + panel + ".");
        updatePointTable();
        draw();
      }
      return;
    }
    if (state.pending) {
      if (state.pending.panel === panel) {
        state.pending.xy = [raw[0], raw[1]];
        setMessage("Moved pending point in cam" + panel +
                   ". Click the same point in the other camera.");
        updateSaveState();
        draw();
        return;
      }
      const cam0 = state.pending.panel === 0 ? state.pending.xy : raw;
      const cam1 = state.pending.panel === 1 ? state.pending.xy : raw;
      const point = {
        id: nextId(),
        name: nextName(),
        cam0: [cam0[0], cam0[1]],
        cam1: [cam1[0], cam1[1]],
      };
      state.points.push(point);
      state.pending = null;
      state.dirty = true;
      setMessage("Added " + point.name + ". Save to compute its pixel and 3-D coordinates.");
      updatePointTable();
      draw();
      return;
    }
    state.pending = {panel: panel, xy: [raw[0], raw[1]]};
    setMessage("First point set in cam" + panel +
               ". Click the same physical point in the other camera.");
    updateSaveState();
    draw();
  }

  async function savePair() {
    if (!state.pair) return false;
    if (state.pending) {
      setMessage("Finish the pending point, or click Cancel pending/edit, " +
                 "before saving.", true);
      return false;
    }
    if (state.edit) {
      setMessage("Finish the point edit, or click Cancel pending/edit, " +
                 "before saving.", true);
      return false;
    }
    try {
      const data = await api("/api/pair/" + encodeURIComponent(state.pair) +
                             "/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({points: state.points}),
      });
      if (Array.isArray(data.points)) state.points = data.points;
      state.dirty = false;
      updatePointTable();
      setMessage("Saved " + state.points.length +
                 " point(s) with pixel and board 3-D coordinates to " +
                 state.pair + ".");
      draw();
      return true;
    } catch (error) {
      setMessage(error.message, true);
      return false;
    }
  }

  async function loadPair(name) {
    const token = ++state.token;
    const previous = state.pair;
    setMessage("Loading " + name + "...");
    try {
      const data = await api("/api/pair/" + encodeURIComponent(name));
      if (token !== state.token) return;
      state.pair = name;
      state.points = Array.isArray(data.points) ? data.points : [];
      state.pending = null;
      state.edit = null;
      state.dirty = false;
      state.images = [null, null];
      state.views = [null, null];
      state.cursor = [null, null];
      $("pairSelect").value = name;
      $("pairName").textContent = name;
      const warnings = [];
      if (data.warning) warnings.push(data.warning);
      if (Array.isArray(data.warnings)) warnings.push(...data.warnings);
      $("warning").textContent = warnings.join("  ") ||
        "C920s are not synchronized; the reference object must stay still.";
      const [image0, image1] = await Promise.all([
        loadImage(data.cam0_url), loadImage(data.cam1_url)
      ]);
      if (token !== state.token) return;
      state.images = [image0, image1];
      setupCanvas(0);
      setupCanvas(1);
      fitView(0);
      fitView(1);
      updatePointTable();
      setMessage("Loaded " + name + ". Click a point in cam0, then the same point in cam1.");
      draw();
    } catch (error) {
      if (token === state.token) {
        setMessage("Could not load " + name + ": " + error.message, true);
        if (previous) $("pairSelect").value = previous;
      }
    }
  }

  async function navigate(delta) {
    if (!state.pairs.length || !state.pair) return;
    if (hasChanges() && !(await savePair())) return;
    const index = state.pairs.indexOf(state.pair);
    if (index < 0) return;
    const next = index + delta;
    if (next < 0 || next >= state.pairs.length) return;
    await loadPair(state.pairs[next]);
  }

  function activeView() { return state.views[state.active]; }

  function zoomAt(panel, clientX, clientY, factor) {
    const view = state.views[panel];
    if (!view) return;
    const canvas = viewCanvas(panel);
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) * canvas.width / rect.width;
    const y = (clientY - rect.top) * canvas.height / rect.height;
    const before = screenToRaw(view, x, y);
    const minimum = view.fitScale * 0.5;
    const maximum = view.fitScale * 24;
    view.scale = Math.max(minimum, Math.min(maximum, view.scale * factor));
    view.offsetX = x - before[0] * view.scale;
    view.offsetY = y - before[1] * view.scale;
    state.active = panel;
    draw();
  }

  function canvasPoint(panel, event) {
    const canvas = viewCanvas(panel);
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height,
    };
  }

  function onPointerDown(panel, event) {
    const canvas = viewCanvas(panel);
    const view = state.views[panel];
    if (!view) return;
    state.active = panel;
    if (event.button === 1 || event.button === 2 ||
        (event.button === 0 && event.shiftKey)) {
      event.preventDefault();
      state.drag = {
        panel: panel,
        sx: event.clientX,
        sy: event.clientY,
        ox: view.offsetX,
        oy: view.offsetY,
      };
      try { canvas.setPointerCapture(event.pointerId); } catch (e) {}
      draw();
      return;
    }
    if (event.button !== 0) return;
    const point = canvasPoint(panel, event);
    const raw = screenToRaw(view, point.x, point.y);
    state.clickStart = {
      panel: panel,
      raw: raw,
      sx: event.clientX,
      sy: event.clientY,
    };
    try { canvas.setPointerCapture(event.pointerId); } catch (e) {}
    draw();
  }

  function onPointerMove(panel, event) {
    const view = state.views[panel];
    if (!view) return;
    const canvas = viewCanvas(panel);
    const point = canvasPoint(panel, event);
    state.cursor[panel] = screenToRaw(view, point.x, point.y);
    if (state.drag && state.drag.panel === panel) {
      const rect = canvas.getBoundingClientRect();
      const dx = (event.clientX - state.drag.sx) * canvas.width / rect.width;
      const dy = (event.clientY - state.drag.sy) * canvas.height / rect.height;
      view.offsetX = state.drag.ox + dx;
      view.offsetY = state.drag.oy + dy;
    }
    draw();
  }

  function onPointerUp(panel, event) {
    const canvas = viewCanvas(panel);
    if (state.drag && state.drag.panel === panel) {
      state.drag = null;
      try { canvas.releasePointerCapture(event.pointerId); } catch (e) {}
      return;
    }
    if (state.clickStart && state.clickStart.panel === panel) {
      const moved = Math.hypot(event.clientX - state.clickStart.sx,
                               event.clientY - state.clickStart.sy);
      if (moved < 4) handleClick(panel, state.clickStart.raw);
      state.clickStart = null;
      try { canvas.releasePointerCapture(event.pointerId); } catch (e) {}
    }
  }

  function onPointerLeave(panel) {
    if (state.drag) return;
    state.cursor[panel] = null;
    draw();
  }

  function onWheel(panel, event) {
    event.preventDefault();
    zoomAt(panel, event.clientX, event.clientY,
           event.deltaY < 0 ? 1.25 : 0.8);
  }

  document.querySelectorAll("canvas.view").forEach((canvas, panel) => {
    canvas.addEventListener("pointerdown", (event) => onPointerDown(panel, event));
    canvas.addEventListener("pointermove", (event) => onPointerMove(panel, event));
    canvas.addEventListener("pointerup", (event) => onPointerUp(panel, event));
    canvas.addEventListener("pointercancel", (event) => onPointerUp(panel, event));
    canvas.addEventListener("pointerleave", () => onPointerLeave(panel));
    canvas.addEventListener("wheel", (event) => onWheel(panel, event),
                            {passive: false});
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  });

  $("saveButton").addEventListener("click", () => savePair());
  $("prevButton").addEventListener("click", () => navigate(-1));
  $("nextButton").addEventListener("click", () => navigate(1));
  $("fitButton").addEventListener("click", fitAll);
  $("resultsButton").addEventListener("click", () => {
    window.location.href = "/results";
  });
  $("compareButton").addEventListener("click", () => {
    window.location.href = "/compare";
  });
  $("deleteLastButton").addEventListener("click", () => {
    if (!state.points.length) { setMessage("No points to delete."); return; }
    const point = state.points.pop();
    if (state.edit && state.edit.id === point.id) state.edit = null;
    state.dirty = true;
    setMessage("Deleted last point " + point.name + ".");
    updatePointTable();
    draw();
  });
  $("cancelButton").addEventListener("click", () => {
    state.pending = null;
    state.edit = null;
    setMessage("Cancelled pending point/edit.");
    updatePointTable();
    draw();
  });
  $("pairSelect").addEventListener("change", async (event) => {
    const name = event.target.value;
    const previous = state.pair;
    if (name === previous) return;
    if (hasChanges() && !(await savePair())) {
      event.target.value = previous || "";
      return;
    }
    await loadPair(name);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
    if (event.key === "s") { event.preventDefault(); savePair(); }
    else if (event.key === "n") { event.preventDefault(); navigate(1); }
    else if (event.key === "p") { event.preventDefault(); navigate(-1); }
    else if (event.key === "Escape") {
      state.pending = null;
      state.edit = null;
      setMessage("Cancelled pending point/edit.");
      updatePointTable();
      draw();
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (hasChanges()) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  async function init() {
    try {
      const data = await api("/api/pairs");
      state.pairs = Array.isArray(data.pairs) ? data.pairs : [];
      const select = $("pairSelect");
      select.innerHTML = "";
      for (const name of state.pairs) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      }
      if (!state.pairs.length) {
        setMessage("No pairs available.", true);
        return;
      }
      const requested = new URLSearchParams(window.location.search).get("pair");
      const first = state.pairs.includes(requested) ? requested : state.pairs[0];
      await loadPair(first);
    } catch (error) {
      setMessage("Could not start label page: " + error.message, true);
    }
  }

  init();
})();
</script>
</body>
</html>
"""


COMPARE_WEB_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediction vs actual</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#121212; color:#eee; font-family:system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; padding:12px 16px;
           background:#181818; border-bottom:1px solid #2c2c2c; }
  h1 { margin:0 0 8px; font-size:1.1rem; font-weight:600; }
  h1 span { color:#9ad; font-weight:400; }
  h2 { margin:0 0 8px; font-size:1rem; color:#cfe8ff; }
  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button, a.button { background:#2b5fa8; color:#fff; border:0; border-radius:6px;
                     padding:8px 11px; font-size:.86rem; cursor:pointer;
                     text-decoration:none; display:inline-block; }
  button:hover, a.button:hover { background:#3976c8; }
  button.primary { background:#2e8b41; }
  button.primary:hover { background:#3aa44f; }
  button.secondary, a.button.secondary { background:#3a3a3a; }
  button.secondary:hover, a.button.secondary:hover { background:#4a4a4a; }
  #dirty { color:#ffd479; font-size:.82rem; }
  #message { color:#cfe; margin-top:8px; min-height:1.2em; font-size:.86rem;
             white-space:pre-wrap; }
  #message.error { color:#ff6b6b; }
  main { padding:14px; }
  .note { color:#aaa; font-size:.82rem; line-height:1.45; margin-bottom:9px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
           gap:10px; margin-bottom:14px; }
  .card { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:9px;
          padding:10px 12px; }
  .card .label { color:#999; font-size:.75rem; text-transform:uppercase;
                 letter-spacing:.04em; }
  .card .value { font-size:1.2rem; font-weight:650; margin-top:4px; }
  .card .sub { color:#999; font-size:.76rem; margin-top:3px; }
  .card.warn { border-color:#8a6514; }
  .card.warn .value { color:#ffd479; }
  .card.bad { border-color:#8a3030; }
  .card.bad .value { color:#ff8c8c; }
  .panel { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:9px;
           padding:12px; margin-bottom:14px; }
  .table-wrap { overflow:auto; border:1px solid #292929; border-radius:6px; }
  table { width:100%; border-collapse:collapse; font-size:.79rem; }
  th, td { text-align:left; padding:6px 7px; border-bottom:1px solid #292929;
           vertical-align:middle; }
  th { color:#9ad; font-weight:600; background:#202020; white-space:nowrap; }
  tr:last-child td { border-bottom:0; }
  tr.invalid { background:#2a1717; }
  td.num { text-align:right; font-variant-numeric:tabular-nums;
           white-space:nowrap; }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                font-size:.76rem; color:#cfe8ff; }
  input[type=number] { width:112px; background:#222; color:#eee;
                       border:1px solid #444; border-radius:5px;
                       padding:5px 6px; font-size:.8rem; }
  input[type=number]:focus { outline:1px solid #4d9; border-color:#4d9; }
  .pill { display:inline-block; border-radius:999px; padding:2px 7px;
          font-size:.72rem; white-space:nowrap; }
  .pill.good { background:#173d22; color:#9be7a8; border:1px solid #2e6b3b; }
  .pill.bad { background:#3a1414; color:#ff9a9a; border:1px solid #8a3030; }
  .pill.warn { background:#3a2b08; color:#ffd479; border:1px solid #8a6514; }
  .pill.muted { background:#262626; color:#aaa; border:1px solid #3a3a3a; }
  .pill.info { background:#102d3a; color:#9ad; border:1px solid #2b6175; }
  .warnings { color:#ffd479; font-size:.82rem; line-height:1.5; }
  .warnings li { margin-bottom:3px; }
  .origin-plot { width:100%; overflow:auto; background:#141414;
                 border:1px solid #292929; border-radius:6px; }
  .origin-plot svg { display:block; width:100%; height:auto; min-width:760px;
                     background:#141414; }
  .origin-legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:8px;
                   color:#aaa; font-size:.8rem; }
  .origin-legend span { display:inline-flex; align-items:center; gap:5px; }
  .origin-legend i { display:inline-block; width:9px; height:9px;
                     border-radius:50%; }
  .origin-legend i.actual { background:#ffdf5d; transform:rotate(45deg);
                            border-radius:1px; }
  .muted { color:#888; font-size:.8rem; }
  @media (max-width:700px) {
    .cards { grid-template-columns:1fr 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Prediction vs actual — <span id="sessionLabel">...</span></h1>
  <div class="toolbar">
    <button id="refreshButton">Refresh</button>
    <button id="saveButton" class="primary" disabled>Save actuals</button>
    <a class="button secondary" href="/">Back to labeler</a>
    <a class="button secondary" href="/results">Results</a>
    <span id="dirty"></span>
  </div>
  <div id="message"></div>
</header>
<main>
  <div class="note">
    Predictions are triangulated big-ChArUco-board coordinates in millimetres.
    Enter measured actual board X/Y/Z in millimetres. Board +Z points physically
    downward; use the height column for upward height. Actuals are keyed by
    point name and shared by every pair containing that point. Saving writes
    position tests into <code>measurement_plan.json</code>; calibration files
    are never changed. Pixel-space z is horizontal disparity, not metric depth.
  </div>
  <div id="overview" class="cards"></div>

  <section class="panel">
    <h2>Actual board coordinates</h2>
    <div class="note">Leave a point blank to remove a UI-created actual. Existing
      hand-written position tests are not deleted automatically.</div>
    <div id="actualsTable"></div>
  </section>

  <section class="panel">
    <h2>Predicted vs actual</h2>
    <div class="note">Error = predicted − actual. Positive per-axis values mean
      the prediction is larger than the entered actual.</div>
    <div id="compareTable"></div>
  </section>

  <section class="panel">
    <h2>Board origin and predicted points</h2>
    <div class="note">Origin <code>(0,0,0)</code> is the long-board pattern
      origin. Circles are predicted board points; yellow diamonds are entered
      actuals; the dashed yellow line is the error. The vertical axis is the
      inferred physical up direction (currently approximately
      <code>−Z_board</code>). Point labels are board <code>X,Y,Z</code> in
      millimetres.</div>
    <div id="originPlot"></div>
  </section>

  <section class="panel" id="warningsPanel" hidden>
    <h2>Warnings</h2>
    <ul class="warnings" id="warnings"></ul>
  </section>
</main>
<script>
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const AXES = ["X", "Y", "Z"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const state = {
    data: null,
    actuals: {},
    pointNames: [],
    dirty: false,
    editedPoints: new Set(),
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      node.setAttribute(key, String(value));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function finiteTriplet(value) {
    if (!Array.isArray(value) || value.length !== 3) return null;
    const out = value.map(Number);
    return out.every((item) => Number.isFinite(item)) ? out : null;
  }

  function isNum(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function fmt(value, digits = 3) {
    if (value === null || value === undefined || value === "") return "—";
    let number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (Object.is(number, -0)) number = 0;
    return number.toFixed(digits);
  }

  function fmtInt(value) {
    return isNum(value) ? String(Math.round(value)) : "—";
  }

  function vec(value, digits = 3) {
    if (!Array.isArray(value)) return "—";
    return "[" + value.map((item) => fmt(item, digits)).join(", ") + "]";
  }

  function pill(text, kind) {
    return el("span", "pill " + (kind || "muted"), text);
  }

  function note(text) {
    return el("div", "note", text);
  }

  function card(label, value, sub, kind) {
    const node = el("div", "card" + (kind ? " " + kind : ""));
    node.append(el("div", "label", label));
    node.append(el("div", "value", value));
    if (sub) node.append(el("div", "sub", sub));
    return node;
  }

  function table(columns, rows) {
    const wrap = el("div", "table-wrap");
    const node = el("table");
    const thead = el("thead");
    const headRow = el("tr");
    for (const column of columns) headRow.append(el("th", "", column.label));
    thead.append(headRow);
    node.append(thead);
    const tbody = el("tbody");
    for (const row of rows) {
      const tr = el("tr");
      if (row.__invalid) tr.className = "invalid";
      for (const column of columns) {
        const td = el("td");
        let value;
        if (column.render) value = column.render(row);
        else if (column.get) value = column.get(row);
        else value = row[column.key];
        if (value instanceof Node) td.append(value);
        else td.textContent = value === null || value === undefined
          ? "—" : String(value);
        if (column.cls) {
          td.className = typeof column.cls === "function"
            ? column.cls(row) : column.cls;
        }
        tr.append(td);
      }
      tbody.append(tr);
    }
    node.append(tbody);
    wrap.append(node);
    return wrap;
  }

  function actualValue(point, axis) {
    const values = state.actuals[point];
    if (!Array.isArray(values) || values.length !== 3) return null;
    const value = values[axis];
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function hasActual(point) {
    return AXES.every((_, axis) => actualValue(point, axis) !== null);
  }

  function setActual(point, axis, value) {
    let values = state.actuals[point];
    if (!Array.isArray(values) || values.length !== 3) values = [null, null, null];
    values[axis] = value;
    state.actuals[point] = values;
  }

  function actualsFromData(actuals) {
    const out = {};
    for (const [point, info] of Object.entries(actuals || {})) {
      const truth = info && info.truth_board_mm;
      if (Array.isArray(truth) && truth.length === 3 &&
          truth.every((item) => Number.isFinite(Number(item)))) {
        out[point] = truth.map(Number);
      }
    }
    return out;
  }

  function setDirty(value) {
    state.dirty = value;
    $("dirty").textContent = value ? "unsaved changes" : "";
    $("saveButton").disabled = !value;
  }

  function setMessage(text, isError) {
    $("message").textContent = text || "";
    $("message").className = isError ? "error" : "";
  }

  function predictionError(row) {
    if (!Array.isArray(row.X_board_mm) || !hasActual(row.point)) return null;
    const actual = state.actuals[row.point];
    if (!Array.isArray(actual)) return null;
    const error = [];
    for (let axis = 0; axis < 3; axis += 1) {
      const predicted = Number(row.X_board_mm[axis]);
      const actualValue = Number(actual[axis]);
      if (!Number.isFinite(predicted) || !Number.isFinite(actualValue)) return null;
      error.push(predicted - actualValue);
    }
    return error;
  }

  function errorNorm(error) {
    if (!Array.isArray(error)) return null;
    const sum = error.reduce((total, item) => total + item * item, 0);
    return Math.sqrt(sum);
  }

  function renderOverview() {
    const root = $("overview");
    clear(root);
    if (!state.data) return;
    const predictions = state.data.predictions || [];
    const validComparisons = [];
    for (const row of predictions) {
      if (!row.valid) continue;
      const error = predictionError(row);
      if (error) validComparisons.push(row);
    }
    const norms = validComparisons.map((row) => errorNorm(predictionError(row)))
      .filter((value) => value !== null);
    const mean = norms.length
      ? norms.reduce((sum, value) => sum + value, 0) / norms.length : null;
    const max = norms.length ? Math.max(...norms) : null;
    const withActual = state.pointNames.filter(hasActual).length;
    const invalid = predictions.filter((row) => !row.valid).length;
    root.append(card("Predictions", fmtInt(predictions.length),
      fmtInt(state.pointNames.length) + " distinct point name(s)"));
    root.append(card("Actuals entered", fmtInt(withActual),
      "shared by point name across pairs"));
    root.append(card("Comparisons", fmtInt(validComparisons.length),
      "valid prediction + complete actual"));
    root.append(card("Mean |error|", mean === null ? "—" : fmt(mean, 3) + " mm",
      "board-frame coordinate error"));
    root.append(card("Max |error|", max === null ? "—" : fmt(max, 3) + " mm",
      "largest valid coordinate error"));
    root.append(card("Invalid predictions", fmtInt(invalid),
      "shown but excluded from comparison",
      invalid ? "bad" : ""));
  }

  function renderActuals() {
    const root = $("actualsTable");
    clear(root);
    if (!state.pointNames.length) {
      root.append(note("No labeled points or actual values yet."));
      return;
    }
    const rows = state.pointNames.map((point) => {
      const info = state.data && state.data.actuals
        ? state.data.actuals[point] : null;
      return {
        point,
        info,
        unsaved: state.editedPoints.has(point),
      };
    });
    root.append(table([
      {label: "Point", key: "point"},
      ...AXES.map((axis, index) => ({
        label: "Actual " + axis + " (mm)",
        render: (row) => {
          const input = el("input");
          input.type = "number";
          input.step = "0.001";
          input.placeholder = "—";
          const value = actualValue(row.point, index);
          input.value = value === null ? "" : String(value);
          input.addEventListener("input", () => {
            const text = input.value.trim();
            if (text === "") {
              setActual(row.point, index, null);
            } else {
              const number = Number(text);
              setActual(row.point, index, Number.isFinite(number) ? number : null);
            }
            state.editedPoints.add(row.point);
            setDirty(true);
            updateComparison();
            renderOverview();
          });
          return input;
        },
      })),
      {label: "Plan status", render: (row) => {
        if (row.unsaved) return pill("unsaved", "warn");
        if (!row.info) return pill("not saved", "muted");
        const text = row.info.test_name || "position test";
        return pill(text + (row.info.enabled ? "" : " · disabled"),
          row.info.enabled ? "good" : "warn");
      }},
      {label: "Action", render: (row) => {
        const button = el("button", "secondary", "Clear");
        button.type = "button";
        button.addEventListener("click", () => {
          state.actuals[row.point] = [null, null, null];
          state.editedPoints.add(row.point);
          setDirty(true);
          renderActuals();
          updateComparison();
          renderOverview();
        });
        return button;
      }},
    ], rows));
  }

  function renderComparison() {
    const root = $("compareTable");
    clear(root);
    const predictions = (state.data && state.data.predictions) || [];
    if (!predictions.length) {
      root.append(note("No labeled points yet. Label a pair first."));
      return;
    }
    const rows = predictions.map((row) => {
      const actual = hasActual(row.point) ? state.actuals[row.point] : null;
      const error = predictionError(row);
      const norm = errorNorm(error);
      let status;
      if (!row.valid) status = pill("invalid prediction", "bad");
      else if (!actual) status = pill("no actual", "muted");
      else status = pill("compared", "good");
      return {
        ...row,
        __invalid: !row.valid,
        actual,
        error,
        norm,
        status,
      };
    });
    root.append(table([
      {label: "Pair", key: "pair"},
      {label: "Point", key: "point"},
      {label: "Valid", render: (row) => row.valid
        ? pill("valid", "good") : pill("invalid", "bad")},
      {label: "Predicted board X,Y,Z (mm)",
       render: (row) => vec(row.X_board_mm)},
      {label: "Height (mm)", cls: "num",
       render: (row) => fmt(row.height_above_board_mm)},
      {label: "Predicted pixel x,y,z (px)",
       render: (row) => vec(row.pixel_xyz, 2)},
      {label: "Actual X,Y,Z (mm)",
       render: (row) => row.actual ? vec(row.actual) : "—"},
      {label: "Error X,Y,Z (mm)",
       render: (row) => row.error ? vec(row.error) : "—"},
      {label: "|Error| (mm)", cls: "num",
       render: (row) => row.norm === null ? "—" : fmt(row.norm)},
      {label: "Status", render: (row) => row.status},
      {label: "Flags", render: (row) =>
        Array.isArray(row.flags) && row.flags.length
          ? row.flags.join(", ") : "—"},
    ], rows));
  }

  function updateComparison() {
    renderComparison();
    renderOriginPlot();
    renderOverview();
  }

  function renderWarnings() {
    const panel = $("warningsPanel");
    const list = $("warnings");
    clear(list);
    const warnings = [];
    if (state.data && Array.isArray(state.data.warnings)) {
      warnings.push(...state.data.warnings);
    }
    if (state.data && Array.isArray(state.data.save_warnings)) {
      warnings.push(...state.data.save_warnings);
    }
    if (!warnings.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    for (const warning of warnings) list.append(el("li", "", warning));
  }

  function renderOriginPlot() {
    const root = $("originPlot");
    clear(root);
    if (!state.data) return;

    const predictions = state.data.predictions || [];
    const upValue = finiteTriplet(
      state.data.up && state.data.up.up_board) || [0, 0, -1];
    const palette = ["#7ee787", "#79c0ff", "#ffa657", "#d2a8ff",
                     "#ffdf5d", "#56d4dd", "#ff7b72"];
    const points = [];
    for (const row of predictions) {
      const xyz = finiteTriplet(row.X_board_mm);
      if (!xyz) continue;
      const [x, y, z] = xyz;
      const h = x * upValue[0] + y * upValue[1] + z * upValue[2];
      points.push({...row, x, y, z, h});
    }
    if (!points.length) {
      root.append(note("No finite predicted board points to plot."));
      return;
    }
    const pairNames = [...new Set(points.map((point) => point.pair))];
    for (const point of points) {
      const index = Math.max(0, pairNames.indexOf(point.pair));
      point.color = palette[index % palette.length];
    }

    const actualMap = new Map();
    for (const point of points) {
      if (!hasActual(point.point)) continue;
      const xyz = finiteTriplet(state.actuals[point.point]);
      if (!xyz) continue;
      const [x, y, z] = xyz;
      const h = x * upValue[0] + y * upValue[1] + z * upValue[2];
      actualMap.set(point.point, {x, y, z, h});
    }

    const allValues = points.flatMap((point) =>
      [Math.abs(point.x), Math.abs(point.y), Math.abs(point.h)]);
    for (const actual of actualMap.values()) {
      allValues.push(Math.abs(actual.x), Math.abs(actual.y), Math.abs(actual.h));
    }
    const maxAbs = Math.max(100, ...allValues);
    const axisLength = Math.max(100, Math.min(350, maxAbs * 0.35));

    const cos30 = Math.cos(Math.PI / 6);
    const sin30 = Math.sin(Math.PI / 6);
    const project = (x, y, h) => ({
      x: (x - y) * cos30,
      y: (x + y) * sin30 - h,
    });
    const bounds = [
      {x: 0, y: 0, h: 0},
      {x: axisLength, y: 0, h: 0},
      {x: 0, y: axisLength, h: 0},
      {x: 0, y: 0, h: axisLength},
      {x: 0, y: 0, h: -axisLength},
      ...points,
      ...Array.from(actualMap.values()),
    ];
    const projected = bounds.map((point) => project(point.x, point.y, point.h));
    const minX = Math.min(...projected.map((point) => point.x));
    const maxX = Math.max(...projected.map((point) => point.x));
    const minY = Math.min(...projected.map((point) => point.y));
    const maxY = Math.max(...projected.map((point) => point.y));
    const width = 1150;
    const height = 720;
    const margin = 75;
    const scale = Math.min(
      (width - 2 * margin) / Math.max(maxX - minX, 1e-6),
      (height - 2 * margin) / Math.max(maxY - minY, 1e-6));
    const offsetX = width / 2 - scale * (minX + maxX) / 2;
    const offsetY = height / 2 - scale * (minY + maxY) / 2;
    const screen = (x, y, h) => {
      const point = project(x, y, h);
      return [offsetX + scale * point.x, offsetY + scale * point.y];
    };

    const svg = svgEl("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Board origin and predicted points",
      preserveAspectRatio: "xMidYMid meet",
    });
    svg.append(svgEl("rect", {
      x: 0, y: 0, width, height, fill: "#141414",
    }));

    const addLine = (a, b, attrs) => {
      const p1 = screen(a[0], a[1], a[2]);
      const p2 = screen(b[0], b[1], b[2]);
      svg.append(svgEl("line", {
        x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
        stroke: "#888", "stroke-width": 1.5, ...attrs,
      }));
    };
    const addText = (x, y, text, attrs) => {
      const node = svgEl("text", {
        x, y, fill: "#ddd", "font-size": 12,
        "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        ...attrs,
      });
      node.textContent = text;
      svg.append(node);
    };

    addLine([0, 0, 0], [axisLength, 0, 0],
      {stroke: "#ff7b72", "stroke-width": 2.5});
    addLine([0, 0, 0], [0, axisLength, 0],
      {stroke: "#7ee787", "stroke-width": 2.5});
    addLine([0, 0, 0], [0, 0, axisLength],
      {stroke: "#79c0ff", "stroke-width": 2.5});
    addLine([0, 0, 0], [0, 0, -axisLength],
      {stroke: "#888", "stroke-width": 1.5, "stroke-dasharray": "6 5"});

    const xEnd = screen(axisLength, 0, 0);
    const yEnd = screen(0, axisLength, 0);
    const upEnd = screen(0, 0, axisLength);
    const downEnd = screen(0, 0, -axisLength);
    addText(xEnd[0] + 8, xEnd[1] + 4, "+X (short edge)",
      {fill: "#ff7b72", "font-size": 12, "font-weight": 600});
    addText(yEnd[0] + 8, yEnd[1] + 4, "+Y (long edge)",
      {fill: "#7ee787", "font-size": 12, "font-weight": 600});
    addText(upEnd[0] + 8, upEnd[1], "+up (≈ −Z_board)",
      {fill: "#79c0ff", "font-size": 12, "font-weight": 600});
    addText(downEnd[0] + 8, downEnd[1] + 12, "+Z_board (down)",
      {fill: "#aaa", "font-size": 11});

    for (const point of points) {
      const originToPoint = point.color;
      addLine([0, 0, 0], [point.x, point.y, point.h], {
        stroke: originToPoint, "stroke-width": 1,
        "stroke-dasharray": "3 5", opacity: 0.55,
      });
      addLine([point.x, point.y, 0], [point.x, point.y, point.h], {
        stroke: originToPoint, "stroke-width": 1,
        "stroke-dasharray": "2 4", opacity: 0.35,
      });
    }

    for (const point of points) {
      const actual = actualMap.get(point.point);
      if (!actual) continue;
      addLine([point.x, point.y, point.h],
        [actual.x, actual.y, actual.h], {
          stroke: "#ffdf5d", "stroke-width": 1.8,
          "stroke-dasharray": "5 4", opacity: 0.9,
        });
    }

    for (const [name, actual] of actualMap.entries()) {
      const at = screen(actual.x, actual.y, actual.h);
      const size = 6;
      const marker = svgEl("polygon", {
        points: `${at[0]},${at[1] - size} ${at[0] + size},${at[1]} ` +
                `${at[0]},${at[1] + size} ${at[0] - size},${at[1]}`,
        fill: "#ffdf5d", stroke: "#fff", "stroke-width": 1,
      });
      const title = svgEl("title");
      title.textContent = `${name} actual: X=${fmt(actual.x)}, ` +
        `Y=${fmt(actual.y)}, Z=${fmt(actual.z)} mm`;
      marker.append(title);
      svg.append(marker);
      addText(at[0] + 9, at[1] + 4, `${name} actual`, {
        fill: "#ffdf5d", "font-size": 11,
      });
    }

    points.forEach((point, index) => {
      const at = screen(point.x, point.y, point.h);
      const marker = svgEl("circle", {
        cx: at[0], cy: at[1], r: point.valid ? 6 : 5.5,
        fill: point.valid ? point.color : "#2a1717",
        stroke: point.valid ? point.color : "#ff7b72",
        "stroke-width": point.valid ? 1.5 : 2,
      });
      const title = svgEl("title");
      title.textContent = `${point.pair} / ${point.point}: ` +
        `X=${fmt(point.x)}, Y=${fmt(point.y)}, Z=${fmt(point.z)} mm` +
        (point.valid ? "" : " (invalid)");
      marker.append(title);
      svg.append(marker);

      const label = `${point.point} [${fmt(point.x, 1)}, ` +
        `${fmt(point.y, 1)}, ${fmt(point.z, 1)}]` +
        (point.valid ? "" : " invalid");
      const anchorEnd = at[0] > width * 0.68;
      addText(anchorEnd ? at[0] - 9 : at[0] + 9,
        at[1] - 8 + (index % 3) * 12, label, {
          fill: point.valid ? "#eee" : "#ff9a9a",
          "font-size": 11,
          "text-anchor": anchorEnd ? "end" : "start",
        });
    });

    const origin = screen(0, 0, 0);
    svg.append(svgEl("circle", {
      cx: origin[0], cy: origin[1], r: 5.5,
      fill: "#fff", stroke: "#111", "stroke-width": 1.5,
    }));
    addText(origin[0] + 9, origin[1] - 9, "origin (0, 0, 0)", {
      fill: "#fff", "font-size": 12, "font-weight": 600,
    });

    root.append(svg);

    const legend = el("div", "origin-legend");
    for (const pair of pairNames) {
      const item = el("span");
      const dot = el("i");
      dot.style.background = palette[
        Math.max(0, pairNames.indexOf(pair)) % palette.length];
      item.append(dot, document.createTextNode(pair));
      legend.append(item);
    }
    if (actualMap.size) {
      const item = el("span");
      item.append(el("i", "actual"), document.createTextNode("entered actual"));
      legend.append(item);
    }
    if (points.some((point) => !point.valid)) {
      const item = el("span");
      const dot = el("i");
      dot.style.background = "#2a1717";
      dot.style.border = "1px solid #ff7b72";
      item.append(dot, document.createTextNode("invalid prediction"));
      legend.append(item);
    }
    root.append(legend);
  }

  function renderAll() {
    if (!state.data) return;
    $("sessionLabel").textContent = state.data.session || "";
    state.pointNames = Array.isArray(state.data.point_names)
      ? state.data.point_names : [];
    state.actuals = actualsFromData(state.data.actuals || {});
    state.editedPoints.clear();
    setDirty(false);
    renderOverview();
    renderActuals();
    renderComparison();
    renderOriginPlot();
    renderWarnings();
  }

  async function load() {
    setMessage("Loading predictions…");
    try {
      const response = await fetch("/api/compare", {cache: "no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
      state.data = data;
      renderAll();
      if (!state.pointNames.length) {
        setMessage("No labeled points yet. Label a pair first.");
      } else {
        setMessage("Loaded predictions and saved actuals.");
      }
    } catch (error) {
      setMessage("Could not load predictions: " + error.message, true);
    }
  }

  async function save() {
    const actuals = {};
    const incomplete = [];
    for (const point of state.pointNames) {
      const filled = AXES.filter((_, axis) => actualValue(point, axis) !== null);
      if (filled.length > 0 && filled.length < 3) incomplete.push(point);
      actuals[point] = hasActual(point)
        ? state.actuals[point].map((value) => Number(value)) : null;
    }
    if (incomplete.length) {
      setMessage("Complete all three X/Y/Z values for: " + incomplete.join(", "),
                 true);
      return;
    }
    $("saveButton").disabled = true;
    try {
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({actuals}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
      state.data = data;
      renderAll();
      setMessage("Saved actual values to measurement_plan.json. Run measure " +
                 "to add them to the full result outputs.");
      if (Array.isArray(data.save_warnings) && data.save_warnings.length) {
        setMessage("Saved with warnings:\n  " + data.save_warnings.join("\n  "));
      }
    } catch (error) {
      setDirty(true);
      setMessage("Could not save actual values: " + error.message, true);
    }
  }

  $("refreshButton").addEventListener("click", () => {
    if (state.dirty && !window.confirm("Discard unsaved actual values?")) return;
    state.dirty = false;
    load();
  });
  $("saveButton").addEventListener("click", save);
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  load();
})();
</script>
</body>
</html>
"""


RESULTS_WEB_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dry stereo 3-D results</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#121212; color:#eee; font-family:system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; padding:12px 16px;
           background:#181818; border-bottom:1px solid #2c2c2c; }
  h1 { margin:0 0 8px; font-size:1.1rem; font-weight:600; }
  h1 span { color:#9ad; font-weight:400; }
  h2 { margin:0 0 8px; font-size:1rem; font-weight:600; color:#cfe8ff; }
  h3 { margin:14px 0 6px; font-size:.92rem; color:#9ad; }
  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button, a.button { background:#2b5fa8; color:#fff; border:0; border-radius:6px;
                     padding:8px 11px; font-size:.86rem; cursor:pointer;
                     text-decoration:none; display:inline-block; }
  button:hover, a.button:hover { background:#3976c8; }
  button.secondary, a.button.secondary { background:#3a3a3a; }
  button.secondary:hover, a.button.secondary:hover { background:#4a4a4a; }
  #generatedLabel { color:#999; font-size:.8rem; }
  #message { color:#ffd479; margin-top:8px; min-height:1.2em;
             font-size:.86rem; white-space:pre-wrap; }
  #message.error { color:#ff6b6b; }
  main { padding:14px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(205px,1fr));
           gap:10px; margin-bottom:14px; }
  .card { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:9px;
          padding:10px 12px; min-width:0; }
  .card .label { color:#999; font-size:.76rem; text-transform:uppercase;
                 letter-spacing:.04em; }
  .card .value { font-size:1.25rem; font-weight:650; margin-top:4px;
                 word-break:break-word; }
  .card .sub { color:#999; font-size:.76rem; margin-top:3px; line-height:1.4; }
  .card.pass { border-color:#2e6b3b; }
  .card.pass .value { color:#9be7a8; }
  .card.fail { border-color:#8a3030; }
  .card.fail .value { color:#ff8c8c; }
  .card.warn { border-color:#8a6514; }
  .card.warn .value { color:#ffd479; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
  .tabs button { background:#242424; border:1px solid #3a3a3a; }
  .tabs button.active { background:#2b5fa8; border-color:#2b5fa8; }
  .panel { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:9px;
           padding:12px; margin-bottom:14px; }
  .panel-note { color:#aaa; font-size:.82rem; margin-bottom:8px; }
  .table-wrap { overflow:auto; border:1px solid #292929; border-radius:6px; }
  table { width:100%; border-collapse:collapse; font-size:.79rem; }
  th, td { text-align:left; padding:6px 7px; border-bottom:1px solid #292929;
           vertical-align:top; }
  th { color:#9ad; font-weight:600; background:#202020; white-space:nowrap; }
  tr:last-child td { border-bottom:0; }
  tr.failed { background:#3a1414; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                font-size:.76rem; color:#cfe8ff; }
  .pill { display:inline-block; border-radius:999px; padding:2px 7px;
          font-size:.72rem; white-space:nowrap; }
  .pill.good { background:#173d22; color:#9be7a8; border:1px solid #2e6b3b; }
  .pill.bad { background:#3a1414; color:#ff9a9a; border:1px solid #8a3030; }
  .pill.warn { background:#3a2b08; color:#ffd479; border:1px solid #8a6514; }
  .pill.muted { background:#262626; color:#aaa; border:1px solid #3a3a3a; }
  .pill.info { background:#102d3a; color:#9ad; border:1px solid #2b6175; }
  .diag-list { display:grid; gap:7px; }
  .diag { border:1px solid #2c2c2c; border-radius:7px; padding:8px 10px;
          background:#191919; }
  .diag .head { display:flex; gap:8px; align-items:center; margin-bottom:3px; }
  .diag .msg { color:#ccc; font-size:.82rem; line-height:1.45; }
  .diag.warn { border-color:#6e5211; }
  .diag.error { border-color:#7d2a2a; }
  .image-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr));
                gap:12px; }
  figure { margin:0; min-width:0; }
  figure a { display:block; }
  figure img { display:block; width:100%; height:auto; border-radius:6px;
               background:#000; }
  figcaption { color:#9ad; font-size:.78rem; margin-top:4px; word-break:break-all; }
  .plot img { max-height:78vh; object-fit:contain; background:#fff; }
  .files-list { display:grid; gap:6px; }
  .file-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
              padding:7px 9px; background:#191919; border:1px solid #2c2c2c;
              border-radius:6px; }
  .file-row a { color:#9ad; text-decoration:none; }
  .file-row a:hover { text-decoration:underline; }
  .file-row .size { color:#888; font-size:.76rem; }
  .muted { color:#888; font-size:.8rem; }
  .mono-block { white-space:pre-wrap; word-break:break-word; background:#151515;
                border:1px solid #2c2c2c; border-radius:6px; padding:8px;
                font-size:.78rem; color:#ccc; }
  .split { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
           gap:12px; }
  @media (max-width:600px) {
    .cards { grid-template-columns:1fr 1fr; }
    .image-grid { grid-template-columns:1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Dry stereo 3-D results — <span id="sessionLabel">...</span></h1>
  <div class="toolbar">
    <button id="refreshButton">Refresh</button>
    <a class="button secondary" href="/">Back to labeler</a>
    <a id="summaryLink" class="button secondary" href="#" target="_blank" hidden>Open summary</a>
    <a class="button secondary" href="/compare">Compare actuals</a>
    <span id="generatedLabel"></span>
  </div>
  <div id="message"></div>
</header>
<main>
  <div id="overview" class="cards"></div>
  <div id="tabs" class="tabs" hidden></div>
  <section id="panel-measurements" class="panel" hidden></section>
  <section id="panel-distances" class="panel" hidden></section>
  <section id="panel-points" class="panel" hidden></section>
  <section id="panel-statistics" class="panel" hidden></section>
  <section id="panel-diagnostics" class="panel" hidden></section>
  <section id="panel-images" class="panel" hidden></section>
  <section id="panel-files" class="panel" hidden></section>
</main>
<script>
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let DATA = null;
  let activeTab = null;

  const TABS = [
    {id: "measurements", label: "Measurements", kinds: ["full"]},
    {id: "distances", label: "Distances", kinds: ["preliminary"]},
    {id: "points", label: "Points 3D", kinds: ["full", "preliminary"]},
    {id: "statistics", label: "Statistics", kinds: ["full"]},
    {id: "diagnostics", label: "Diagnostics", kinds: ["full"]},
    {id: "images", label: "Images", kinds: ["full"]},
    {id: "files", label: "Files", kinds: ["full", "preliminary"]},
  ];

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function isNum(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function fmt(value, digits = 3) {
    if (value === null || value === undefined || value === "") return "—";
    let number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (Object.is(number, -0)) number = 0;
    return number.toFixed(digits);
  }

  function fmtInt(value) {
    return isNum(value) ? String(Math.round(value)) : "—";
  }

  function vec(value, digits = 3) {
    if (!Array.isArray(value)) return "—";
    return "[" + value.map((item) => fmt(item, digits)).join(", ") + "]";
  }

  function pill(text, kind) {
    return el("span", "pill " + (kind || "muted"), text);
  }

  function prettyIso(value) {
    if (!value) return "unknown";
    return String(value).replace("T", " ").slice(0, 19);
  }

  function card(label, value, sub, kind) {
    const node = el("div", "card" + (kind ? " " + kind : ""));
    node.append(el("div", "label", label));
    node.append(el("div", "value", value));
    if (sub) node.append(el("div", "sub", sub));
    return node;
  }

  function note(text, className) {
    return el("div", className || "panel-note", text);
  }

  function table(columns, rows, rowClass) {
    const wrap = el("div", "table-wrap");
    const node = el("table");
    const thead = el("thead");
    const headRow = el("tr");
    for (const column of columns) {
      headRow.append(el("th", "", column.label));
    }
    thead.append(headRow);
    node.append(thead);
    const tbody = el("tbody");
    for (const row of rows) {
      const tr = el("tr");
      if (rowClass) {
        const cls = rowClass(row);
        if (cls) tr.className = cls;
      }
      for (const column of columns) {
        const td = el("td");
        let value;
        if (column.render) value = column.render(row);
        else if (column.get) value = column.get(row);
        else value = row[column.key];
        if (value instanceof Node) td.append(value);
        else td.textContent = value === null || value === undefined ? "—" : String(value);
        if (column.cls) {
          td.className = typeof column.cls === "function"
            ? column.cls(row) : column.cls;
        }
        tr.append(td);
      }
      tbody.append(tr);
    }
    node.append(tbody);
    wrap.append(node);
    return wrap;
  }

  function sectionTitle(text, sub) {
    const header = el("div");
    header.append(el("h2", "", text));
    if (sub) header.append(note(sub));
    return header;
  }

  function renderOverview() {
    const root = $("overview");
    clear(root);
    if (!DATA || !DATA.overview) return;
    const o = DATA.overview;
    if (o.mode === "full") {
      const tolerance = DATA.tolerance || {};
      let tolValue = "NOT SET";
      let tolSub = "No PASS/FAIL without a user tolerance.";
      let tolKind = "";
      if (tolerance.tolerance_mm !== null && tolerance.tolerance_mm !== undefined) {
        tolValue = fmt(tolerance.tolerance_mm, 3) + " mm";
        if (tolerance.passed === true) {
          tolSub = "PASS — all measurements within tolerance.";
          tolKind = "pass";
        } else if (tolerance.passed === false) {
          tolSub = "FAIL — " + fmtInt(tolerance.n_failed) + " measurement(s) exceed it.";
          tolKind = "fail";
        }
      }
      root.append(card("Tolerance", tolValue, tolSub, tolKind));
      root.append(card(
        "Measurements",
        fmtInt(o.n_measurements),
        fmtInt(o.n_length_height) + " length/height · " +
        fmtInt(o.n_position) + " position",
      ));
      root.append(card(
        "3-D points",
        fmtInt(o.n_valid_points) + " / " + fmtInt(o.n_points),
        fmtInt(o.n_pairs) + " pair(s) · valid / clicked",
        o.n_valid_points === 0 ? "fail" : "",
      ));
      root.append(card(
        "Accuracy",
        fmt(o.mean_abs_mm, 3) + " mm",
        "mean abs error · RMSE " + fmt(o.rmse_mm, 3) +
        " · max " + fmt(o.max_abs_mm, 3),
      ));
      if (o.position_mean_abs_mm !== null && o.position_mean_abs_mm !== undefined) {
        root.append(card(
          "Position error",
          fmt(o.position_mean_abs_mm, 3) + " mm",
          "mean |reconstructed - truth| in board frame",
        ));
      }
      root.append(card(
        "Stereo baseline",
        fmt(o.baseline_mm, 3) + " mm",
        "saved |T| used for triangulation",
      ));
      root.append(card(
        "Diagnostics",
        fmtInt(o.n_warnings) + " warning(s)",
        fmtInt(o.n_errors) + " error(s) · board-frame coordinates",
        o.n_errors > 0 ? "fail" : (o.n_warnings > 0 ? "warn" : ""),
      ));
    } else if (o.mode === "preliminary") {
      root.append(card("Mode", "PRELIMINARY", "No enabled truth tests.", "warn"));
      root.append(card("Pairs", fmtInt(o.n_pairs), "with correspondences"));
      root.append(card(
        "3-D points",
        fmtInt(o.n_valid_points) + " / " + fmtInt(o.n_points),
        "valid / clicked",
      ));
      root.append(card("Distances", fmtInt(o.n_distances), "pairwise reconstructed distances"));
    }
  }

  function renderTabs(kind) {
    const root = $("tabs");
    clear(root);
    const visible = TABS.filter((tab) => tab.kinds.includes(kind));
    if (!visible.length) {
      root.hidden = true;
      return;
    }
    root.hidden = false;
    for (const tab of visible) {
      const button = el("button", "", tab.label);
      button.type = "button";
      button.dataset.tab = tab.id;
      button.addEventListener("click", () => showTab(tab.id));
      root.append(button);
    }
    activeTab = null;
    showTab(visible[0].id);
  }

  function showTab(id) {
    activeTab = id;
    for (const tab of TABS) {
      const panel = $("panel-" + tab.id);
      if (panel) panel.hidden = tab.id !== id;
    }
    for (const button of $("tabs").querySelectorAll("button")) {
      button.classList.toggle("active", button.dataset.tab === id);
    }
  }

  function renderMeasurements() {
    const panel = $("panel-measurements");
    clear(panel);
    panel.append(sectionTitle("Measurements"));
    if (!DATA.measurements || !DATA.measurements.length) {
      panel.append(note("No enabled length, height or position tests. " +
                        "Add tests to measurement_plan.json and rerun measure."));
      return;
    }
    const tolerance = DATA.tolerance || {};
    const columns = [
      {label: "Pair", key: "pair"},
      {label: "Test", key: "test"},
      {label: "Type", key: "type"},
      {label: "Points", render: (row) => row.p1 + " → " + row.p2},
      {label: "Truth", render: (row) => row.type === "position"
          ? vec(row.truth_board_mm) : fmt(row.truth_mm)},
      {label: "Reconstructed", render: (row) => row.type === "position"
          ? vec(row.reconstructed_board_mm) : fmt(row.reconstructed_mm)},
      {label: "Signed", cls: "num", render: (row) => row.type === "position"
          ? "—" : fmt(row.signed_error_mm)},
      {label: "Abs", cls: "num", render: (row) => fmt(row.abs_error_mm)},
      {label: "%", cls: "num", render: (row) => row.type === "position"
          ? "—" : (row.pct_error === null || row.pct_error === undefined
                   ? "—" : fmt(row.pct_error, 2) + "%")},
      {label: "Details", render: (row) => {
        if (row.type === "position") {
          return "error " + vec([row.error_x_mm, row.error_y_mm, row.error_z_mm]);
        }
        const bits = [];
        if (row.height_basis) bits.push("basis " + row.height_basis);
        if (isNum(row.signed_up_component_mm)) {
          bits.push("up " + fmt(row.signed_up_component_mm) + " mm");
        }
        if (Array.isArray(row.components_board_mm)) {
          bits.push("board " + vec(row.components_board_mm));
        }
        return bits.join(" · ") || "—";
      }},
    ];
    const tol = tolerance.tolerance_mm;
    panel.append(table(columns, DATA.measurements, (row) => {
      if (!isNum(tol) || !isNum(row.abs_error_mm)) return "";
      return row.abs_error_mm > tol ? "failed" : "";
    }));
    if (!isNum(tol)) {
      panel.append(note("No tolerance supplied: errors are shown but no PASS/FAIL " +
                        "judgement is made."));
    }
  }

  function renderDistances() {
    const panel = $("panel-distances");
    clear(panel);
    panel.append(sectionTitle("Preliminary pairwise distances",
      "No truth values are enabled. These are reconstructed distances only."));
    if (!DATA.distances || !DATA.distances.length) {
      panel.append(note("No valid pairwise distances."));
      return;
    }
    const columns = [
      {label: "Pair", key: "pair"},
      {label: "Point 1", key: "p1"},
      {label: "Point 2", key: "p2"},
      {label: "3-D distance", cls: "num", render: (row) => fmt(row.euclidean_mm) + " mm"},
      {label: "Up component", cls: "num", render: (row) => fmt(row.signed_up_mm) + " mm"},
      {label: "Camera-0 distance", cls: "num",
       render: (row) => fmt(row.X_cam0_distance_mm) + " mm"},
      {label: "Height 1", cls: "num", render: (row) => fmt(row.height_p1_mm) + " mm"},
      {label: "Height 2", cls: "num", render: (row) => fmt(row.height_p2_mm) + " mm"},
    ];
    panel.append(table(columns, DATA.distances));
  }

  function renderPoints() {
    const panel = $("panel-points");
    clear(panel);
    panel.append(sectionTitle("Triangulated points",
      "Coordinates are millimetres. Board Z is not physical up; use height_above_board_mm for vertical height."));
    if (!DATA.points || !DATA.points.length) {
      panel.append(note("No reconstructed points."));
      return;
    }
    const columns = [
      {label: "Pair", key: "pair"},
      {label: "Point", key: "point"},
      {label: "Valid", render: (row) => row.valid
          ? pill("valid", "good") : pill("invalid", "bad")},
      {label: "Pixel x,y,z (px)", render: (row) => vec(row.pixel_xyz, 2)},
      {label: "Board X/Y/Z", render: (row) => vec(row.X_board_mm)},
      {label: "Height above board", cls: "num",
       render: (row) => fmt(row.height_above_board_mm) + " mm"},
      {label: "Camera-0 X/Y/Z", render: (row) => vec(row.X_cam0_mm)},
      {label: "Tank X/Y/Z", render: (row) => {
        if (row.X_tank_mm === null || row.X_tank_mm === undefined) {
          return pill("not verified", "warn");
        }
        return vec(row.X_tank_mm);
      }},
      {label: "Reprojection cam0/cam1", render: (row) =>
        fmt(row.reproj_error_cam0_px) + " / " + fmt(row.reproj_error_cam1_px) + " px"},
      {label: "Sampson", cls: "num", render: (row) => fmt(row.epipolar_sampson_px) + " px"},
      {label: "Parallax", cls: "num", render: (row) => fmt(row.parallax_deg) + "°"},
      {label: "Raw cam0 px", render: (row) => vec(row.cam0_raw_px, 2)},
      {label: "Raw cam1 px", render: (row) => vec(row.cam1_raw_px, 2)},
      {label: "Flags", render: (row) =>
        Array.isArray(row.flags) && row.flags.length ? row.flags.join(", ") : "—"},
    ];
    panel.append(table(columns, DATA.points));
  }

  function renderStatistics() {
    const panel = $("panel-statistics");
    clear(panel);
    panel.append(sectionTitle("Summary statistics"));
    const stats = DATA.statistics || {};
    const categories = stats.categories || {};
    const rows = Object.entries(categories).map(([name, data]) => ({
      name, ...(data || {}),
      ...((data || {}).errors || {}),
      scale: (data || {}).scale || {},
    }));
    if (rows.length) {
      panel.append(el("h3", "", "Error categories"));
      panel.append(table([
        {label: "Category", key: "name"},
        {label: "N", cls: "num", render: (row) => fmtInt(row.n)},
        {label: "Mean signed", cls: "num", render: (row) => fmt(row.mean_signed_mm)},
        {label: "Mean abs", cls: "num", render: (row) => fmt(row.mean_abs_mm)},
        {label: "RMSE", cls: "num", render: (row) => fmt(row.rmse_mm)},
        {label: "Max abs", cls: "num", render: (row) => fmt(row.max_abs_mm)},
        {label: "Std", cls: "num", render: (row) => fmt(row.std_signed_mm)},
        {label: "Scale bias", cls: "num", render: (row) =>
          row.scale && row.scale.mean_scale_bias_pct !== undefined
            ? fmt(row.scale.mean_scale_bias_pct, 2) + "%" : "—"},
      ], rows));
    } else {
      panel.append(note("No truth-based statistics."));
    }

    const byTest = stats.by_test || {};
    const testRows = Object.entries(byTest).map(([name, data]) => ({
      name, ...(data || {}),
      ...((data || {}).reconstructed || {}),
      ...((data || {}).errors || {}),
    }));
    if (testRows.length) {
      panel.append(el("h3", "", "Per-test aggregates"));
      panel.append(table([
        {label: "Test", key: "name"},
        {label: "Type", key: "type"},
        {label: "Kind", key: "kind"},
        {label: "N", cls: "num", render: (row) => fmtInt(row.n)},
        {label: "Truth", cls: "num", render: (row) => fmt(row.truth_mm)},
        {label: "Mean recon", cls: "num", render: (row) => fmt(row.mean)},
        {label: "Std", cls: "num", render: (row) => fmt(row.std)},
        {label: "Mean abs", cls: "num", render: (row) => fmt(row.mean_abs_mm)},
        {label: "RMSE", cls: "num", render: (row) => fmt(row.rmse_mm)},
        {label: "Max abs", cls: "num", render: (row) => fmt(row.max_abs_mm)},
        {label: "Range", cls: "num", render: (row) => fmt(row.range)},
      ], testRows));
    }

    const position = stats.position;
    if (position) {
      panel.append(el("h3", "", "Known-position tests"));
      panel.append(table([
        {label: "Quantity", key: "quantity"},
        {label: "N", cls: "num", render: (row) => fmtInt(row.n)},
        {label: "Mean", cls: "num", render: (row) => fmt(row.mean)},
        {label: "Std", cls: "num", render: (row) => fmt(row.std)},
        {label: "Max", cls: "num", render: (row) => fmt(row.max)},
      ], [
        {quantity: "|position error|", ...(position.position_error_mm || {})},
        {quantity: "|X error|", ...(position.abs_x_error_mm || {})},
        {quantity: "|Y error|", ...(position.abs_y_error_mm || {})},
        {quantity: "|Z error|", ...(position.abs_z_error_mm || {})},
      ]));
    }

    if (stats.overall_scale && stats.overall_scale.mean_scale_bias_pct !== undefined) {
      panel.append(note("Overall scale bias: " +
        fmt(stats.overall_scale.mean_scale_bias_pct, 2) + "%. Positive means " +
        "reconstructed distances are larger than truth. This is diagnostic only; " +
        "the calibration is never rescaled automatically."));
    }
  }

  function renderDiagnostics() {
    const panel = $("panel-diagnostics");
    clear(panel);
    panel.append(sectionTitle("Diagnostics and calibration checks"));
    const checks = (DATA.calibration && DATA.calibration.checks) || [];
    if (checks.length) {
      panel.append(el("h3", "", "Calibration checks"));
      panel.append(table([
        {label: "Level", render: (row) => pill(row.level, row.level === "error"
            ? "bad" : (row.level === "warn" ? "warn" :
                       (row.level === "ok" ? "good" : "info")))},
        {label: "Code", key: "code"},
        {label: "Message", key: "message"},
      ], checks));
    }
    const diagnostics = DATA.diagnostics || [];
    panel.append(el("h3", "", "Run diagnostics"));
    if (!diagnostics.length) {
      panel.append(note("No diagnostics."));
    } else {
      const list = el("div", "diag-list");
      for (const diagnostic of diagnostics) {
        const box = el("div", "diag " + (diagnostic.level || ""));
        const head = el("div", "head");
        const kind = diagnostic.level === "error" ? "bad"
          : (diagnostic.level === "warn" ? "warn" : "info");
        head.append(pill(diagnostic.level || "info", kind));
        head.append(el("code", "", diagnostic.code || ""));
        box.append(head);
        box.append(el("div", "msg", diagnostic.message || ""));
        list.append(box);
      }
      panel.append(list);
    }

    const up = DATA.coordinate_frames && DATA.coordinate_frames.up;
    if (up) {
      panel.append(el("h3", "", "Inferred up and camera height"));
      panel.append(table([
        {label: "Quantity", key: "quantity"},
        {label: "Value", key: "value"},
      ], [
        {quantity: "up in board coordinates", value: vec(up.up_board)},
        {quantity: "camera height above board", value: fmt(up.camera_height_above_board_mm) + " mm"},
        {quantity: "method", value: up.method || "—"},
        {quantity: "note", value: up.note || "—"},
      ]));
    }

    const tank = DATA.tank_transform || {};
    panel.append(el("h3", "", "Tank-coordinate status"));
    panel.append(table([
      {label: "Quantity", key: "quantity"},
      {label: "Value", key: "value"},
    ], [
      {quantity: "board→tank is identity", value: tank.is_identity ? "yes" : "no"},
      {quantity: "tank coordinates verified", value: tank.verified ? "yes" : "no"},
      {quantity: "matrix", value: Array.isArray(tank.matrix) ? JSON.stringify(tank.matrix) : "—"},
    ]));
    if (!tank.verified) {
      panel.append(note("Tank coordinates are not reported as established; " +
                        "board-frame coordinates are used.", "panel-note"));
    }

    const floor = DATA.floor_plane || {};
    if (floor.source) {
      panel.append(el("h3", "", "Floor plane check"));
      const rows = [
        {quantity: "source", value: floor.source},
      ];
      if (floor.normal) rows.push({quantity: "normal (board)", value: vec(floor.normal)});
      if (floor.angle_to_up_deg !== undefined) {
        rows.push({quantity: "angle to inferred up", value: fmt(floor.angle_to_up_deg) + "°"});
      }
      if (floor.offset_along_up_mm !== undefined) {
        rows.push({quantity: "offset along up", value: fmt(floor.offset_along_up_mm) + " mm"});
      }
      if (floor.planarity_ratio !== undefined && floor.planarity_ratio !== null) {
        rows.push({quantity: "planarity ratio", value: String(floor.planarity_ratio)});
      }
      panel.append(table([
        {label: "Quantity", key: "quantity"},
        {label: "Value", key: "value"},
      ], rows));
    }

    const cf = DATA.coordinate_frames || {};
    const conventions = Object.entries(cf)
      .filter(([key, value]) => key !== "up" && typeof value === "string");
    if (conventions.length) {
      panel.append(el("h3", "", "Coordinate conventions"));
      const block = el("div", "mono-block");
      block.textContent = conventions.map(([key, value]) => key + ": " + value).join("\n");
      panel.append(block);
    }
  }

  function renderImages() {
    const panel = $("panel-images");
    clear(panel);
    panel.append(sectionTitle("Images",
      "Click an image to open the full-resolution file."));
    let count = 0;
    if (DATA.plot_url) {
      const fig = el("figure", "plot");
      const link = el("a");
      link.href = DATA.plot_url;
      link.target = "_blank";
      const img = el("img");
      img.src = DATA.plot_url;
      img.alt = "3-D reconstruction plot";
      link.append(img);
      fig.append(link);
      fig.append(el("figcaption", "", "reconstruction_3d.png — equal physical scale"));
      panel.append(fig);
      count += 1;
    }
    const annotated = DATA.annotated_images || [];
    if (annotated.length) {
      panel.append(el("h3", "", "Annotated pairs"));
      const grid = el("div", "image-grid");
      for (const item of annotated) {
        const fig = el("figure");
        const link = el("a");
        link.href = item.url;
        link.target = "_blank";
        const img = el("img");
        img.src = item.url;
        img.alt = item.name;
        img.loading = "lazy";
        link.append(img);
        fig.append(link);
        fig.append(el("figcaption", "", item.name));
        grid.append(fig);
      }
      panel.append(grid);
      count += annotated.length;
    }
    if (!count) {
      panel.append(note("No result images. Images are written after a full measure run."));
    }
  }

  function fileSize(bytes) {
    if (!isNum(bytes)) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KiB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MiB";
  }

  function renderFiles() {
    const panel = $("panel-files");
    clear(panel);
    panel.append(sectionTitle("Result files",
      "JSON, CSV, summary text and images in the session's results directory."));
    const files = DATA.files || [];
    if (!files.length) {
      panel.append(note("No result files yet."));
      return;
    }
    const list = el("div", "files-list");
    for (const file of files) {
      const row = el("div", "file-row");
      const link = el("a", "", file.name);
      link.href = file.url;
      link.target = "_blank";
      row.append(link);
      row.append(el("span", "size", fileSize(file.size_bytes)));
      list.append(row);
    }
    panel.append(list);
  }

  function renderAll() {
    const kind = DATA ? DATA.kind : "none";
    renderOverview();
    if (!DATA || kind === "none" || kind === "error") {
      $("tabs").hidden = true;
      for (const tab of TABS) {
        const panel = $("panel-" + tab.id);
        if (panel) panel.hidden = true;
      }
      return;
    }
    renderTabs(kind);
    renderMeasurements();
    renderDistances();
    renderPoints();
    renderStatistics();
    renderDiagnostics();
    renderImages();
    renderFiles();
  }

  async function load() {
    $("message").textContent = "Loading results…";
    $("message").className = "";
    try {
      const response = await fetch("/api/results", {cache: "no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
      DATA = data;
      $("sessionLabel").textContent = data.session || "";
      $("generatedLabel").textContent = data.generated_iso
        ? "generated " + prettyIso(data.generated_iso) : "";
      const summary = (data.files || []).find((file) => file.name === "summary.txt");
      if (summary) {
        $("summaryLink").href = summary.url;
        $("summaryLink").hidden = false;
      } else {
        $("summaryLink").hidden = true;
      }
      if (data.kind === "error") {
        $("message").textContent = data.message || "Could not load results.";
        $("message").className = "error";
      } else if (data.kind === "none") {
        $("message").textContent = data.message ||
          "No results yet. Run measure to create them.";
      } else if (data.kind === "preliminary") {
        $("message").textContent = data.message ||
          "Preliminary distances only; no truth tests are enabled.";
      } else {
        $("message").textContent = "";
      }
      renderAll();
    } catch (error) {
      DATA = {kind: "error", message: String(error.message || error)};
      $("message").textContent = "Could not load results: " + error.message;
      $("message").className = "error";
      renderAll();
    }
  }

  $("refreshButton").addEventListener("click", load);
  load();
})();
</script>
</body>
</html>
"""


LABEL_GEOMETRY_FIELDS = (
    "pixel_xyz",
    "X_cam0_mm",
    "X_cam1_mm",
    "X_board_mm",
    "height_above_board_mm",
    "height_above_tank_floor_mm",
    "X_tank_mm",
    "X_tank_verified",
    "valid",
    "flags",
    "reproj_error_cam0_px",
    "reproj_error_cam1_px",
    "epipolar_sampson_px",
    "parallax_deg",
    "depth_sensitivity_mm_per_px",
    "geometry_error",
)


def _finite_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_triplet(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return arr.tolist() if np.all(np.isfinite(arr)) else None


def _session_label_plan(session: Path) -> dict:
    path = session / "measurement_plan.json"
    if not path.is_file():
        return {}
    try:
        plan = read_json(path)
    except Exception:  # noqa: BLE001 - labels still save without a valid plan
        return {}
    return plan if isinstance(plan, dict) else {}


def enrich_labeled_points(points: list[dict], cal: Calibration,
                          plan: dict) -> tuple[list[dict], dict]:
    """Attach saved pixel and triangulated 3-D geometry to label points.

    The raw ``cam0``/``cam1`` observations remain the authoritative input.
    These fields are derived conveniences for inspection and are recomputed on
    every label save and by ``measure``.
    """
    up_info = board_up_direction(cal, str(plan.get("up_direction", "auto")))
    up = np.asarray(up_info["up_board"], np.float64)
    thickness = _finite_scalar(plan.get("board_thickness_mm"))
    tank_verified = (bool(plan.get("tank_transform_verified", False)) and
                     not cal.tank_transform_is_identity)
    enriched = []
    for point in points:
        item = dict(point)
        for field in LABEL_GEOMETRY_FIELDS:
            item.pop(field, None)
        try:
            x0, y0 = float(item["cam0"][0]), float(item["cam0"][1])
            x1, y1 = float(item["cam1"][0]), float(item["cam1"][1])
            rec = triangulate_point([x0, y0], [x1, y1], cal)
        except Exception as exc:  # noqa: BLE001 - retain the raw observations
            item["pixel_xyz"] = None
            item["valid"] = False
            item["flags"] = ["geometry_error"]
            item["geometry_error"] = str(exc)
            enriched.append(item)
            continue

        X_cam0 = _finite_triplet(rec.get("X_cam0_mm"))
        X_cam1 = _finite_triplet(rec.get("X_cam1_mm"))
        X_board = None
        height_board = None
        height_floor = None
        X_tank = None
        if X_cam0 is not None:
            board_point = cam0_to_board(np.asarray(X_cam0, np.float64), cal)
            X_board = _finite_triplet(board_point)
            if X_board is not None:
                height_board = _finite_scalar(
                    float(np.asarray(X_board, np.float64) @ up))
                if thickness is not None and height_board is not None:
                    height_floor = _finite_scalar(height_board + thickness)
                if tank_verified:
                    X_tank = _finite_triplet(
                        board_to_tank(np.asarray(X_board, np.float64), cal))

        item["pixel_xyz"] = [x0, y0, x0 - x1]
        item["X_cam0_mm"] = X_cam0
        item["X_cam1_mm"] = X_cam1
        item["X_board_mm"] = X_board
        item["height_above_board_mm"] = height_board
        item["height_above_tank_floor_mm"] = height_floor
        item["X_tank_mm"] = X_tank
        item["X_tank_verified"] = bool(tank_verified and X_tank is not None)
        item["valid"] = bool(rec.get("valid", False))
        item["flags"] = list(rec.get("flags", []))
        item["reproj_error_cam0_px"] = _finite_scalar(
            rec.get("reproj_error_cam0_px"))
        item["reproj_error_cam1_px"] = _finite_scalar(
            rec.get("reproj_error_cam1_px"))
        item["epipolar_sampson_px"] = _finite_scalar(
            rec.get("epipolar_sampson_px"))
        item["parallax_deg"] = _finite_scalar(rec.get("parallax_deg"))
        item["depth_sensitivity_mm_per_px"] = _finite_scalar(
            rec.get("depth_sensitivity_mm_per_px"))
        enriched.append(item)

    info = {
        "calculated_iso": utc_now_iso(),
        "pixel_units": "px",
        "three_d_units": "mm",
        "pixel_xyz_order": ["x", "y", "z"],
        "pixel_xyz_convention": (
            "x,y = cam0 raw pixel coordinates; z = cam0.x - cam1.x "
            "(horizontal disparity in pixels). z is not metric depth."),
        "three_d_convention": (
            "triangulated from raw cam0/cam1 pixels with the saved stereo "
            "calibration; Camera-0 and big-ChArUco-board coordinates."),
        "up_board": up_info["up_board"],
        "board_thickness_mm": thickness,
        "tank_transform_verified": tank_verified,
        "calibration_sha256": dict(cal.hashes),
        "note": (
            "Derived geometry is recomputed on every label save and by "
            "measure; the raw cam0/cam1 pixels remain authoritative."),
    }
    return enriched, info


RESULT_OUTPUT_FILES = (
    "summary.txt",
    "measurements.json",
    "measurements.csv",
    "points.csv",
    "preliminary_distances.json",
    "preliminary_distances.csv",
)


def _result_file_entry(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name
    if not path.is_file():
        return None
    return {
        "name": path.name,
        "url": f"/results/file/{quote(path.name, safe='')}",
        "size_bytes": int(path.stat().st_size),
    }


def _result_file_entries(out_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".txt", ".json", ".csv", ".png"):
            entry = _result_file_entry(out_dir, path.name)
            if entry is not None:
                entries.append(entry)
    return entries


def _flatten_points(pair_summaries: list[dict]) -> list[dict]:
    rows = []
    for pair in pair_summaries or []:
        pair_name = str(pair.get("pair") or "")
        points = pair.get("points") or {}
        if not isinstance(points, dict):
            continue
        for name in sorted(points):
            point = points[name]
            if not isinstance(point, dict):
                continue
            row = {"pair": pair_name, "point": name}
            row.update(point)
            if "pixel_xyz" not in row:
                c0 = point.get("cam0_raw_px") or point.get("cam0")
                c1 = point.get("cam1_raw_px") or point.get("cam1")
                try:
                    row["pixel_xyz"] = [
                        float(c0[0]), float(c0[1]),
                        float(c0[0]) - float(c1[0]),
                    ]
                except (TypeError, ValueError, IndexError):
                    row["pixel_xyz"] = None
            rows.append(row)
    return rows


def _result_overview(payload: dict) -> dict:
    kind = payload.get("kind", "none")
    points = payload.get("points") or []
    valid_points = sum(1 for point in points if point.get("valid"))
    if kind == "full":
        stats = payload.get("statistics") or {}
        categories = stats.get("categories") or {}
        errors = (categories.get("all") or {}).get("errors") or {}
        position = (stats.get("position") or {}).get("position_error_mm") or {}
        diagnostics = payload.get("diagnostics") or []
        tolerance = payload.get("tolerance") or {}
        measurements = payload.get("measurements") or []
        return {
            "mode": "full",
            "n_measurements": len(measurements),
            "n_length_height": sum(1 for item in measurements
                                   if item.get("type") != "position"),
            "n_position": sum(1 for item in measurements
                              if item.get("type") == "position"),
            "n_pairs": len(payload.get("pairs") or []),
            "n_points": len(points),
            "n_valid_points": valid_points,
            "mean_abs_mm": errors.get("mean_abs_mm"),
            "rmse_mm": errors.get("rmse_mm"),
            "max_abs_mm": errors.get("max_abs_mm"),
            "position_mean_abs_mm": position.get("mean"),
            "baseline_mm": (payload.get("calibration") or {}).get(
                "stereo_baseline_mm"),
            "tolerance_mm": tolerance.get("tolerance_mm"),
            "tolerance_passed": tolerance.get("passed"),
            "n_failed": tolerance.get("n_failed", 0),
            "n_warnings": sum(1 for item in diagnostics
                              if item.get("level") == "warn"),
            "n_errors": sum(1 for item in diagnostics
                            if item.get("level") == "error"),
        }
    if kind == "preliminary":
        return {
            "mode": "preliminary",
            "n_pairs": len(payload.get("pairs") or []),
            "n_points": len(points),
            "n_valid_points": valid_points,
            "n_distances": len(payload.get("distances") or []),
        }
    return {"mode": kind}


def _results_payload(session: Path) -> dict:
    out_dir = session / "results"
    payload = {
        "kind": "none",
        "session": str(session.resolve()),
        "results_dir": str(out_dir.resolve()),
        "generated_iso": None,
        "message": "No results yet. Run measure to create them.",
        "files": [],
        "annotated_images": [],
        "plot_url": None,
        "measurements": [],
        "pairs": [],
        "points": [],
        "distances": [],
        "statistics": {},
        "diagnostics": [],
        "warnings": [],
        "calibration": {},
        "coordinate_frames": {},
        "tank_transform": {},
        "floor_plane": {},
        "tolerance": {
            "tolerance_mm": None,
            "passed": None,
            "n_failed": 0,
            "failed_tests": [],
        },
        "overview": {"mode": "none"},
    }
    if not out_dir.is_dir():
        return payload

    payload["files"] = _result_file_entries(out_dir)
    plot = _result_file_entry(out_dir, "reconstruction_3d.png")
    payload["plot_url"] = plot["url"] if plot else None
    for path in sorted(out_dir.glob("*_annotated.png")):
        entry = _result_file_entry(out_dir, path.name)
        if entry is not None:
            payload["annotated_images"].append(entry)

    try:
        measurements_path = out_dir / "measurements.json"
        preliminary_path = out_dir / "preliminary_distances.json"
        if measurements_path.is_file():
            result = read_json(measurements_path)
            if not isinstance(result, dict):
                raise ValueError("measurements.json is not a JSON object")
            cal = result.get("calibration") or {}
            tolerance = result.get("tolerance_result")
            if not isinstance(tolerance, dict):
                tolerance = {
                    "tolerance_mm": result.get("tolerance_mm"),
                    "passed": None,
                    "n_failed": 0,
                    "failed_tests": [],
                }
            payload.update({
                "kind": "full",
                "generated_iso": result.get("generated_iso"),
                "session": result.get("session", payload["session"]),
                "calibration": {
                    "files": cal.get("files") or {},
                    "checks": cal.get("checks") or [],
                    "image_size": cal.get("image_size"),
                    "stereo_baseline_mm": cal.get("stereo_baseline_mm"),
                },
                "coordinate_frames": result.get("coordinate_frames") or {},
                "tank_transform": result.get("tank_transform") or {},
                "plan": result.get("plan") or {},
                "measurements": result.get("measurements") or [],
                "pairs": [
                    {key: pair.get(key) for key in (
                        "pair", "n_points", "n_valid",
                        "combined_reproj_rms_px", "max_reproj_px",
                        "min_parallax_deg")}
                    for pair in (result.get("pairs") or [])
                ],
                "points": _flatten_points(result.get("pairs") or []),
                "statistics": result.get("statistics") or {},
                "diagnostics": result.get("diagnostics") or [],
                "warnings": result.get("warnings") or [],
                "floor_plane": result.get("floor_plane") or {},
                "tolerance": tolerance,
                "message": "",
            })
        elif preliminary_path.is_file():
            data = read_json(preliminary_path)
            if not isinstance(data, dict):
                raise ValueError("preliminary_distances.json is not a JSON object")
            payload.update({
                "kind": "preliminary",
                "generated_iso": data.get("generated_iso"),
                "coordinate_frames": {"up": data.get("up") or {}},
                "pairs": data.get("pairs") or [],
                "points": _flatten_points(data.get("pairs") or []),
                "distances": data.get("distances") or [],
                "message": ("No enabled measurement tests; showing pairwise "
                            "reconstructed distances."),
            })
        else:
            payload["message"] = (
                "No measurements.json or preliminary_distances.json found. "
                "Run measure to create results.")
    except Exception as exc:  # noqa: BLE001 - results page must stay loadable
        payload["kind"] = "error"
        payload["message"] = f"Could not read results: {exc}"
    payload["overview"] = _result_overview(payload)
    return payload


def _coerce_actual(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("truth_board_mm")
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("actual value must be [x, y, z] in mm, or null")
    values = [_finite_scalar(item) for item in value]
    if any(item is None for item in values):
        raise ValueError("actual values must be finite numbers in mm")
    return [float(item) for item in values if item is not None]


def _plan_actuals(plan: dict) -> tuple[dict, list[str]]:
    actuals: dict[str, dict] = {}
    warnings: list[str] = []
    tests = plan.get("position_tests", [])
    if not isinstance(tests, list):
        return actuals, ["measurement_plan.json position_tests is not a list"]
    for test in tests:
        if not isinstance(test, dict):
            continue
        point = test.get("point")
        truth = test.get("truth_board_mm")
        if not isinstance(point, str) or not point.strip():
            continue
        if not isinstance(truth, (list, tuple)) or len(truth) != 3:
            continue
        values = [_finite_scalar(item) for item in truth]
        if any(item is None for item in values):
            continue
        if point in actuals:
            warnings.append(
                f"point {point!r} has multiple position tests; using "
                f"{actuals[point].get('test_name')!r}")
            continue
        actuals[point] = {
            "truth_board_mm": [float(item) for item in values
                               if item is not None],
            "test_name": test.get("name"),
            "enabled": enabled(test),
        }
    return actuals, warnings


def _compare_payload(session: Path, cal: Calibration,
                     pairs: list[Path]) -> dict:
    plan = _session_label_plan(session)
    actuals, warnings = _plan_actuals(plan)
    predictions = []
    pair_names = []
    for pair_dir in pairs:
        sizes = _pair_image_sizes(pair_dir)
        if sizes is None:
            warnings.append(
                f"{pair_dir.name}: could not read cam0.png/cam1.png; skipped")
            continue
        if sizes[0] != cal.size0 or sizes[1] != cal.size1:
            warnings.append(
                f"{pair_dir.name}: image size {sizes[0]}/{sizes[1]} differs "
                f"from calibration {cal.size0}/{cal.size1}; skipped")
            continue
        try:
            points = load_correspondences(pair_dir)
        except Exception as exc:  # noqa: BLE001 - keep other pairs usable
            warnings.append(f"{pair_dir.name}: could not read labels: {exc}")
            continue
        if not points:
            warnings.append(f"{pair_dir.name}: no labels saved; skipped")
            continue
        try:
            enriched, _info = enrich_labeled_points(points, cal, plan)
        except Exception as exc:  # noqa: BLE001 - keep other pairs usable
            warnings.append(f"{pair_dir.name}: could not compute geometry: {exc}")
            continue
        pair_names.append(pair_dir.name)
        for point in enriched:
            name = str(point.get("name") or point.get("label") or "?")
            predictions.append({
                "pair": pair_dir.name,
                "point": name,
                "valid": bool(point.get("valid", False)),
                "flags": list(point.get("flags", [])),
                "pixel_xyz": point.get("pixel_xyz"),
                "X_board_mm": point.get("X_board_mm"),
                "X_cam0_mm": point.get("X_cam0_mm"),
                "height_above_board_mm": point.get("height_above_board_mm"),
                "reproj_error_cam0_px": point.get("reproj_error_cam0_px"),
                "reproj_error_cam1_px": point.get("reproj_error_cam1_px"),
                "epipolar_sampson_px": point.get("epipolar_sampson_px"),
                "parallax_deg": point.get("parallax_deg"),
            })
    predictions.sort(key=lambda row: (row["pair"], row["point"]))
    prediction_names = {row["point"] for row in predictions}
    point_names = sorted(prediction_names | set(actuals))
    for point in sorted(actuals):
        if point not in prediction_names:
            warnings.append(
                f"actual value for point {point!r} has no prediction in the "
                "current pairs")
            continue
        point_pairs = {
            row["pair"] for row in predictions
            if row["point"] == point and row["valid"]
        }
        missing = sorted(set(pair_names) - point_pairs)
        if missing:
            warnings.append(
                f"actual value for point {point!r} has no valid prediction "
                f"in: {', '.join(missing)}; measure will fail on those pairs")
    up_info = board_up_direction(cal, str(plan.get("up_direction", "auto")))
    thickness = _finite_scalar(plan.get("board_thickness_mm"))
    tank_verified = (bool(plan.get("tank_transform_verified", False)) and
                     not cal.tank_transform_is_identity)
    return {
        "session": str(session.resolve()),
        "plan_path": str((session / "measurement_plan.json").resolve()),
        "generated_iso": utc_now_iso(),
        "pair_count": len(pairs),
        "pairs_with_predictions": pair_names,
        "prediction_count": len(predictions),
        "predictions": predictions,
        "point_names": point_names,
        "actuals": actuals,
        "warnings": warnings,
        "up": up_info,
        "board_thickness_mm": thickness,
        "tank_transform_verified": tank_verified,
        "calibration": {
            "stereo_baseline_mm": cal.baseline_mm,
            "image_size": list(cal.size0),
        },
    }


def _save_compare_actuals(session: Path, updates: dict) -> tuple[dict, list[str]]:
    plan_path = session / "measurement_plan.json"
    if plan_path.is_file():
        plan = read_json(plan_path)
        if not isinstance(plan, dict):
            raise ValueError("measurement_plan.json is not a JSON object")
    else:
        plan = default_plan(sorted(str(key) for key in updates))
    tests = plan.get("position_tests", [])
    if tests is None:
        tests = []
    if "position_tests" not in plan or plan.get("position_tests") is None:
        plan["position_tests"] = tests
    if not isinstance(tests, list):
        raise ValueError("measurement_plan.json position_tests must be a list")

    warnings: list[str] = []
    existing_names = {
        str(test.get("name")) for test in tests if isinstance(test, dict)
    }
    changed = False
    for raw_point, raw_actual in updates.items():
        point = str(raw_point).strip()
        if not point:
            raise ValueError("actual point names must be non-empty")
        values = _coerce_actual(raw_actual)
        matches = [
            test for test in tests
            if isinstance(test, dict) and str(test.get("point")) == point
        ]
        ui_matches = [test for test in matches
                      if test.get("source") == "web_compare"]
        test = (ui_matches or matches or [None])[0]
        if len(matches) > 1:
            warnings.append(
                f"{point}: multiple position tests exist; editing "
                f"{test.get('name') if isinstance(test, dict) else 'none'!r}")
        if values is None:
            if test is None:
                continue
            if test.get("source") == "web_compare":
                tests.remove(test)
                changed = True
            else:
                warnings.append(
                    f"{point}: existing hand-written position test "
                    f"{test.get('name')!r} was not removed; clear it in "
                    "measurement_plan.json if intended")
            continue
        if test is None:
            base = f"web_compare_{point}"
            name = base
            suffix = 2
            while name in existing_names:
                name = f"{base}_{suffix}"
                suffix += 1
            existing_names.add(name)
            tests.append({
                "name": name,
                "point": point,
                "truth_board_mm": values,
                "enabled": True,
                "source": "web_compare",
            })
        else:
            test["truth_board_mm"] = values
            test["enabled"] = True
        changed = True

    if changed:
        validate_plan(plan)
        write_json(plan_path, plan)
    return plan, warnings


def _pair_image_sizes(pair_dir: Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    img0 = cv2.imread(str(pair_dir / "cam0.png"), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(pair_dir / "cam1.png"), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        return None
    return ((img0.shape[1], img0.shape[0]),
            (img1.shape[1], img1.shape[0]))


def create_label_web_app(session: Path, cal: Calibration, pairs: list[Path],
                         display_width: int):
    """Create the browser correspondence labeler for saved pairs."""
    from flask import Flask, jsonify, request, send_file

    app = Flask(__name__)
    by_name = {pair.name: pair for pair in pairs}
    canvas_width = int(display_width) if display_width and display_width > 0 else 900
    page = LABEL_WEB_TEMPLATE.replace(
        "__DISPLAY_WIDTH__", str(max(320, canvas_width)))

    @app.route("/")
    def index():
        return page

    @app.route("/results")
    def results_page():
        return RESULTS_WEB_TEMPLATE

    @app.route("/compare")
    def compare_page():
        return COMPARE_WEB_TEMPLATE

    @app.route("/api/compare", methods=["GET", "POST"])
    def compare_api():
        if request.method == "GET":
            return jsonify(_compare_payload(session, cal, pairs))
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("actuals"), dict):
            return jsonify({"error": "request needs an actuals object"}), 400
        try:
            _plan, warnings = _save_compare_actuals(session, body["actuals"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": f"could not write measurement_plan.json: "
                                     f"{exc}"}), 500
        payload = _compare_payload(session, cal, pairs)
        payload["save_warnings"] = warnings
        return jsonify(payload)

    @app.route("/api/results")
    def results_api():
        return jsonify(_results_payload(session))

    @app.route("/results/file/<path:name>")
    def results_file(name):
        if (not name or name.startswith(".") or "/" in name or
                "\\" in name):
            return ("not found", 404)
        results_dir = session / "results"
        path = results_dir / name
        if not path.is_file():
            return ("not found", 404)
        return send_file(path, as_attachment=request.args.get("download") == "1")

    @app.route("/api/pairs")
    def pair_list():
        return jsonify({"pairs": [pair.name for pair in pairs]})

    @app.route("/api/pair/<name>")
    def pair_info(name):
        pair_dir = by_name.get(name)
        if pair_dir is None:
            return jsonify({"error": f"unknown pair {name!r}"}), 404
        sizes = _pair_image_sizes(pair_dir)
        if sizes is None:
            return jsonify({"error": f"could not read images in {name}"}), 404
        size0, size1 = sizes
        try:
            points = load_correspondences(pair_dir)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"could not read correspondences: {exc}"}), 500
        try:
            points, label_geometry = enrich_labeled_points(
                points, cal, _session_label_plan(session))
        except Exception as exc:  # noqa: BLE001 - do not lose raw labels
            label_geometry = {"error": str(exc)}
        meta = _load_pair_meta(pair_dir)
        warnings = []
        if size0 != cal.size0 or size1 != cal.size1:
            warnings.append(
                f"image size cam0/cam1 {size0}/{size1} differs from calibration "
                f"{cal.size0}/{cal.size1}; measure will reject this pair.")
        if meta.get("warning"):
            warnings.append(str(meta["warning"]))
        return jsonify({
            "pair": name,
            "points": points,
            "sizes": {"cam0": list(size0), "cam1": list(size1)},
            "cam0_url": f"/pair/{quote(name, safe='')}/cam0.png",
            "cam1_url": f"/pair/{quote(name, safe='')}/cam1.png",
            "warnings": warnings,
            "label_geometry": label_geometry,
        })

    @app.route("/pair/<name>/<camera>.png")
    def pair_image(name, camera):
        pair_dir = by_name.get(name)
        if pair_dir is None or camera not in ("cam0", "cam1"):
            return ("not found", 404)
        path = pair_dir / f"{camera}.png"
        if not path.exists():
            return ("not found", 404)
        response = send_file(path, mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/pair/<name>/save", methods=["POST"])
    def save_pair(name):
        pair_dir = by_name.get(name)
        if pair_dir is None:
            return jsonify({"error": f"unknown pair {name!r}"}), 404
        sizes = _pair_image_sizes(pair_dir)
        if sizes is None:
            return jsonify({"error": f"could not read images in {name}"}), 404
        size0, size1 = sizes
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("points"), list):
            return jsonify({"error": "request needs a points list"}), 400
        points = body["points"]
        try:
            errors = validate_correspondences(points, size0, size1)
        except Exception as exc:  # noqa: BLE001 - malformed client payload
            return jsonify({"error": f"invalid points payload: {exc}"}), 400
        if errors:
            return jsonify({"error": "Invalid correspondences:\n  " +
                                     "\n  ".join(errors)}), 400
        points, label_geometry = enrich_labeled_points(
            points, cal, _session_label_plan(session))
        data = {
            "pair_id": pair_dir.name,
            "image_size": [size0[0], size0[1]],
            "image_sizes": {
                "cam0": [size0[0], size0[1]],
                "cam1": [size1[0], size1[1]],
            },
            "updated_iso": utc_now_iso(),
            "coordinate_convention": "raw full-resolution pixel coordinates, "
                                     "origin top-left, x right, y down",
            "label_geometry": label_geometry,
            "points": points,
        }
        write_json(correspondence_path(pair_dir), data)
        return jsonify({"ok": True, "points": points,
                        "label_geometry": label_geometry,
                        "saved_iso": data["updated_iso"]})

    return app


def cmd_label(args: argparse.Namespace) -> int:
    session = Path(args.session)
    cal_paths = ensure_session(session, args.calibration_paths(), quiet=True)
    cal = load_calibration(cal_paths)
    if cal.errors:
        for c in cal.errors:
            print(f"ERROR {c['code']}: {c['message']}", file=sys.stderr)
        return 2
    pairs = list_pair_dirs(session)
    if args.pair:
        wanted = set(args.pair)
        pairs = [p for p in pairs if p.name in wanted or str(p) in wanted]
    if not pairs:
        print(f"No saved pairs under {session / 'pairs'}.  Run capture first.")
        return 2
    try:
        app = create_label_web_app(session, cal, pairs, args.display_width)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: browser labeler needs Flask: {exc}", file=sys.stderr)
        return 2
    print(f"Label session:  {session}")
    print(f"Pairs available: {len(pairs)}")
    _print_web_url("Label view", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


# ============================================================
# SELF TEST
# ============================================================

def _collect_point_names(session: Path) -> list[str]:
    names = set()
    for pair_dir in list_pair_dirs(session):
        for p in load_correspondences(pair_dir):
            name = p.get("name") or p.get("label")
            if name:
                names.add(str(name))
    return sorted(names)


def cmd_plan(args: argparse.Namespace) -> int:
    session = Path(args.session)
    pairs = list_pair_dirs(session)
    if not pairs:
        print(f"No pairs under {session / 'pairs'}.  Capture first.")
        return 2
    names = _collect_point_names(session)
    path = session / "measurement_plan.json"
    if path.exists() and not args.force:
        print(f"Plan already exists: {path} (use --force to overwrite)")
        return 0
    plan = default_plan(names)
    write_json(path, plan)
    print(f"Wrote measurement-plan template to {path}")
    print("Edit length_tests/height_tests/position_tests with your measured "
          "truth values, then run: python validate_stereo_3d.py measure")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    session = Path(args.session)
    cal_paths = ensure_session(session, args.calibration_paths(), quiet=True)
    cal = load_calibration(cal_paths)
    if cal.errors:
        for c in cal.errors:
            print(f"ERROR {c['code']}: {c['message']}", file=sys.stderr)
        return 2
    names = _collect_point_names(session)
    plan, plan_path, created = load_or_create_plan(session, names)
    if args.physical_baseline_mm is not None:
        plan["physical_baseline_mm"] = args.physical_baseline_mm
    validate_plan(plan)
    plan_enabled = ([t for t in plan.get("length_tests", []) if enabled(t)] +
                    [t for t in plan.get("height_tests", []) if enabled(t)] +
                    [t for t in plan.get("position_tests", []) if enabled(t)])
    if created or not plan_enabled:
        print(f"Measurement plan has no enabled tests: {plan_path}")
        if names:
            print(f"Available point names: {', '.join(names)}")
        preliminary_distances(session, cal, plan, pair_filter=args.pair)
        if created:
            print("\nThe plan template was created.  Fill in truth_mm values and "
                  "re-run measure.")
        else:
            print("\nAdd length_tests/height_tests/position_tests to the plan and "
                  "re-run measure.")
        return 0

    print(f"\n--- Dry stereo validation ---")
    print(f"Session: {session}")
    print(f"Plan:    {plan_path}")
    print(f"Stereo baseline used for triangulation: {cal.baseline_mm:.3f} mm")
    print(f"Up: {board_up_direction(cal, plan.get('up_direction', 'auto'))['up_board']}")
    result = run_measurement(session, plan, cal,
                             tolerance_mm=args.tolerance_mm,
                             pair_filter=args.pair)
    write_outputs(session, result, cal, plan)
    print(f"\nWrote results to {session / 'results'}")
    print(f"  {session / 'results' / 'summary.txt'}")
    print(f"  {session / 'results' / 'measurements.json'}")
    print(f"  {session / 'results' / 'measurements.csv'}")
    print(f"  {session / 'results' / 'reconstruction_3d.png'}")
    print("\n" + (session / "results" / "summary.txt").read_text(encoding="utf-8"))
    return 0


# ============================================================
# SELF TEST
# ============================================================

def cmd_selftest(args: argparse.Namespace) -> int:
    """Synthetic end-to-end test using the installed calibration files."""
    session = Path(tempfile.mkdtemp(prefix="validate_stereo_3d_selftest_"))
    try:
        cal_paths = ensure_session(session, args.calibration_paths(), quiet=True)
        cal = load_calibration(cal_paths)
        if cal.errors:
            for c in cal.errors:
                print(f"ERROR {c['code']}: {c['message']}", file=sys.stderr)
            return 2

        up_info = board_up_direction(cal, "auto")
        up = np.asarray(up_info["up_board"], np.float64)

        # Well-conditioned reference points near the middle of the long board.
        A = np.array([44.0, 650.0, 0.0])
        B = np.array([132.0, 650.0, 0.0])
        C = np.array([88.0, 780.0, 0.0])
        D = C + 40.0 * up
        pts = {"A": A, "B": B, "C": C, "D": D}

        # Project from board -> cam0 -> cam1 using the saved transforms.
        corr = []
        for name, Pb in pts.items():
            X0 = board_to_cam0(Pb, cal)
            X1 = cal.R @ X0 + cal.T
            if X0[2] <= 0 or X1[2] <= 0:
                raise RuntimeError("selftest point is behind a camera")
            p0, _ = cv2.projectPoints(X0.reshape(1, 1, 3), np.zeros(3),
                                      np.zeros(3), cal.K0, cal.D0)
            p1, _ = cv2.projectPoints(X1.reshape(1, 1, 3), np.zeros(3),
                                      np.zeros(3), cal.K1, cal.D1)
            xy0 = p0.reshape(2)
            xy1 = p1.reshape(2)
            if not (0 <= xy0[0] < cal.size0[0] and 0 <= xy0[1] < cal.size0[1] and
                    0 <= xy1[0] < cal.size1[0] and 0 <= xy1[1] < cal.size1[1]):
                raise RuntimeError("selftest point projected outside an image")
            corr.append({"id": len(corr), "name": name,
                         "cam0": xy0.tolist(), "cam1": xy1.tolist()})

        # Synthetic but nonempty images at the real resolution.
        img0 = np.full((cal.size0[1], cal.size0[0], 3), 18, np.uint8)
        img1 = np.full((cal.size1[1], cal.size1[0], 3), 18, np.uint8)
        for p in corr:
            cv2.circle(img0, (int(round(p["cam0"][0])), int(round(p["cam0"][1]))),
                       5, (255, 255, 255), -1)
            cv2.circle(img1, (int(round(p["cam1"][0])), int(round(p["cam1"][1]))),
                       5, (255, 255, 255), -1)
        pair_dir = session / "pairs" / "pair_0001_selftest"
        pair_dir.mkdir(parents=True)
        cv2.imwrite(str(pair_dir / "cam0.png"), img0)
        cv2.imwrite(str(pair_dir / "cam1.png"), img1)
        write_json(pair_dir / "pair.json", {
            "pair_id": pair_dir.name,
            "created_iso": utc_now_iso(),
            "synthetic_selftest": True,
        })
        write_json(correspondence_path(pair_dir), {
            "pair_id": pair_dir.name,
            "image_size": list(cal.size0),
            "points": corr,
        })

        horizontal_truth = float(np.linalg.norm(
            (B - A) - ((B - A) @ up) * up))
        plan = default_plan(list(pts))
        plan.update({
            "board_thickness_mm": 0.0,
            "tank_transform_verified": False,
            "length_tests": [
                {"name": "horizontal_synthetic", "p1": "A", "p2": "B",
                 "kind": "horizontal", "truth_mm": horizontal_truth},
                {"name": "vertical_synthetic", "p1": "C", "p2": "D",
                 "kind": "vertical", "truth_mm": 40.0},
            ],
            "height_tests": [
                {"name": "C_height_zero", "point": "C", "truth_mm": 0.0},
                {"name": "D_height_40", "point": "D", "truth_mm": 40.0},
            ],
            "position_tests": [
                {"name": "A_known_position", "point": "A",
                 "truth_board_mm": [44.0, 650.0, 0.0]},
            ],
            "floor_reference_points": ["A", "B", "C"],
        })
        plan_path = session / "measurement_plan.json"
        write_json(plan_path, plan)

        result = run_measurement(session, plan, cal, tolerance_mm=0.05)
        write_outputs(session, result, cal, plan)

        errors = [m["abs_error_mm"] for m in result["measurements"]
                  if m["abs_error_mm"] is not None]
        max_err = max(errors, default=float("inf"))
        valid = [p for pair in result["pairs"]
                 for p in pair["points"].values() if p["valid"]]
        print(f"Selftest session: {session}")
        print(f"  reconstructed points valid: {len(valid)}/4")
        print(f"  measurements: {len(result['measurements'])}")
        print(f"  maximum absolute truth error: {max_err:.6f} mm")
        print(f"  tolerance: 0.05 mm")
        print(f"  result: {'PASS' if max_err <= 0.05 else 'FAIL'}")
        print(f"  outputs: {session / 'results'}")
        if max_err > 0.05:
            print((session / "results" / "summary.txt").read_text(
                encoding="utf-8"), file=sys.stderr)
            return 1
        if args.keep:
            print("  kept temporary session (--keep)")
        else:
            shutil.rmtree(session, ignore_errors=True)
        return 0
    finally:
        if not args.keep and session.exists():
            shutil.rmtree(session, ignore_errors=True)


# ============================================================
# CLI
# ============================================================

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session", default=DEFAULT_SESSION,
                   help="Validation session directory (default: validation)")
    p.add_argument("--calib-dir", default=str(SCRIPT_DIR),
                   help="Directory containing the four calibration .npz files")
    p.add_argument("--cam0-intrinsics", default=None)
    p.add_argument("--cam1-intrinsics", default=None)
    p.add_argument("--stereo-params", default=None)
    p.add_argument("--tank-alignment", default=None)


def _calibration_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = Path(args.calib_dir)
    return {
        "cam0_intrinsics.npz": Path(args.cam0_intrinsics) if args.cam0_intrinsics
        else base / "cam0_intrinsics.npz",
        "cam1_intrinsics.npz": Path(args.cam1_intrinsics) if args.cam1_intrinsics
        else base / "cam1_intrinsics.npz",
        "stereo_params.npz": Path(args.stereo_params) if args.stereo_params
        else base / "stereo_params.npz",
        "tank_alignment.npz": Path(args.tank_alignment) if args.tank_alignment
        else base / "tank_alignment.npz",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dry 3-D stereo validation (capture, label, triangulate, "
                    "compare with known lengths/heights).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Capture full-resolution stereo pairs")
    add_common_args(cap)
    cap.add_argument("--cam0", default="0")
    cap.add_argument("--cam1", default="1")
    cap.add_argument("--width", type=int, default=1920)
    cap.add_argument("--height", type=int, default=1080)
    cap.add_argument("--fps", type=int, default=30)
    cap.add_argument("--focus", type=int, default=30)
    cap.add_argument("--exposure", type=int, default=None)
    cap.add_argument("--display-width", type=int, default=900,
                     help="Browser preview width per camera (saved frames stay "
                          "full size)")
    cap.add_argument("--preview-origin-y-rows", type=float, default=4.0,
                     help="Shift the live-preview origin along board +Y by "
                          "this many square rows (preview only; saved "
                          "coordinates still use the true pattern origin)")
    cap.add_argument("--auto", type=int, default=0,
                     help="Headless: capture this many pairs without a web UI")
    cap.add_argument("--interval", type=float, default=1.0,
                     help="Seconds between automatic captures")
    cap.add_argument("--no-gui", action="store_true",
                     help="Only valid with --auto; interactive capture is web-based")
    cap.add_argument("--host", default="0.0.0.0",
                     help="Browser server bind address")
    cap.add_argument("--port", type=int, default=5002,
                     help="Browser server TCP port")
    cap.set_defaults(func=cmd_capture)

    lab = sub.add_parser("label", help="Select point correspondences in saved pairs")
    add_common_args(lab)
    lab.add_argument("--pair", action="append", default=None,
                     help="Only this pair directory name (repeatable)")
    lab.add_argument("--display-width", type=int, default=900,
                     help="Canvas width per camera in the browser")
    lab.add_argument("--host", default="0.0.0.0",
                     help="Browser server bind address")
    lab.add_argument("--port", type=int, default=5003,
                     help="Browser server TCP port")
    lab.set_defaults(func=cmd_label)

    plan = sub.add_parser("plan", help="Write a measurement-plan template")
    add_common_args(plan)
    plan.add_argument("--force", action="store_true")
    plan.set_defaults(func=cmd_plan)

    meas = sub.add_parser("measure", help="Triangulate and compare with truth values")
    add_common_args(meas)
    meas.add_argument("--pair", action="append", default=None,
                      help="Only this pair directory name (repeatable)")
    meas.add_argument("--tolerance-mm", type=float, default=None,
                      help="Optional user-justified absolute-error tolerance")
    meas.add_argument("--physical-baseline-mm", type=float, default=None,
                      help="Independent physical lens-centre baseline for "
                           "comparison only (never used to rescale)")
    meas.set_defaults(func=cmd_measure)

    self_test = sub.add_parser("selftest", help="Run a synthetic end-to-end test")
    add_common_args(self_test)
    self_test.add_argument("--keep", action="store_true",
                           help="Keep the temporary selftest session")
    self_test.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.calibration_paths = lambda: _calibration_paths(args)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        print(f"\nERROR: {message}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - unexpected error: show detail
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
