"""Stage 4 -- camera-to-wave-tank ("tank") alignment.

``calibrate_studio.py`` answers *"where is camera 1 relative to camera 0?"*.
This module answers *"where is each camera relative to the tank?"* by observing
the **original long 4x35 floor board** (44 mm squares, 33 mm markers), which is
laid in the tank to define the tank's physical coordinate frame.

Unlike stereo extrinsics, the two cameras do **not** need to see the board at
the same instant: the board is a fixed landmark, so each camera is solved
independently with ``cv2.solvePnP`` using the saved intrinsics.  Each valid
picture is kept (repeated poses are not rejected) and all of a camera's
pictures are pooled into one solve, averaging out detection jitter.  This keeps
working when the tank (and its board) is wider than either camera's field of
view -- the requirement is only that each camera sees a well-spread patch of the
board.

Board frame convention (from ``CharucoBoard.getChessboardCorners``)::

    +x  along the short (squaresX) edge
    +y  along the long  (squaresY) edge
    +z  out of the printed face

Sitting face-up on the floor, ``+z`` points up.  ``--world-t`` / ``--tank-t``
(mm) and ``--world-rpy`` / ``--tank-rpy`` (degrees, fixed-axis XYZ:
roll about x, pitch about y, yaw about z, ``R = Rz @ Ry @ Rx``) place that
board frame inside the tank frame.  The defaults make the tank frame identical
to the board frame (x across, y along, z up).

Output ``tank_alignment.npz``::

    R_board_to_cam0 (3,3), t_board_to_cam0 (3,1)   X_cam0 = R @ X_board + t
    R_board_to_cam1, t_board_to_cam1               same for cam1
    R_board_to_tank (3,3), t_board_to_tank (3,1)   X_tank = R @ X_board + t
    square_length_mm (), marker_length_mm ()       board geometry used
    board_size (2,)                                (squaresX, squaresY)
    T_world_board (4,4)                            board -> tank transform
    stereo_scale (str)                             'provisional' or 'validated'

    # Derived camera poses in the tank frame (kept for existing consumers):
    R_cam0_world, t_cam0_world, cam0_center_world
    R_cam1_world, t_cam1_world, cam1_center_world
    cam0_frames, cam1_frames                       pictures combined per camera
    board (str), image_size (2,)

Run::

    python calibrate_world.py --board long --cam0 0 --cam1 1 --save-images

By default it keeps and combines 10 valid pictures per camera (``--samples``).
"""

import argparse
import os

import cv2
import numpy as np

from charuco_common import add_board_argument, get_board, open_camera

#: Fewest board corners a single camera needs for a usable planar pose.  IPPE
#: needs four; six gives the solve room to breathe on the thin 4-wide strip.
MIN_WORLD_CORNERS = 6


# ============================================================
# FRAME MATH
# ============================================================

def parse_triplet(text, name):
    """Parse ``"a,b,c"`` / ``"a b c"`` into a length-3 float64 array."""
    parts = [p for p in str(text).replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"{name} needs three numbers, got {text!r}")
    return np.array([float(p) for p in parts], dtype=np.float64)


def board_to_world(translation="0,0,0", rpy_deg="0,0,0"):
    """Return the 4x4 board->tank transform from ``--tank-t`` / ``--tank-rpy``.

    ``rpy_deg`` is fixed-axis XYZ (extrinsic roll,pitch,yaw) in degrees:
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    r, p, y = np.radians(parse_triplet(rpy_deg, "--tank-rpy"))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]], dtype=np.float64)
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=np.float64)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = parse_triplet(translation, "--tank-t")
    return T


def draw_detection(frame, corners, ids, label=""):
    """Return a copy of ``frame`` with ChArUco corners/IDs drawn on it."""
    out = frame.copy()
    n = 0 if ids is None else int(len(ids))
    if n:
        cv2.aruco.drawDetectedCornersCharuco(
            out, np.asarray(corners).reshape(-1, 1, 2), ids)
    cv2.putText(out, f"{label}  charuco corners: {n}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def _solve_board_points(obj, img, K, D, T_world_board):
    """Least-squares board->camera pose from one or more captures' points."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)
    D = np.asarray(D, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None

    R_cb = cv2.Rodrigues(rvec)[0]              # board -> camera
    t_cb = tvec.reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    R_wb = T_world_board[:3, :3]               # board -> world
    t_wb = T_world_board[:3, 3]
    R_cw = R_cb @ R_wb.T                       # world -> camera
    t_cw = t_cb - R_cw @ t_wb
    center_world = -R_cw.T @ t_cw              # camera centre in world
    return {"ok": True, "rms": rms, "n": len(obj),
            "R_board_cam": R_cb, "t_board_cam": t_cb,
            "R_cam_world": R_cw, "t_cam_world": t_cw,
            "center_world": center_world,
            "obj": obj, "img": img}


def solve_camera_world(corners, ids, K, D, board, board_3d, T_world_board):
    """Pose of one camera in the world frame from a single board view.

    Returns a dict with ``ok`` plus, on success, ``rms`` (px), the raw
    board->camera pose (``R_board_cam``, ``t_board_cam``), ``R_cam_world``,
    ``t_cam_world`` (``X_cam = R @ X_world + t``), ``center_world`` (camera
    centre, mm), ``n`` (corners used) and the matched ``obj``/``img`` points.
    """
    if ids is None:
        return {"ok": False, "reason": "no board"}
    idx = np.asarray(ids).flatten().astype(int)
    obj = np.asarray(board_3d, np.float64)[idx].reshape(-1, 3)
    img = np.asarray(corners, np.float64).reshape(-1, 2)
    if len(obj) < MIN_WORLD_CORNERS:
        return {"ok": False, "reason": f"only {len(obj)} corners"}

    res = _solve_board_points(obj, img, K, D, T_world_board)
    if res is None:
        return {"ok": False, "reason": "solvePnP failed"}
    return res


def combine_camera_world(results, K, D, T_world_board):
    """Combine every accepted capture for one camera into a single pose.

    Captures are deliberately *not* checked for uniqueness: each picture's
    corner observations are pooled into one least-squares solve, so repeated or
    very similar views are all kept and their jitter averages out.
    """
    frames = [r for r in results if r is not None and r.get("ok")]
    if not frames:
        return {"ok": False, "reason": "no board"}
    if len(frames) == 1:
        res = dict(frames[0])
        res["frames"] = 1
        return res

    obj = np.concatenate([r["obj"] for r in frames], axis=0)
    img = np.concatenate([r["img"] for r in frames], axis=0)
    res = _solve_board_points(obj, img, K, D, T_world_board)
    if res is None:
        return {"ok": False, "reason": "solvePnP failed"}
    res["frames"] = len(frames)
    return res


def pack_world_params(best, T_world_board, spec, image_size,
                      stereo_scale="provisional"):
    """Build the ``tank_alignment.npz`` payload from per-camera solve results."""
    out = {
        # Board -> tank placement supplied by the user (measured, not solved).
        "R_board_to_tank": T_world_board[:3, :3].copy(),
        "t_board_to_tank": T_world_board[:3, 3].reshape(3, 1).copy(),
        "T_world_board": T_world_board,
        "square_length_mm": np.array(spec.square_mm, dtype=np.float64),
        "marker_length_mm": np.array(spec.marker_mm, dtype=np.float64),
        "board_size": np.array(spec.grid, dtype=int),
        "board": spec.name,
        "image_size": np.array(image_size, dtype=int),
        "stereo_scale": np.array(stereo_scale),
        "stereo_scale_provisional": np.array(stereo_scale == "provisional"),
    }
    for key in ("cam0", "cam1"):
        res = best[key]
        if res is None:
            continue
        out[f"R_board_to_{key}"] = res["R_board_cam"]
        out[f"t_board_to_{key}"] = res["t_board_cam"].reshape(3, 1)
        out[f"R_{key}_world"] = res["R_cam_world"]
        out[f"t_{key}_world"] = res["t_cam_world"].reshape(3, 1)
        out[f"{key}_center_world"] = res["center_world"]
        out[f"{key}_frames"] = np.array(res.get("frames", 1), dtype=int)
    return out


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Stage 4 tank alignment from the long floor board.")
    p.add_argument("--cam0", default="0", help="First camera ID")
    p.add_argument("--cam1", default="1", help="Second camera ID")
    p.add_argument("--param0", default="cam0_intrinsics.npz")
    p.add_argument("--param1", default="cam1_intrinsics.npz")
    p.add_argument("--stereo", default="stereo_params.npz",
                   help="Used only to cross-check the baseline")
    p.add_argument("--output", default="tank_alignment.npz",
                   help="Output .npz path")
    p.add_argument("--world-t", "--tank-t", dest="world_t", default="0,0,0",
                   help="Board origin position in the tank frame (mm), 'x,y,z'")
    p.add_argument("--world-rpy", "--tank-rpy", dest="world_rpy", default="0,0,0",
                   help="Board orientation in the tank frame (deg, fixed-axis "
                        "XYZ roll,pitch,yaw), 'r,p,y'")
    p.add_argument("--stereo-scale", choices=["provisional", "validated"],
                   default="provisional",
                   help="Record whether stereo_params.npz metric scale is "
                        "provisional (default) or validated")
    p.add_argument("--save-images", action="store_true",
                   help="Save an annotated PNG snapshot of the best frame for "
                        "each camera next to the output file")
    p.add_argument("--focus", type=int, default=30, help="Manual focus value (0-255)")
    p.add_argument("--exposure", type=int, default=None,
                   help="Freeze manual exposure (disables auto-exposure/WB)")
    p.add_argument("--samples", type=int, default=10,
                   help="Pictures to combine per camera (default 10); every "
                        "valid picture is kept, even if it repeats a pose")
    p.add_argument("--max-frames", type=int, default=6000, help="Hard frame stop")
    p.add_argument("--headless", action="store_true",
                   help="No GUI window (auto-enabled when DISPLAY is unset)")
    add_board_argument(p)
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.board != "long":
        raise SystemExit(
            "Stage 4 tank alignment requires --board long (4x35, 44/33 mm); "
            f"got {args.board!r}. The small board is only for intrinsics.")
    spec = get_board(args.board)
    board, detector = spec.board, spec.detector
    headless = args.headless or not os.environ.get("DISPLAY")

    K, D = {}, {}
    for key, path in (("cam0", args.param0), ("cam1", args.param1)):
        d = np.load(path)
        K[key] = d["camera_matrix"].astype(np.float64)
        D[key] = d["dist_coeffs"].astype(np.float64)

    T_world_board = board_to_world(args.world_t, args.world_rpy)
    board_3d = np.asarray(board.getChessboardCorners(), np.float64)

    cap0, _ = open_camera(args.cam0, 1920, 1080, 30, focus_val=args.focus,
                          exposure_val=args.exposure)
    cap1, _ = open_camera(args.cam1, 1920, 1080, 30, focus_val=args.focus,
                          exposure_val=args.exposure)

    print("\n--- Stage 4: Camera-to-Tank Alignment ---")
    print(f"Board: {spec.label} | {spec.grid[0]}x{spec.grid[1]} squares, "
          f"{spec.square_mm:g} mm tiles, {spec.marker_mm:g} mm markers")
    print("Lay the long floor board in the tank and keep BOTH cameras on it.")
    print("Each camera is solved independently; every valid picture is kept and "
          f"combined, up to {args.samples} per camera.")
    print(f"Stereo metric scale recorded as: {args.stereo_scale}\n")

    captures = {"cam0": [], "cam1": []}
    best_frame = {"cam0": None, "cam1": None}
    best_rms = {"cam0": float("inf"), "cam1": float("inf")}
    total = 0
    image_size = None

    while True:
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1:
            break
        total += 1
        image_size = (frame0.shape[1], frame0.shape[0])

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        corners0, ids0, _, _ = detector.detectBoard(gray0)
        corners1, ids1, _, _ = detector.detectBoard(gray1)

        for key, corners, ids, frame in (("cam0", corners0, ids0, frame0),
                                         ("cam1", corners1, ids1, frame1)):
            if len(captures[key]) >= args.samples:
                continue
            res = solve_camera_world(corners, ids, K[key], D[key], board,
                                     board_3d, T_world_board)
            if not res["ok"]:
                continue
            # Keep every valid picture: repeated/similar views are averaged
            # together at the end instead of being rejected.
            captures[key].append(res)
            if res["rms"] < best_rms[key]:
                best_rms[key] = res["rms"]
                if args.save_images:
                    best_frame[key] = draw_detection(frame, corners, ids, key)
            c = res["center_world"]
            print(f"  {key}: picture {len(captures[key])}/{args.samples}  "
                  f"rms {res['rms']:5.2f}px  {res['n']:3d} corners  "
                  f"centre ({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f}) mm")

        if all(len(captures[k]) >= args.samples for k in captures):
            break
        if total >= args.max_frames:
            print("Hit frame limit.")
            break

        if not headless:
            disp = np.hstack((cv2.resize(frame0, (960, 540)),
                              cv2.resize(frame1, (960, 540))))
            cv2.imshow("World Calibration (Cam 0 | Cam 1)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap0.release()
    cap1.release()
    if not headless:
        cv2.destroyAllWindows()

    best = {}
    for key in ("cam0", "cam1"):
        combined = combine_camera_world(captures[key], K[key], D[key],
                                        T_world_board)
        best[key] = combined if combined.get("ok") else None

    if best["cam0"] is None or best["cam1"] is None:
        print("Need a valid board view in BOTH cameras; nothing saved.")
        return

    out = pack_world_params(best, T_world_board, spec, image_size,
                            stereo_scale=args.stereo_scale)
    np.savez(args.output, **out)

    if args.save_images:
        stem = os.path.splitext(args.output)[0]
        for key, img in best_frame.items():
            if img is None:
                continue
            path = f"{stem}_{key}.png"
            if cv2.imwrite(path, img):
                print(f"Saved annotated {key} frame to {path}")
            else:
                print(f"WARNING: could not write annotated frame {path}")

    c0 = best["cam0"]["center_world"]
    c1 = best["cam1"]["center_world"]
    baseline = float(np.linalg.norm(c0 - c1))
    print("\nCamera pose in the tank frame (mm):")
    print(f"  cam0 centre (combined {best['cam0'].get('frames', 1)} pictures): "
          f"({c0[0]:.1f}, {c0[1]:.1f}, {c0[2]:.1f})")
    print(f"  cam1 centre (combined {best['cam1'].get('frames', 1)} pictures): "
          f"({c1[0]:.1f}, {c1[1]:.1f}, {c1[2]:.1f})")
    print(f"  camera-to-camera baseline: {baseline:.1f} mm")
    if os.path.exists(args.stereo):
        T = np.load(args.stereo)["T"]
        print(f"  stereo_params.npz baseline: {np.linalg.norm(T):.1f} mm "
              f"({args.stereo_scale} scale)")
    print(f"Saved tank alignment to {args.output}\n")


if __name__ == "__main__":
    main()
