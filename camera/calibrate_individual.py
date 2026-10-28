import argparse
import os
import cv2
import numpy as np
from charuco_common import (add_board_argument, get_board, open_camera,
                            StabilizedGrabber)

MIN_CORNERS = 10

def is_usable_view(obj_pts):
    """Reject views that are too sparse or degenerate (every detected corner on a
    single board row or column): OpenCV's homography-based intrinsic init
    (`initIntrinsicParams2D`) asserts on collinear points.

    The default long board is a very elongated 4x35 grid, so a *good* view of
    IT is itself extreme (eigenvalue ratio ~0.007 even with the whole board
    visible).  Testing the raw elongation therefore rejected every full-board
    capture, so test the actual degeneracy instead: a usable view must span at
    least two board rows *and* two board columns.  Object points are exact
    multiples of the square size, so distinct coordinates identify distinct
    rows/columns regardless of board shape.
    """
    if obj_pts is None or len(obj_pts) < MIN_CORNERS:
        return False
    pts = np.asarray(obj_pts, np.float64).reshape(-1, 3)
    cols = np.unique(np.round(pts[:, 0], 3))
    rows = np.unique(np.round(pts[:, 1], 3))
    return cols.size >= 2 and rows.size >= 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0", help="Camera ID (0 or 1)")
    parser.add_argument("--output", default="cam_intrinsics.npz", help="Output file path")
    parser.add_argument("--samples", type=int, default=20, help="Target number of frames")
    parser.add_argument("--focus", type=int, default=30, help="Manual focus value (0-255)")
    parser.add_argument("--exposure", type=int, default=None,
                        help="Freeze manual exposure (disables auto-exposure/WB); tune so the "
                             "board is well lit. Try ~100-250 on a C920.")
    parser.add_argument("--headless", action="store_true",
                        help="Auto-capture without a GUI window (auto-enabled when DISPLAY is unset)")
    parser.add_argument("--max-frames", type=int, default=60000,
                        help="Hard stop after this many frames in headless mode "
                             "(only a safety net; the run stops once samples are reached)")
    parser.add_argument("--settle", type=int, default=6,
                        help="Consecutive still frames required before auto-capturing")
    parser.add_argument("--motion", type=float, default=1.5,
                        help="Max corner displacement (px) between frames to still count as 'still'")
    add_board_argument(parser)
    args = parser.parse_args()

    spec = get_board(args.board)
    board, detector = spec.board, spec.detector

    # Over SSH (no DISPLAY) the bundled Qt can't create windows, so fall back to headless
    headless = args.headless or not os.environ.get("DISPLAY")

    cap, image_size = open_camera(args.camera, 1920, 1080, 30, focus_val=args.focus,
                                  exposure_val=args.exposure)

    all_corners = []
    all_ids = []
    object_points = []
    image_points = []
    dropped = 0
    grabber = StabilizedGrabber(settle=args.settle, motion_px=args.motion)

    print("\n--- Intrinsic Calibration ---")
    print(f"Board: {spec.label} | {spec.grid[0]}x{spec.grid[1]} squares, "
          f"{spec.square_mm:g} mm tiles")
    print("Move board to cover edges, corners, and varying angles.")
    if headless:
        print(f"Headless: captures are taken automatically after the board is still "
              f"({args.settle} still frames). Hold each pose briefly.\n")
    else:
        print("[SPACE]: Save frame  |  [Q]: Finish and process calibration\n")

    total_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)

        valid_frame = (ids is not None) and (len(ids) >= MIN_CORNERS)

        if headless:
            if valid_frame:
                det = {int(i): tuple(float(v) for v in corners[k].reshape(-1))
                       for k, i in enumerate(ids.flatten())}
                capture = grabber.update(det)
                if capture is not None:
                    ids_sorted = sorted(capture)
                    avg_corners = np.array([capture[i] for i in ids_sorted], np.float32).reshape(-1, 1, 2)
                    avg_ids = np.array(ids_sorted, np.int32).reshape(-1, 1)
                    obj_pts, img_pts = board.matchImagePoints(avg_corners, avg_ids, None, None)
                    if obj_pts is not None and len(obj_pts) >= MIN_CORNERS and is_usable_view(obj_pts):
                        object_points.append(obj_pts)
                        image_points.append(img_pts)
                        print(f"Captured pose {len(object_points)}/{args.samples} "
                              f"({len(ids_sorted)} corners).")
                    else:
                        dropped += 1
                        print(f"Skipped pose: too few corners, or the board is edge-on - "
                              f"show a proper 2D patch of the board (not just one row/column).")
                elif total_frames % 30 == 0:
                    if grabber.state == "settling":
                        print(f"[{total_frames:5d}] board detected - hold still, "
                              f"{grabber.stable}/{args.settle} still frames")
                    elif grabber.state == "await":
                        print(f"[{total_frames:5d}] pose captured - move to a new pose")
                    else:
                        print(f"[{total_frames:5d}] no board detected in view")
            elif total_frames % 30 == 0:
                print(f"[{total_frames:5d}] no board detected in view")

            if len(object_points) >= args.samples or total_frames >= args.max_frames:
                break
            continue

        display = frame.copy()
        color = (0, 255, 0) if valid_frame else (0, 0, 255)

        if valid_frame:
            cv2.aruco.drawDetectedCornersCharuco(display, corners.reshape(-1, 1, 2), ids)

        cv2.putText(display, f"Captures: {len(all_ids)} / {args.samples}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow(f"Single Calibration - Cam {args.camera}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and valid_frame:
            all_corners.append(corners)
            all_ids.append(ids)
            print(f"Recorded frame {len(all_ids)} with {len(ids)} detected corners.")
        elif key == ord('q'):
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("Computing camera intrinsic parameters...")

    # GUI manual captures (space key) still need filtering after the loop
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids, None, None)
        if is_usable_view(obj_pts):
            object_points.append(obj_pts)
            image_points.append(img_pts)
        else:
            dropped += 1

    if dropped and all_corners:
        print(f"Note: skipped {dropped} captures (too few or near-collinear corners). "
              "Prefer poses where the board fills most of the frame.")

    if len(object_points) < 10:
        print("Error: Too few valid captures for calibration. Minimum 10 required.")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(f"\nCalibration Results:")
    print(f"  Reprojection RMS Error: {ret:.4f} pixels (Ideal: < 0.5)")
    print(f"  Focal Length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
    print(f"  Principal Point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")

    # Pose-diversity summary: focal length is only constrained by poses where the
    # board is rotated OUT of the image plane, so report how much tilt we got.
    tilts, dists = [], []
    for rvec, tvec in zip(rvecs, tvecs):
        Rm, _ = cv2.Rodrigues(rvec)
        tilts.append(np.degrees(np.arccos(min(1.0, abs(float(Rm[2, 2]))))))
        dists.append(float(np.linalg.norm(tvec)))
    tilts, dists = np.array(tilts), np.array(dists)
    n_tilted = int((tilts > 20).sum())
    print(f"\nPose diversity ({len(tilts)} poses):")
    print(f"  out-of-plane tilt : min {tilts.min():.0f} / median {np.median(tilts):.0f} / "
          f"max {tilts.max():.0f} deg  ({n_tilted} poses > 20 deg)")
    print(f"  board distance    : {dists.min():.0f} - {dists.max():.0f} mm")
    if n_tilted < 4 or np.median(tilts) < 10:
        print("  WARNING: too few tilted poses - focal length is weakly constrained. "
              "Tilt the board 20-45 deg about BOTH axes in several poses.")

    # Flag a degenerate/degenerate-looking solve instead of silently trusting it.
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    w = image_size[0]
    if ret > 1.0:
        print(f"  WARNING: RMS {ret:.2f}px is high (want < 0.5). The solve is likely "
              "under-constrained - TILT the board out of the image plane in several "
              "poses, don't just translate/rotate it flat.")
    if not (0.6 * w <= fx <= 1.6 * w):
        print(f"  WARNING: fx={fx:.0f} implies a "
              f"{np.degrees(2 * np.arctan(w / 2 / fx)):.0f} deg horizontal FOV, "
              "implausible for a normal camera - focal length is unconstrained "
              "(usually too little out-of-plane board rotation).")
    if abs(fx / fy - 1.0) > 0.1:
        print(f"  WARNING: fx/fy = {fx / fy:.2f}; should be ~1.0 for square pixels.")

    np.savez(args.output, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, image_size=image_size)
    print(f"Successfully saved to {args.output}\n")

if __name__ == "__main__":
    main()
