import argparse
import os
import cv2
import numpy as np
from charuco_common import (add_board_argument, get_board, open_camera,
                            StabilizedGrabber)

MIN_COMMON = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", default="0", help="First camera ID")
    parser.add_argument("--cam1", default="1", help="Second camera ID")
    parser.add_argument("--param0", default="cam0_intrinsics.npz", help="Intrinsics file for cam0")
    parser.add_argument("--param1", default="cam1_intrinsics.npz", help="Intrinsics file for cam1")
    parser.add_argument("--output", default="stereo_params.npz", help="Output file path")
    parser.add_argument("--focus", type=int, default=30, help="Manual focus value (0-255)")
    parser.add_argument("--exposure", type=int, default=None,
                        help="Freeze manual exposure (disables auto-exposure/WB); tune so the "
                             "board is well lit. Try ~100-250 on a C920.")
    parser.add_argument("--samples", type=int, default=15, help="Target number of stereo pairs")
    parser.add_argument("--headless", action="store_true",
                        help="Auto-capture without a GUI window (auto-enabled when DISPLAY is unset)")
    parser.add_argument("--max-frames", type=int, default=3000,
                        help="Hard stop after this many frames in headless mode")
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

    # Load previously computed individual intrinsics
    intrinsics0 = np.load(args.param0)
    intrinsics1 = np.load(args.param1)
    
    K0, D0 = intrinsics0['camera_matrix'], intrinsics0['dist_coeffs']
    K1, D1 = intrinsics1['camera_matrix'], intrinsics1['dist_coeffs']
    size0 = tuple(intrinsics0['image_size'])

    cap0, _ = open_camera(args.cam0, 1920, 1080, 30, focus_val=args.focus, exposure_val=args.exposure)
    cap1, _ = open_camera(args.cam1, 1920, 1080, 30, focus_val=args.focus, exposure_val=args.exposure)

    board_3d_corners = board.getChessboardCorners()

    obj_points = []       # 3D points in real-world space
    img_points_0 = []     # 2D points in Camera 0
    img_points_1 = []     # 2D points in Camera 1

    print("\n--- Stereo Extrinsic Calibration ---")
    print(f"Board: {spec.label} | {spec.grid[0]}x{spec.grid[1]} squares, "
          f"{spec.square_mm:g} mm tiles")
    print("Position the board in regions where BOTH cameras can clearly see it.")
    if headless:
        print(f"Headless: pairs are captured automatically when the board is still "
              f"in BOTH cameras ({args.settle} still frames). Hold each pose briefly.\n")
    else:
        print("[SPACE]: Save stereo pair  |  [Q]: Finish and run stereo calibration\n")

    grabber = StabilizedGrabber(settle=args.settle, motion_px=args.motion)
    total_frames = 0

    while True:
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1:
            break

        total_frames += 1

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        corners0, ids0, _, _ = detector.detectBoard(gray0)
        corners1, ids1, _, _ = detector.detectBoard(gray1)

        common_ids = np.array([])
        if ids0 is not None and ids1 is not None:
            # Find board corner IDs seen by BOTH cameras in the same instant
            common_ids = np.intersect1d(ids0.flatten(), ids1.flatten())

        common_count = len(common_ids)

        if headless:
            if common_count >= MIN_COMMON:
                det = {}
                for cid in common_ids:
                    idx0 = np.where(ids0.flatten() == cid)[0][0]
                    idx1 = np.where(ids1.flatten() == cid)[0][0]
                    det[(0, int(cid))] = tuple(float(v) for v in corners0[idx0].reshape(-1))
                    det[(1, int(cid))] = tuple(float(v) for v in corners1[idx1].reshape(-1))

                capture = grabber.update(det)
                if capture is not None:
                    cids = sorted(cid for _, cid in capture)
                    pts0 = np.array([capture[(0, cid)] for cid in cids], np.float32)
                    pts1 = np.array([capture[(1, cid)] for cid in cids], np.float32)
                    obj_pts = np.array([board_3d_corners[cid] for cid in cids], np.float32)
                    img_points_0.append(pts0)
                    img_points_1.append(pts1)
                    obj_points.append(obj_pts)
                    print(f"Auto-captured pair {len(obj_points)}/{args.samples} "
                          f"with {len(cids)} matched points.")
                elif total_frames % 30 == 0:
                    if grabber.state == "settling":
                        print(f"[{total_frames:5d}] both cams see board - hold still, "
                              f"{grabber.stable}/{args.settle} still frames")
                    elif grabber.state == "await":
                        print(f"[{total_frames:5d}] pair captured - move to a new pose")
                    else:
                        print(f"[{total_frames:5d}] board not seen by BOTH cameras")
            elif total_frames % 30 == 0:
                print(f"[{total_frames:5d}] board not seen by BOTH cameras")

            if len(obj_points) >= args.samples or total_frames >= args.max_frames:
                break
            continue

        disp0, disp1 = frame0.copy(), frame1.copy()

        if ids0 is not None and ids1 is not None:
            cv2.aruco.drawDetectedCornersCharuco(disp0, corners0.reshape(-1, 1, 2), ids0)
            cv2.aruco.drawDetectedCornersCharuco(disp1, corners1.reshape(-1, 1, 2), ids1)

        color = (0, 255, 0) if common_count >= MIN_COMMON else (0, 0, 255)

        cv2.putText(disp0, f"Common Corners: {common_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(disp1, f"Stereo Pairs Captured: {len(obj_points)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Scale visualization for side-by-side desktop rendering
        preview = np.hstack((cv2.resize(disp0, (960, 540)), cv2.resize(disp1, (960, 540))))
        cv2.imshow("Stereo Stream (Cam 0 | Cam 1)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and common_count >= MIN_COMMON:
            pts0, pts1, obj_pts = [], [], []
            for cid in common_ids:
                idx0 = np.where(ids0.flatten() == cid)[0][0]
                idx1 = np.where(ids1.flatten() == cid)[0][0]
                
                pts0.append(corners0[idx0])
                pts1.append(corners1[idx1])
                obj_pts.append(board_3d_corners[cid])

            img_points_0.append(np.array(pts0, dtype=np.float32))
            img_points_1.append(np.array(pts1, dtype=np.float32))
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            print(f"Recorded stereo frame pair {len(obj_points)} with {common_count} matched points.")
        elif key == ord('q'):
            break

    cap0.release()
    cap1.release()
    if not headless:
        cv2.destroyAllWindows()

    if len(obj_points) < 10:
        print("Error: Insufficient stereo pairs. Capture at least 10–15 pairs.")
        return

    print("Solving for stereo extrinsic baseline and rotation...")
    # CALIB_FIX_INTRINSIC preserves individual camera intrinsics and only computes relative motion
    flags = cv2.CALIB_FIX_INTRINSIC

    ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=obj_points,
        imagePoints1=img_points_0,
        imagePoints2=img_points_1,
        cameraMatrix1=K0, distCoeffs1=D0,
        cameraMatrix2=K1, distCoeffs2=D1,
        imageSize=size0,
        flags=flags
    )

    print(f"\nStereo Calibration Results:")
    print(f"  Stereo Reprojection RMS Error: {ret:.4f} pixels (Ideal: < 0.8)")
    print(f"  Translation Vector T (mm):\n{T.ravel()}")
    print(f"  Calculated Baseline (Distance): {np.linalg.norm(T):.2f} mm")

    np.savez(
        args.output,
        K0=K0, D0=D0,
        K1=K1, D1=D1,
        R=R, T=T, E=E, F=F
    )
    print(f"Saved stereo extrinsic data to {args.output}\n")

if __name__ == "__main__":
    main()
