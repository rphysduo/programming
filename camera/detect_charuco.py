# detect_charuco.py
import argparse
import os
import time
import cv2
import numpy as np
from charuco_common import add_board_argument, get_board, open_camera


def annotate(frame, charuco_corners, charuco_ids, marker_corners, marker_ids,
             image_size):
    """Draw detected markers/corners and live counts onto ``frame`` in place."""
    marker_count = 0 if marker_ids is None else len(marker_ids)
    corner_count = 0 if charuco_ids is None else len(charuco_ids)

    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, list(marker_corners), marker_ids)
    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            frame, charuco_corners.reshape(-1, 1, 2), charuco_ids)

    cv2.putText(frame, f"ArUco Markers: {marker_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"ChArUco Corners: {corner_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Resolution: {image_size[0]}x{image_size[1]}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live ChArUco detection test script")
    parser.add_argument("--camera", default="0", help="Camera index or stream path (e.g., 0, 1)")
    parser.add_argument("--width", type=int, default=1920, help="Target frame width")
    parser.add_argument("--height", type=int, default=1080, help="Target frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate")
    parser.add_argument("--focus", type=int, default=30, help="Manual focus value (0-255)")
    parser.add_argument("--exposure", type=int, default=None,
                        help="Freeze manual exposure (disables auto-exposure/WB)")
    parser.add_argument("--save-annotated", default=None,
                        help="Save the best detected frame with corners/IDs drawn "
                             "to this image path (works headless)")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window; print detection stats (auto-enabled when DISPLAY is unset)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="How long to run in headless mode, in seconds")
    add_board_argument(parser)
    args = parser.parse_args()

    spec = get_board(args.board)
    detector = spec.detector

    # Over SSH (no DISPLAY) the bundled Qt can't create windows, so fall back to headless
    headless = args.headless or not os.environ.get("DISPLAY")

    # Open the stream with manual focus locked
    cap, image_size = open_camera(
        source=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        focus_val=args.focus,
        exposure_val=args.exposure,
    )

    print("\n--- Testing ChArUco Detection ---")
    print(f"Board: {spec.label}")
    print(f"Target resolution: {image_size[0]}x{image_size[1]}")
    if headless:
        print(f"Headless mode: logging stats for {args.duration:.0f}s. Ctrl+C to stop early.")
    else:
        print("Press 'Q' in the window to exit.\n")

    max_markers = 0
    max_corners = 0
    total_frames = 0
    start_time = time.time()
    best_saved_frame = None
    best_saved_corners = -1
    best_ids = None
    best_xy = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers and interpolate ChArUco corners
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        marker_count = 0
        corner_count = 0

        if marker_ids is not None:
            marker_count = len(marker_ids)
        if charuco_ids is not None:
            corner_count = len(charuco_ids)

        max_markers = max(max_markers, marker_count)
        max_corners = max(max_corners, corner_count)
        total_frames += 1

        if corner_count > best_saved_corners:
            best_saved_corners = corner_count
            best_ids = (None if charuco_ids is None
                        else np.asarray(charuco_ids).flatten().astype(int))
            best_xy = (None if charuco_corners is None
                       else np.asarray(charuco_corners).reshape(-1, 2))
            if args.save_annotated:
                best_saved_frame = annotate(
                    frame.copy(), charuco_corners, charuco_ids,
                    marker_corners, marker_ids, image_size)

        if headless:
            if marker_count > 0:
                print(f"markers={marker_count:>2} charuco_corners={corner_count:>2}  "
                      f"best={max_markers}:{max_corners}  frames={total_frames}")
            if time.time() - start_time >= args.duration:
                break
            continue

        annotate(frame, charuco_corners, charuco_ids,
                 marker_corners, marker_ids, image_size)
        cv2.imshow(f"ChArUco Test - Camera {args.camera}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    if args.save_annotated:
        if best_saved_frame is None:
            print("No detections; annotated image not saved.")
        elif cv2.imwrite(args.save_annotated, best_saved_frame):
            print(f"Saved annotated detections to {args.save_annotated}")
        else:
            print(f"WARNING: could not write {args.save_annotated}")

    print(f"\nSummary: {total_frames} frames processed, "
          f"best detection {max_markers} markers / {max_corners} ChArUco corners.")

    if best_ids is not None and best_xy is not None and len(best_ids):
        cols = spec.board.getChessboardSize()[0] - 1  # ChArUco corners per row
        rows_total = spec.board.getChessboardSize()[1] - 1
        rows_seen = len(set((best_ids // cols).tolist()))
        cols_seen = len(set((best_ids % cols).tolist()))
        x0, y0 = best_xy.min(axis=0)
        x1, y1 = best_xy.max(axis=0)
        print(f"Best-frame distribution: {len(best_ids)} corners, "
              f"{rows_seen}/{rows_total} board rows x {cols_seen}/{cols} columns; "
              f"pixel bbox {x1 - x0:.0f} x {y1 - y0:.0f} px "
              f"at ({x0:.0f}, {y0:.0f})-({x1:.0f}, {y1:.0f}).")

if __name__ == "__main__":
    main()
