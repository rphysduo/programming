import sys
from collections import namedtuple

import cv2
import numpy as np

# ============================================================
# CHARUCO BOARD CONFIGURATION
# ============================================================

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Physical board size, measured on the real strip: 144.78 cm x ~17.5 cm
# (1447.8 mm x 175 mm).  The BOARD definition below describes the printed
# ChArUco pattern (4 x 35 squares at 44 mm each), not the strip's outer size.
PHYSICAL_BOARD_MM = (1447.8, 175.0)   # (length, width) in mm

BOARD = cv2.aruco.CharucoBoard(
    (4, 35),   # squaresX, squaresY
    44.0,      # squareLength (mm)
    33.0,      # markerLength (mm)
    DICTIONARY
)

# Set to True if the board was printed using OpenCV < 4.6
BOARD.setLegacyPattern(False)

DETECTOR = cv2.aruco.CharucoDetector(BOARD)

# ============================================================
# SMALL BOARD (6 x 8) -- ADDITIONAL INTRINSIC CALIBRATION
# ============================================================
#
# A conventional 9 x 7 in ChArUco target, used as a SECOND target for
# intrinsic calibration.  Every view of the long strip is nearly collinear,
# which leaves the focal length weakly constrained; a 6x8 board gives
# genuinely two-dimensional coverage instead, so intrinsics calibrated from
# it are much better conditioned.
#
# Physical size of the printed board: 228.6 mm x 177.8 mm (22.86 cm x 17.78 cm,
# i.e. a 9 in x 7 in target), with 28 mm squares and 21 mm tags.
SMALL_SQUARE_MM = 28.0       # square (tile) side length, mm
SMALL_MARKER_MM = 21.0       # ArUco/AprilTag marker side length, mm
PHYSICAL_SMALL_BOARD_MM = (228.6, 177.8)   # measured (long side, short side), mm

SMALL_BOARD = cv2.aruco.CharucoBoard(
    (6, 8),             # squaresX, squaresY
    SMALL_SQUARE_MM,    # squareLength (mm)
    SMALL_MARKER_MM,    # markerLength (mm)
    DICTIONARY
)

SMALL_BOARD.setLegacyPattern(False)

SMALL_DETECTOR = cv2.aruco.CharucoDetector(SMALL_BOARD)

# The small board is the intrinsic-calibration board; keep the names used by
# the calibration scripts.
INTRINSIC_BOARD = SMALL_BOARD
INTRINSIC_DETECTOR = SMALL_DETECTOR

# The original long 4x35 board (44 mm squares, 33 mm markers) is the tank
# landmark for Stage 4 alignment.  These explicit aliases keep it separate from
# the small 6x8 intrinsic board -- do not reuse the small detector for the tank.
TANK_BOARD = BOARD
TANK_DETECTOR = DETECTOR


# ============================================================
# BOARD REGISTRY  (shared "--board {long,small}" CLI flag)
# ============================================================

#: Everything a script needs to know about one physical ChArUco board.
#: ``size_mm`` is the *measured* physical size of the printed board, which is
#: authoritative over the pattern-derived ``grid x square_mm`` product.
BoardSpec = namedtuple(
    "BoardSpec",
    "name board detector square_mm marker_mm size_mm grid label",
)

BOARDS = {
    "long": BoardSpec(
        "long", BOARD, DETECTOR, 44.0, 33.0, PHYSICAL_BOARD_MM,
        BOARD.getChessboardSize(),
        "long 4x35 strip (144.78 cm x ~17.5 cm)",
    ),
    "small": BoardSpec(
        "small", INTRINSIC_BOARD, INTRINSIC_DETECTOR,
        SMALL_SQUARE_MM, SMALL_MARKER_MM, PHYSICAL_SMALL_BOARD_MM,
        INTRINSIC_BOARD.getChessboardSize(),
        "small 6x8 board (22.86 cm x 17.78 cm)",
    ),
}


def get_board(name):
    """Return the :class:`BoardSpec` for ``name`` (``'long'`` or ``'small'``)."""
    try:
        return BOARDS[name]
    except KeyError:
        raise ValueError(f"unknown board {name!r}; expected one of {sorted(BOARDS)}") from None


def add_board_argument(parser, default="long"):
    """Add the shared ``--board {long,small}`` flag to a script's parser.

    Returns ``parser`` so it can be chained.  Every board-using script gets the
    same flag so a physical board swap is a one-flag change everywhere.
    """
    parser.add_argument(
        "--board", default=default, choices=sorted(BOARDS),
        help="Physical ChArUco board in view: 'long' = 4x35 strip "
             "(44mm squares), 'small' = 6x8 board, 22.86x17.78cm "
             "(28mm squares, 21mm tags). Default: %(default)s")
    return parser


def board_ids(board):
    """Marker IDs printed on ``board``, as a sorted list of Python ints."""
    return sorted(int(i) for i in np.asarray(board.getIds()).flatten())


# ============================================================
# STABILIZED AUTO-CAPTURE
# ============================================================

class StabilizedGrabber:
    """Stillness gate + subpixel averaging for auto-captured calibration frames.

    Feed a per-frame dict of detected points (hashable key -> (x, y)) and this
    returns an averaged capture only when the board has been *still* for ``settle``
    consecutive frames AND the pose changed since the last capture.  Averaging cancels
    per-corner subpixel detection jitter; the stillness gate rejects motion-blurred
    frames and, in stereo, frames where the two unsynchronized cameras caught the
    board at different positions.
    """

    def __init__(self, settle=6, motion_px=1.5, max_run=16, overlap=0.8, new_pose_px=None):
        self.settle = settle
        self.motion_px = motion_px
        self.max_run = max_run
        self.overlap = overlap
        # A *new* pose must be displaced from the last captured one by more than
        # this (px).  Comparing corner POSITIONS -- not just the set of visible
        # tag ids -- is what lets a board be tilted in place and still register
        # as a new pose.  Without it, calibration only ever sees translations of
        # the board in its own plane, which leaves the focal length unconstrained.
        self.new_pose_px = motion_px * 4 if new_pose_px is None else new_pose_px
        self._sig = None
        self._prev = None
        self._run = []
        self._await = False
        self._done = None

    def reset(self):
        self._sig = self._prev = self._done = None
        self._run = []
        self._await = False

    @property
    def state(self):
        """'idle' (no pose yet), 'settling' (accumulating still frames), or 'await'
        (this pose already captured, waiting for a new one)."""
        if self._await:
            return "await"
        if self._sig is not None:
            return "settling"
        return "idle"

    @property
    def stable(self):
        return len(self._run)

    def update(self, det):
        """Feed one detection dict. Returns an averaged {key: (x, y)} on capture."""
        sig = tuple(sorted(det))
        if not sig:
            return None

        if self._await:
            if (self._sig is None
                    or not self._same_pose(sig, self._sig)
                    or self._moved(self._done, det, self.new_pose_px)):
                self._reset(sig, det)
            return None

        if self._sig is None:
            self._reset(sig, det)
            return None

        same_pose = self._same_pose(sig, self._sig)
        moved = 0.0
        if same_pose and self._prev is not None:
            common = set(det) & set(self._prev)
            if len(common) >= 4:
                moved = float(np.median([
                    np.hypot(det[k][0] - self._prev[k][0], det[k][1] - self._prev[k][1])
                    for k in common
                ]))

        if not same_pose or moved > self.motion_px:
            self._reset(sig, det)
            return None

        self._run.append(det)
        if len(self._run) > self.max_run:
            self._run.pop(0)
        self._prev = det

        if len(self._run) >= self.settle:
            avg = self._average()
            self._done = avg
            self._await = True
            return avg
        return None

    def _same_pose(self, a, b):
        """True if the two id-sets are mostly the same pose. Tolerates a corner
        flickering in/out of detection between frames."""
        A, B = set(a), set(b)
        if not A or not B:
            return False
        return len(A & B) >= min(len(A), len(B)) * self.overlap

    def _moved(self, ref, det, thr):
        """True if ``det`` is displaced from reference pose ``ref`` by more than
        ``thr`` px (median over shared points).  Too little overlap between the
        two also counts as a new pose."""
        if not ref:
            return False
        common = set(det) & set(ref)
        if len(common) < 4:
            return True
        d = float(np.median([
            np.hypot(det[k][0] - ref[k][0], det[k][1] - ref[k][1])
            for k in common
        ]))
        return d > thr

    def _reset(self, sig, det):
        self._sig = sig
        self._prev = det
        self._run = [det]
        self._await = False

    def _average(self):
        """Average every key that was detected in a majority of the run's frames,
        so a corner flickering out for a frame or two doesn't get dropped."""
        if not self._run:
            return None
        half = max(len(self._run) // 2, 1)
        counts = {}
        for frame in self._run:
            for k in frame:
                counts[k] = counts.get(k, 0) + 1
        keep = sorted(k for k, seen in counts.items() if seen >= half)
        if not keep:
            return None
        acc = {}
        for k in keep:
            x = y = n = 0.0
            for frame in self._run:
                if k in frame:
                    x += frame[k][0]
                    y += frame[k][1]
                    n += 1
            acc[k] = (x / n, y / n)
        return acc


# ============================================================
# CAMERA STREAM INITIALIZATION
# ============================================================

def parse_camera_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def open_camera(source, width=1920, height=1080, fps=30, focus_val=30, exposure_val=None):
    source = parse_camera_source(source)

    if sys.platform.startswith("linux") and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        # Use MJPEG codec for higher framerates on Logitech C920
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source)

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Disable C920 hardware autofocus to prevent calibration drift
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if focus_val is not None:
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)

    # Freeze exposure and white balance too: auto gain/exposure pulsing shifts every
    # corner's detected position uniformly and looks exactly like board motion.
    if exposure_val is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on V4L2
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera stream for source: {source}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Camera {source} opened, but failed to grab frame.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Device {source} initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS | Focus: Manual ({focus_val})")
    return cap, (actual_w, actual_h)
