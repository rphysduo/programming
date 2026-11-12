# Stereo Camera Calibration & Tag Detection

**Capture guards:** see [calibration capture fixes](CALIBRATION_FIXES.md).
New calibration runs require inspected `--roi-file` polygons, or explicit
`--allow-full-frame` for a physically isolated target. Existing NPZ files cannot
be overwritten: choose new candidate outputs. No candidate becomes active
automatically. Diagnose the preserved files before deciding to recalibrate.

A small Python/OpenCV toolbox for calibrating and validating a **two-camera
(stereo) vision rig** built around a long **ChArUco calibration board**. It
covers the whole lifecycle: detect the board, recover each camera's internals,
solve the rig's stereo geometry, then prove the result is correct by measuring
a real object in 3D and comparing it to a tape measure. It ships with live
browser viewers and a browser-based calibration assistant on top of the same
core.

Everything here is a **calibration + measurement toolkit**, not an application:
there is no target-specific downstream model. The output is the set of camera
parameters (the `.npz` files) that a separate measurement/rectification pipeline
can consume.

---

## Recorded-wave inspection and annotation

Use the [offline wave-analysis guide](wave_analysis/README.md) to inspect actual
video timing, review all recorded frames, save versioned surface points and
independent traces, and label breaking events. It includes the November 10
feasibility findings and exact lab commands. The geometry export reuses the
existing calibration snapshots and stereo checks.

## Cross-camera point guide (saved stereo pairs)

Start the updated labeler from `~/programming/camera`, using your existing
camera Python environment; no Node.js or camera hardware is required:

```bash
python validate_stereo_3d.py label --session ../outputs/small_board_check01 --port 5003
```

Open `http://JETSON_IP:5003` on your laptop (or `http://localhost:5003` on the
same computer). Substitute your actual capture-session folder when needed.
The existing session's calibration snapshots remain active; check their hashes
in the banner. This feature does not install a different calibration candidate.

1. Click a recognizable physical feature in **either** camera. A yellow curve
   appears in the other view, with a yellow ring around the source click.
2. Zoom with the wheel and inspect the magnifier. Hover near the actual feature
   to see its distance from the curve in original raw pixels. Click the same
   physical feature to complete the pair, then **Save** for the usual errors.
3. For existing points, **Guide 0→1** or **Guide 1→0** displays that prediction;
   **Edit cam0/cam1** guides the edit from the unchanged opposite-camera click.
4. Only for points on the saved **long-board plane**, enable **Assume saved
   board plane Z=0**. The cyan diamond predicts one pixel using the independent
   saved world poses. Its readout includes the assumed board-plane X, Y and
   Z↑=0. Use the guide on the printed face in its saved calibration position.

The label table shows **Board X, Y, Z↑ (mm)**. X and Y follow the pattern and
Z↑ is positive above the saved printed surface. A flat long board in its saved
pose defines the zero reference over its entire face. Stereo measurements of
that face should be near zero within measurement error. The guide's exact zero
comes from the plane assumption. The table keeps the measured stereo height so
you can see any deviation. If the board sits on the tank floor, set
`board_thickness_mm` in the measurement plan to the measured thickness for
the separate `height_above_tank_floor_mm` output.

A single pixel supplies a viewing ray, not its depth. Its corresponding point
can be anywhere along the epipolar curve; lens distortion makes this curve
non-straight in raw images. The optional diamond adds an explicit plane
assumption. It can disagree with the stereo curve when world and stereo poses
are inconsistent. Both remain visible; neither is a measured correspondence.

**Use the actual feature as the truth.** Clicking an arbitrary location on the
curve can reduce epipolar/reprojection error without improving 3-D accuracy.
For an unbiased calibration check, hide the guide, label identifiable ChArUco
corners independently, then show it to inspect their residuals. A systematic
offset across known matching corners still points to a geometry problem.

All clicks remain raw full-resolution pixels; no snapping, automatic second
click, height clamp, rescaling or validity-threshold changes occur. Guide
requests are read-only, and image-size mismatches are rejected. The cursor's
distance is an approximate shortest distance to the displayed raw-pixel curve;
it is different from the table's Sampson error in ideal undistorted pixels.
Zoom/pan affect display only. The `+176 mm` preview origin does not enter this
calculation. **Clear guide**, Cancel, Escape or changing pairs clears overlays.

Implementation uses `X1 = R @ X0 + T`, `E = [T]x R`, and undistorts the source
pixel before computing `E ray0` or `E.T ray1`; it then reapplies target lens
distortion for display. The plane marker reuses the diagnostic's shared ray/
plane math. See [OpenCV camera geometry](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).

Read-only API: `POST /api/pair/PAIR_NAME/guide`, JSON
`{"source_camera": 0, "source_px": [985.28, 522.36], "board_plane": false}`.
The response contains raw-pixel curve segments, optional board-plane prediction,
and warnings. Run focused regressions with
`python -m pytest camera/tests/test_correspondence_guide.py` from the repository
root, and the hardware-free CLI example with
`python camera/validate_stereo_3d.py selftest`.

---

## Record waves and label video frames

Stop any server using the cameras. In the camera Python environment on the
Jetson, use the same camera order, focus and resolution as calibration. The
following example uses the calibration verified in `validation_flipped_board01`.

```bash
cd ~/programming/camera
CAL=../outputs/validation_flipped_board01/calibration
python record_stereo_waves.py web --calib-dir "$CAL" \
  --cam0 0 --cam1 1 --fps 30 --focus 30 --port 5004
```

Open `http://JETSON_IP:5004` for both live previews. Click **Start recording**
when ready and **Stop recording** to save. There is no timer in browser mode.
The previews stay live between trials. Each **Start recording** creates its own
`outputs/wave_<timestamp>/` session at the repository root.
Copy the **Last saved session** path for `RUN` in the export commands below.
The live views update at a reduced rate and size to limit preview overhead.
The videos use the calibrated resolution and the page reports their observed
frame rates after saving.

Timed capture from the terminal also creates a new session on every run:

```bash
cd ~/programming/camera
CAL=../outputs/validation_flipped_board01/calibration
python record_stereo_waves.py record --calib-dir "$CAL" \
  --cam0 0 --cam1 1 --fps 30 --focus 30 --seconds 15
```

Copy the printed **Saved session** path for `RUN`. To choose a different parent
folder in either mode, pass `--output-dir`. Timed mode also accepts `--session`
for an exact new directory.

In timed mode, start the waves during the recording. The recorder stops
automatically or you can press **Ctrl+C** after both cameras are recording.
Both modes save `cam0.avi` and `cam1.avi` under `$RUN/video/`, with one frame
timestamp CSV per camera. It reports the observed frame rates and preserves a
snapshot of the four calibration files. The default 30 fps matches the C920's
1080p mode. The video files have fixed playback rates, while the CSVs record
actual elapsed host read times.

Export every frame for labeling after stopping the recording:

```bash
python record_stereo_waves.py export --session "$RUN" --all
python validate_stereo_3d.py label --calib-dir "$RUN/calibration" --session "$RUN" --port 5003
```

Open `http://JETSON_IP:5003` and use **Prev** and **Next** to step through
pairs in recording order. Mark the same feature in both images, then press
**Save**. Full export includes every frame from both cameras, using the
nearest host read time for its partner. A frame can appear in multiple
pairs when the streams differ. The export skips pairs already saved, so
earlier labels remain available. Restart the labeler if it was already open.
The full-resolution PNGs in `pairs/` take extra disk space beyond the videos.
If the pictures already exist in the session's `pairs/` directory, run only
the label command with that session path. Save writes the selected points
into `correspondences.json` in the corresponding pair directory.

Use `--at 2 4 6` in place of `--all` to export only specific elapsed times.
The default `--max-gap-ms 18` rejects larger gaps for `--at`. With `--all`,
those pairs remain available for labeling and show a timing warning. The two
C920s expose independently. A small host read-time gap does not establish
simultaneous exposure, and motion can affect triangulation. Check timing with
a shared visible flash before using coordinates for quantitative measurements.

---

## 1. The physical setup

| Thing | Value | Where it is defined |
|---|---|---|
| Cameras | 2× Logitech C920-class USB webcams (`/dev/video0`, `/dev/video1`) | CLI args |
| Capture | 1920×1080 @ 30 fps, MJPEG (Linux/V4L2) | `charuco_common.open_camera` |
| Focus | **Manual, locked** (default `30`) | `--focus` |
| Exposure / WB | Optional manual lock | `--exposure` |
| Board type | ChArUco | `charuco_common.py` |
| ArUco dictionary | `DICT_5X5_250` (both boards) | `charuco_common.py` |
| Long board grid | **4 × 35 squares** (extremely elongated) | `charuco_common.py` |
| Long board square / marker | 44.0 mm / 33.0 mm | `charuco_common.py` |
| Long board marker IDs | `0 … 69` (70 markers) | `charuco_common.BOARDS` |
| Long board ChArUco corners | 102 | derived |
| Long board size (measured) | **144.78 cm × ~17.5 cm** (1447.8 mm × 175 mm) | tape measure |
| Long board corner span (from grid) | ≈ 132 mm × 1496 mm (3×44 by 34×44) | derived |
| Small board grid | **6 × 8 squares** (9 in × 7 in target) | `charuco_common.py` |
| Small board tile / tag | 28.0 mm / 21.0 mm (2.8 cm / 2.1 cm) | `charuco_common.py` |
| Small board marker IDs | `0 … 23` (24 markers) | `charuco_common.BOARDS` |
| Small board ChArUco corners | 35 | derived |
| Small board size (measured) | **228.6 mm × 177.8 mm** (22.86 cm × 17.78 cm) | tape measure |

Both boards live in the `charuco_common.BOARDS` registry and are selected
everywhere with one shared flag, `--board {long,small}` (default `long`), so a
physical board swap is a one-flag change in every script.

### Why a long 4×35 board?

The board is a ~9 mm-thick, 144.78 cm-long (≈1.45 m) strip, not a normal A4
grid. A long strip lets a single capture span the *whole* horizontal field of
view of a camera that is aimed at a long, thin work area, so the corners reach
the image edges where lens distortion is worst and needs the most constraints.
The trade-off is that any single view is geometrically extreme (all points
nearly collinear), which is why the code has extra guards — see
`is_usable_view` below.

### Why a second, small board?

Every view of a 1.45 m strip is close to collinear, so intrinsics fitted from
the strip alone leave the focal length weakly constrained — that is exactly
what the tilt/`fx` warnings in Stage 1 are about. The **small 6×8 board**
(228.6 × 177.8 mm measured, 28 mm squares, 21 mm tags) is the *additional*
intrinsic-calibration target: it is quick to move through genuinely varied
poses, its corners fill a two-dimensional region, and the resulting `K`/`D`
are far better conditioned. The long strip stays the board of choice for
wide-FOV coverage, stereo capture and flatness checks.

```bash
# additional intrinsic calibration with the small board
python calibrate_individual.py --board small --camera 0 \
    --allow-full-frame --output ../outputs/candidates/cam0_intrinsics_small.npz --samples 20
python calibrate_individual.py --board small --camera 1 \
    --allow-full-frame --output ../outputs/candidates/cam1_intrinsics_small.npz --samples 20
```

---

## 2. Core CV concepts used here

- **Intrinsics** (`camera_matrix` = K, `dist_coeffs` = D): the lens+sensor
  model — focal length (`fx`, `fy`), principal point (`cx`, `cy`), and the 5
  radial/tangential distortion coefficients. They depend on the lens, sensor
  size, resolution, and focus setting — **not** on where the camera is
  mounted.
- **Extrinsics** (stereo `R`, `T`): the rigid rotation and translation from
  camera 0's frame to camera 1's. They are *relative* — the rig's absolute
  height or position is irrelevant to them.
- **World extrinsics** (camera → tank): each camera's pose in a fixed external
  frame defined by the long floor board (`tank_alignment.npz`). Unlike stereo
  extrinsics these *do* encode absolute placement, so they change if the rig is
  moved in the tank.
- **Reprojection error (RMS)**: how far detected corners land from where the
  model predicts. The headline quality number, in pixels.
- **Triangulation**: intersecting the two cameras' rays for the same board
  corner to get a 3D point in millimetres. This is what actually gets
  validated.
- **ChArUco**: a chessboard with ArUco markers in the white squares. Markers
  give robust, unambiguous IDs; the chessboard corners in between give
  sub-pixel-accurate points. Best of both.
- **Stereo rectification** (downstream consumer): uses K/D/R/T to warp both
  images so matching points share a row, enabling disparity/depth.

---

## 3. Repository map

| File | Role |
|---|---|
| `charuco_common.py` | **Shared core.** Both board definitions (long strip + small 6×8) and the `--board {long,small}` registry, detectors, `open_camera()` (V4L2/MJPEG, manual focus/exposure), and the `StabilizedGrabber` stillness gate. Import-safe (no side effects). |
| `detect_charuco.py` | Smoke test: live detection counts + overlay; `--save-annotated` writes the best annotated frame, e.g. Stage 4 tank-board evidence. |
| `calibrate_individual.py` | Stage 1 — single-camera **intrinsics**. Headless auto-capture or GUI manual capture; writes `camX_intrinsics.npz`. |
| `calibrate_studio.py` | Stage 2 — **stereo extrinsics** from pairs seen by both cameras. Writes `stereo_params.npz`. |
| `calibrate_world.py` | Stage 2b / 4 — **camera-to-tank alignment** from the long floor board. Also holds the shared `solve_camera_world` / `pack_world_params` helpers used by `web_calibrate.py`'s `world` mode. Writes `tank_alignment.npz`; `--save-images` writes the accepted annotated frames. |
| `test_calibration.py` | Stage 3 — live **validation**: triangulates the board and checks real-world spacing / distances. |
| `charuco_geometry_diagnostic.py` | Automatic dry-board geometry diagnostic: compares known corner coordinates, independent camera-ray/board-plane intersections, unconstrained stereo reconstruction, and world-implied versus saved stereo `R,T`. Runs on saved image pairs without camera hardware. |
| `validate_stereo_3d.py` | Dry 3-D **metrology validation**: browser-based full-resolution stereo capture and correspondence labelling, then triangulation against measured truth. Never edits the calibration files. |
| `check_flatness.py` | Diagnostic — per-corner reprojection residue, to catch a **bowed/wrinkled/printed-bad board**. |
| `web_tags.py` | Live dual-camera browser stream with ChArUco/ArUco highlighting. Also hosts shared stream helpers (`LiveView`, `CameraWorker`, `annotate_frame`, `encode_jpeg`). |
| `web_calibrate.py` | Browser **calibration assistant** (manual capture + solve) for both modes; writes the same `.npz` files. |
| `web_stream.py` | Legacy simple dual 640×480 YUYV MJPEG stream (Jetson-era). Superseded by `web_tags.py`. |
| `cam0_intrinsics.npz`, `cam1_intrinsics.npz` | Saved intrinsics. |
| `stereo_params.npz` | Saved stereo extrinsics. |
| `tank_alignment.npz` | Saved board → camera and board → tank alignment (Stage 4). |

### Dependency direction

```
charuco_common.py  ──►  calibrate_individual.py ─┐
       ▲                     ▲                   ├─► web_calibrate.py
       │                     └─ MIN_CORNERS ─────┤
       ├── calibrate_studio.py ── MIN_COMMON ────┤
       ├── calibrate_world.py ──────────► web_calibrate.py (world mode)
       ├── detect_charuco.py
       ├── check_flatness.py
       ├── test_calibration.py
       ├── charuco_geometry_diagnostic.py
       └── web_tags.py ──────► web_calibrate.py
```

`web_calibrate.py` reuses the exact capture/solve logic and defined thresholds
from the CLI scripts, so the CLI and browser paths produce identical results.

---

## 4. The calibration pipeline

```
        ┌─────────────────────┐
        │ detect_charuco.py   │  (optional smoke test)
        └─────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ calibrate_individual │  per camera  →  cam0_intrinsics.npz
        │  (intrinsics, K / D) │               cam1_intrinsics.npz
        └──────────┬──────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
┌───────▼─────────┐   ┌────────▼──────────────┐
│ calibrate_studio│   │ calibrate_world       │  both cams → tank_alignment.npz
│ (R / T cams)    │   │ (camera poses in tank)│
│ stereo_params   │   └───────────────────────┘
└───────┬─────────┘
        │
┌───────▼─────────┐
│ test_calibration│  validate against a tape measure
└─────────────────┘
```

### Stage 1 — Intrinsics (`calibrate_individual.py`)

Run once per camera. Move the board through ~20 varied poses so it reaches the
image corners and is **tilted out of the image plane** (tilt is what constrains
focal length). Captures happen automatically in headless mode after the board is
still, or on `SPACE` in GUI mode. Pass `--board small` to calibrate against the
small 6×8 board instead — recommended for the additional, better-conditioned
intrinsics run (see §1).

- `MIN_CORNERS = 10` valid corners per view.
- `is_usable_view()` rejects degenerate views: the board must span ≥2 board
  rows *and* ≥2 board columns. (The board is so elongated that its eigen-value
  ratio is ~0.007 even when fully visible, so testing raw elongation directly
  would reject every good capture — the code tests genuine degeneracy instead.)
- Solves with `cv2.calibrateCamera` and prints a **pose-diversity report**:
  out-of-plane tilt stats and board distance range.
- Warnings if RMS > 1.0 px, if `fx` implies an implausible FOV, or if
  `fx/fy` deviates from 1.0 (square-pixel check).

### Stage 2 — Stereo extrinsics (`calibrate_studio.py`)

Loads the two intrinsics files, captures poses where **both** cameras see the
board (≥ `MIN_COMMON = 10` shared corners), then solves a rigid transform with
`cv2.stereoCalibrate` using `cv2.CALIB_FIX_INTRINSIC` — the individual
intrinsics are trusted and only `R`, `T`, `E`, `F` are computed.

### Stage 2b / 4 — Camera-to-tank alignment (`calibrate_world.py`)

Answers *"where is each camera relative to the wave tank?"* using the
**original long 4×35 floor board** (44 mm squares, 33 mm markers; exposed as
`charuco_common.TANK_BOARD`, deliberately separate from the small intrinsic
board), which is laid in the tank to define the tank's physical coordinate
frame. Unlike stereo, the cameras do **not** need to see the board at the same
instant: the board is a fixed landmark, so each camera is solved independently
with `cv2.solvePnP(..., SOLVEPNP_IPPE)` using the saved intrinsics (a planar
target, ≥ `MIN_WORLD_CORNERS = 6` corners), and only the tank frame — never the
board pose — must stay fixed between captures. Each valid picture is kept and
pooled into one per-camera solve (10 pictures per camera by default, set with
`--samples`); repeated or near-identical poses are **not** rejected, so no
picture is dropped for being too similar. The board frame is the one
returned by `getChessboardCorners()` (`+x` short edge, `+y` long edge, `+z` out
of the face); `--tank-t` / `--tank-rpy` (aliases `--world-t` / `--world-rpy`)
place it inside the tank frame. `--tank-rpy` takes fixed-axis XYZ (roll, pitch,
yaw) in degrees, composed as `Rz @ Ry @ Rx`. Defaults assume the board frame
*is* the tank frame (`x` across, `y` along, `z` up).

Output `tank_alignment.npz` contains both `board → camera` poses
(`R_board_to_cam0/1`, `t_board_to_cam0/1`) and the measured `board → tank`
placement (`R_board_to_tank`, `t_board_to_tank`), plus the derived camera
centres in tank millimetres and a baseline cross-check against
`stereo_params.npz`. Because the stereo calibration may still be on provisional
metric scale, `stereo_scale` is recorded in the file (`--stereo-scale
provisional|validated`, default `provisional`); it does **not** affect the
per-camera solve, which scales from the 44 mm board.

Use `--save-images` to write the accepted annotated frame for each camera
(`<output>_cam0.png`, `<output>_cam1.png`) as detection evidence. For the
pre-alignment detection check, `detect_charuco.py --board long
--save-annotated ...` also reports the best frame's ChArUco row/column coverage
and pixel bounding box.

### Stage 3 — Validation (`test_calibration.py`)

The most important stage. It streams both cameras, triangulates the board every
frame, and reports:

- `solvePnP dist` — distance from each camera to the board (single-camera view),
- `stereo dist` — distance from each camera after triangulation,
- **`spacing err`** — how far the camera-derived square size is from reality
  (44 mm on the long board, 28 mm on the small one; a direct accuracy check,
  `GOOD ≤ 3%`, `FAIR ≤ 8%`, else `POOR`),
- reprojection RMS per camera.

The printed instruction is the ground truth check: **put a tape measure from
each lens to the board centre and compare.** If reported distance and spacing
match reality, the calibration is trustworthy.

---

## 5. Running it

Activate the virtualenv first (`.venv/` is present):

```bash
source .venv/bin/activate
```

### Recommended order

The three things you need, the board each one uses, and the command:

| Task | Board | Browser command |
|---|---|---|
| **Intrinsics** of each C920 (K / D) | `small` (easy to tilt) | `--mode individual --board small --camera 0` then `--camera 1` |
| **Cameras relative to each other** (R / T) | `small`, visible to both | `--mode stereo --board small --cam0 0 --cam1 1` |
| **Cameras relative to the tank** | `long` floor board | `--mode world --board long --cam0 0 --cam1 1` |

```bash
# 0. Sanity: does detection work at all?
python detect_charuco.py --camera 0

# 1. Intrinsics, once per camera, using the SMALL board
python calibrate_individual.py --board small --camera 0 --allow-full-frame --output ../outputs/candidates/cam0_intrinsics.npz --samples 20
python calibrate_individual.py --board small --camera 1 --allow-full-frame --output ../outputs/candidates/cam1_intrinsics.npz --samples 20

# 2. Stereo extrinsics (both cameras see the SMALL board)
python calibrate_studio.py --board small --cam0 0 --cam1 1 --allow-full-frame --samples 15 \
  --param0 ../outputs/candidates/cam0_intrinsics.npz --param1 ../outputs/candidates/cam1_intrinsics.npz \
  --output ../outputs/candidates/stereo_params.npz

# 2b/4a. Big-board detection evidence, once per camera (long 4x35 board)
python detect_charuco.py --board long --camera 0 --save-annotated tank_cam0.png
python detect_charuco.py --board long --camera 1 --save-annotated tank_cam1.png

# 2b/4b. Camera poses in the tank frame, from the long floor board
python calibrate_world.py --board long --cam0 0 --cam1 1 --save-images \
  --roi-file direct_board_roi.json --output ../outputs/candidates/tank_alignment.npz \
  --param0 ../outputs/candidates/cam0_intrinsics.npz --param1 ../outputs/candidates/cam1_intrinsics.npz \
  --stereo ../outputs/candidates/stereo_params.npz

# 3. Validate against a tape measure
python test_calibration.py --cam0 0 --cam1 1

# 4. Diagnose all three geometric estimates on a saved LONG-board pair
python charuco_geometry_diagnostic.py --board long \
  --pair-dir validation/pairs/pair_0018_20261029_142637 \
  --roi-file diagnostics/pair_0018_direct_board_roi.json \
  --output-dir ../outputs/charuco_geometry_diagnostic

# Optional: is the printed board actually flat?
python check_flatness.py --camera 0 --param cam0_intrinsics.npz
```

> Only one process can hold a V4L2 camera at a time — never run two scripts
> (or a script + the browser tools) against the same device simultaneously.

### Automatic board-geometry diagnostic

See [the diagnostic findings and lab procedure](diagnostics/README.md) for the
staged investigation, current calibration hashes, raw-pixel ROI setup, exact
capture/reproduction commands, and independent ruler/height follow-up.
Reports now include signed residuals, invalid points, annotated raw images,
and optional headless spatial/3D plots. Use a fresh output directory each run.

`charuco_geometry_diagnostic.py` treats every reliably detected long-board
corner as a known 3-D reference point. For 10–20 spatially distributed corner
IDs it reports three estimates without conflating them:

1. the exact coordinate from `CharucoBoard.getChessboardCorners()`;
2. each camera ray independently intersected with the known board plane
   `Z=0`; and
3. unconstrained stereo triangulation transformed into board coordinates.

Homography RANSAC must accept a corner in both images before it is analyzed.
This rejects isolated glare/reflection detections and interpolation outliers;
ambiguous duplicate IDs, which can indicate a reflected copy, are excluded.
Homography filtering alone cannot reject a coherent reflection. The CLI now
requires a user-inspected direct-board polygon for each camera, applied before
decoding while retaining raw pixel coordinates. Full-frame reconstruction is
available only with explicit `--allow-unverified-full-frame`; use
`--preview-only` to inspect detections first.
The stereo result is **not** projected back onto `Z=0`, because its reconstructed
height error is one of the measurements the diagnostic is intended to expose.

The report also derives camera-0-to-camera-1 `R,T` from the two independently
saved board poses and compares the full transform with `stereo_params.npz`.
Its compact diagnosis has three possible values:

- `geometry_consistent`: all three estimates and the relative transforms agree;
- `stereo_geometry_mismatch`: individual plane intersections agree with the
  board but free stereo reconstruction or stereo `R,T` does not; or
- `camera_pose_intrinsics_detection_or_board_mismatch`: at least one
  independent camera-to-board projection is already inaccurate.

JSON and CSV output are written only when `--output-dir` is supplied. The
default 5 mm / 1 degree classification limits are explicit diagnostic
demonstration values, not experimentally calibrated accuracy claims.
When `pair.json` is present, the report also compares its recorded calibration
hashes with the files used for analysis and warns if a newer or different
calibration generation is being applied.

### Browser tools

```bash
# Live tag viewer: both cameras + seen/missing marker chips
python web_tags.py --cam0 0 --cam1 1          # http://<host>:5000/
python web_tags.py --board small              # track the small 6x8 board instead

# Calibration assistant: capture poses in the browser, solve in-browser
# Step A - intrinsics, once per camera, with the SMALL board
python web_calibrate.py --mode individual --board small --camera 0 --allow-full-frame --output ../outputs/candidates/web_cam0.npz
python web_calibrate.py --mode individual --board small --camera 1 --allow-full-frame --output ../outputs/candidates/web_cam1.npz

# Step B - cameras relative to each other, SMALL board seen by both
python web_calibrate.py --mode stereo --board small --cam0 0 --cam1 1 \
    --allow-full-frame --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz \
    --output ../outputs/candidates/web_stereo.npz

# Step C - cameras relative to the tank, LONG floor board
#   (place the board in the tank; add --tank-t / --tank-rpy if the board
#    frame is not already the tank frame; the identity default is a placeholder)
#   10 pictures are captured/kept per camera by default (--samples 10)
python web_calibrate.py --mode world --board long --cam0 0 --cam1 1 \
    --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz \
    --roi-file direct_board_roi.json --tank-output ../outputs/candidates/web_tank.npz
```

The web UI defaults to port **5001** (`--port 5001`), so open
`http://<host>:5001/`. In `web_calibrate.py` press **Capture Now** to store each
pose and **Finish & Calibrate** to solve. World mode requires the requested
count from each camera before it will save. It writes the same NPZ schema as
the CLI to a new candidate path; existing files are never replaced.

The dry-validation tool has its own browser UIs (ports **5002** for capture and
**5003** for labelling):

```bash
# Capture full-resolution stationary pairs; open the printed URL
python validate_stereo_3d.py capture

# Label correspondences in saved pairs; no cameras are opened
python validate_stereo_3d.py label
```

In the capture page, press **Capture Pair** for each stationary pose. The live
previews show an orange crosshair at a preview origin plus 50 mm board axes:
`+X` red, `+Y` green, and upward `+Z` blue. By default the preview
origin is shifted `+4` board rows (`+176 mm`) along `+Y`; use
`--preview-origin-y-rows 0` to show the true pattern origin. This uses the
saved board pose in `tank_alignment.npz`, is drawn only on the preview stream,
and is not written into the saved PNGs. Saved board coordinates always use the
true `(0,0,0)` pattern origin. In the labelling page, click a point in cam0
then the same point in cam1; use the
mouse wheel to zoom, right/shift-drag to pan, and the point table to edit or
delete. **Save** writes raw full-resolution pixel coordinates and also saves
derived geometry for each point:

* `pixel_xyz`: x, y in cam0 raw pixels; z is the horizontal disparity
  `cam0.x - cam1.x` in pixels (not metric depth),
* `X_cam0_mm` and `X_board_mm`: triangulated Camera-0 and big-board coordinates
  in millimetres,
* `height_above_board_mm`, displayed as Z↑, plus validity flags,
  reprojection/epipolar errors and parallax.

The derived fields are recomputed on every save; the raw `cam0`/`cam1` pixels
remain authoritative. Opening an older pair computes them for display, and
pressing **Save** writes them into that pair's `correspondences.json`.
The label-page banner now displays the loaded snapshot's baseline and warns
when available source calibration files differ. Expand it for paths and hashes;
use a new session to validate newer calibration files.

The **Results** button opens the browser results viewer at
`http://<host>:5003/results`. It shows tolerance status, measurement tables,
3-D point coordinates, statistics, diagnostics, the reconstruction plot,
annotated pairs and downloadable JSON/CSV/summary files.

The **Compare** button opens `http://<host>:5003/compare`. It lists the
triangulated board-frame predictions for every labeled pair/point and lets you
enter measured actual X/Y/Z values in millimetres. Actuals are keyed by point
name (so a point `A` shared by repeated captures has one truth value) and are
saved as position tests in `measurement_plan.json`; the page then computes the
error vector and `|error|` live. The page also plots the board origin
`(0,0,0)`, the `+X`, `+Y` and physical-up axes, predicted points, entered
actuals and predicted-to-actual error lines on one equal-scale graph. Enter
actuals in the saved pattern frame used by `truth_board_mm`. The separate Z↑
column shows height above the printed board face. Keep the target stationary
during a capture because the two C920s capture independently.

---

## 6. Output file schemas

**`cam0_intrinsics.npz` / `cam1_intrinsics.npz`**

| Key | Shape | Meaning |
|---|---|---|
| `camera_matrix` | (3, 3) | K = `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `dist_coeffs` | (1, 5) | `[k1, k2, p1, p2, k3]` |
| `image_size` | (2,) | `[width, height]` the intrinsics are valid at |

**`stereo_params.npz`**

| Key | Shape | Meaning |
|---|---|---|
| `K0`, `D0` | (3,3), (1,5) | cam0 intrinsics (echoed) |
| `K1`, `D1` | (3,3), (1,5) | cam1 intrinsics (echoed) |
| `R` | (3,3) | rotation cam0 → cam1 |
| `T` | (3,1) | translation cam0 → cam1, **mm** (`‖T‖` = baseline) |
| `E` | (3,3) | essential matrix |
| `F` | (3,3) | fundamental matrix |

**`tank_alignment.npz`** (board → cameras + board → tank)

| Key | Shape | Meaning |
|---|---|---|
| `R_board_to_cam0`, `t_board_to_cam0` | (3,3), (3,1) | `X_cam0 = R @ X_board + t` from `solvePnP` |
| `R_board_to_cam1`, `t_board_to_cam1` | (3,3), (3,1) | same for cam1 |
| `R_board_to_tank`, `t_board_to_tank` | (3,3), (3,1) | measured board placement: `X_tank = R @ X_board + t` |
| `square_length_mm`, `marker_length_mm` | (,) | board geometry used (44.0, 33.0) |
| `board_size` | (2,) | `[squaresX, squaresY]` = `[4, 35]` |
| `T_world_board` | (4,4) | board→tank transform used (`--tank-t` / `--tank-rpy`) |
| `stereo_scale` | (,) | `provisional` or `validated` |
| `stereo_scale_provisional` | (,) | boolean convenience flag |
| `R_cam0_world`, `t_cam0_world` | (3,3), (3,1) | derived `X_cam0 = R @ X_world + t` |
| `cam0_center_world` | (3,) | cam0 centre in the tank frame, **mm** |
| `R_cam1_world`, `t_cam1_world`, `cam1_center_world` | | same for cam1 |
| `cam0_frames`, `cam1_frames` | (,) | number of pictures combined per camera |
| `board` | (,) | board name (`long`) |
| `image_size` | (2,) | `[width, height]` |

---

## 7. Design decisions & invariants

These are the non-obvious choices baked into the code. Breaking them silently
degrades accuracy.

1. **Everything is locked at capture time.** Autofocus off, manual focus,
   optional manual exposure + white balance. Auto-gain/exposure pulsing moves
   every detected corner uniformly and looks exactly like board motion, which
   also breaks the stillness gate. (`charuco_common.open_camera`)

2. **`StabilizedGrabber` is a stillness gate + sub-pixel averager**
   (`charuco_common.py`). It captures only after the board is still for
   `settle = 6` consecutive frames (median corner motion ≤ `motion_px = 1.5`
   px), averages the last ≤16 frames to cancel detection jitter, and refuses to
   re-capture the *same* pose (`new_pose_px = 4 × motion`). Comparing corner
   **positions** (not just which tag IDs are visible) is what lets a board be
   tilted in place and still count as a new pose — otherwise only in-plane
   translations are seen and focal length stays unconstrained.

3. **Resolution must match everywhere.** `fx`, `fy`, `cx`, `cy` scale with
   pixels; `image_size` is stored and reused by `stereoCalibrate`. Calibrate
   and run at the same width/height.

4. **Focus and working distance must match operation.** Focus changes the
   effective focal length and distortion, so calibrate with the `--focus` value
   you will run with, at roughly the distance you will run at. Distortion
   extrapolates badly: a calibration done at 0.5 m is not accurate at 2 m.

5. **Mounting height is irrelevant; rig rigidity is not.** Intrinsics don't
   depend on height, and stereo `R`/`T` are relative between the cameras, so the
   rig can be deployed at any height — as long as the two cameras do not move
   relative to each other after calibration. Any bump invalidates `R`/`T`;
   re-run Stage 2 (and Stage 3 to confirm).

6. **Trust `CALIB_FIX_INTRINSIC`.** Stereo only solves geometry; intrinsics come
   from Stage 1. That is why `stereo_params.npz` should echo the same `K0/K1`
   as the individual files.

7. **The measured board overrides the grid maths, and `--board` picks it.**
   `charuco_common.BOARDS` holds each board's `square_mm` and its measured
   `size_mm`; every millimetre the pipeline reports is scaled by `square_mm`, so
   the board named on the command line must be the board physically in view —
   and `square_mm` must match the printed tiles, not the nominal layout.

8. **World mode only needs the *board* to be fixed, not both cameras.**
   `calibrate_world.py`/`--mode world` solve each camera independently against
   the same floor-board pose, so a camera may be solved while the other cannot
   see the board. All valid pictures are kept (default 10 per camera) and
   combined into one solve; there is deliberately no pose-uniqueness check, so
   pressing *Capture Now* repeatedly never discards a usable picture. What must
   not move between captures is the board — it *is* the
   tank frame. If the board frame is not the tank frame, encode the offset once
   with `--tank-t` (mm) / `--tank-rpy` (deg) and it is stored as
   `T_world_board`. The camera-to-camera baseline it reports should match the
   `‖T‖` in `stereo_params.npz`; a mismatch means the board moved or a camera
   moved. Until that baseline and the 3-D checks are validated, keep
   `stereo_scale = provisional` in `tank_alignment.npz`.

---

## 8. Diagnostics & troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Very few false negatives / missed corners | Lighting, focus, or motion blur | Improve lighting, set `--focus`, hold still |
| `WARNING: RMS high` in Stage 1 | Under-constrained solve | Tilt the board 20–45° about **both** axes, many poses |
| `WARNING: implausible fx` | Too little out-of-plane rotation | More tilted poses; ensure the board fills the frame |
| `WARNING: fx/fy` far from 1.0 | Bad detection / degenerate board | Re-check board printing and detection |
| `spacing err` POOR in Stage 3 | Calibration drifted or board moved | Recalibrate; confirm focus and resolution unchanged |
| Localized residue on a few corners (`check_flatness.py`) | Board is bowed/wrinkled/printed wrong | Flatten or reprint the board — **cannot be calibrated away** |
| Uniformly high residue (flatness) | Focus/sharpness/optimisation, not flatness | Fix lighting/focus |

`check_flatness.py` distinguishes **localized** error (physical board defect,
recurring on the same corners across poses) from **uniform** error (sharpness
or solver precision) — a defect you must fix physically versus one you can tune.

### Current artifact state (observed)

A quick inspection of the checked-in `.npz` files raises flags worth resolving:

- `cam0_intrinsics.npz` has `fx ≈ 245`, `fy ≈ 335`, and
  `cam1_intrinsics.npz` has `fx ≈ 283`, `fy ≈ 355` — all far below the
  plausible range for a 1920-wide image (`test_calibration.plausible()` expects
  roughly `0.6×`–`1.6×` width, and warns explicitly). These look like a stale,
  degenerate, or wrong-resolution calibration.
- `cam1`'s distortion coefficients are essentially all zero.
- `stereo_params.npz` reports `K0/K1` (`fx ≈ 354` / `541`) that **differ** from
  the individual intrinsics files, which should not happen under
  `CALIB_FIX_INTRINSIC`.

Treat the saved parameters as suspect and re-run Stages 1–3 (and validate) before
relying on them.

---

## 9. Dependencies & environment

- Python 3 (repo has a `.venv/`)
- `opencv-python` / `opencv-contrib-python` with `aruco` (ChArUco), currently
  OpenCV **5.0.0** in this environment
- `numpy`
- `flask` (browser tools only), currently 3.1.x
- Linux + V4L2 cameras (MJPEG path); code falls back to a generic
  `cv2.VideoCapture` elsewhere
- GUI scripts fall back to **headless** automatically when `DISPLAY` is unset
  (e.g. over SSH), because the bundled Qt cannot create windows

There is no `requirements.txt` yet.

---

## 10. Glossary

- **K / camera_matrix** — intrinsic matrix (focal length + principal point).
- **D / dist_coeffs** — lens distortion coefficients.
- **R / T** — stereo rotation and translation (baseline) between the cameras.
- **RMS** — reprojection error, in pixels; lower is better.
- **spacing err** — % error between a triangulated 44 mm square and its real
  size; the end-to-end accuracy check.
- **Baseline** — distance between the two camera centres (`‖T‖`), in mm.
- **Stillness gate** — capture only when the board has stopped moving, to avoid
  motion blur and cross-camera timing mismatch.
- **ChArUco corner** — the sub-pixel chessboard intersection used as a
  measurement point.
