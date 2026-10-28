# Stereo Camera Calibration & Tag Detection

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
    --output cam0_intrinsics_small.npz --samples 20
python calibrate_individual.py --board small --camera 1 \
    --output cam1_intrinsics_small.npz --samples 20
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

# 1. Intrinsics, once per camera (headless auto-capture; hold each pose still)
python calibrate_individual.py --camera 0 --output cam0_intrinsics.npz --samples 20
python calibrate_individual.py --camera 1 --output cam1_intrinsics.npz --samples 20

# 1b. Additional intrinsics with the small 6x8 board (better-conditioned K/D)
python calibrate_individual.py --board small --camera 0 --output cam0_intrinsics_small.npz
python calibrate_individual.py --board small --camera 1 --output cam1_intrinsics_small.npz

# 2. Stereo extrinsics (both cameras see the board)
python calibrate_studio.py --cam0 0 --cam1 1 --samples 15

# 2b/4a. Big-board detection evidence, once per camera (long 4x35 board)
python detect_charuco.py --board long --camera 0 --save-annotated tank_cam0.png
python detect_charuco.py --board long --camera 1 --save-annotated tank_cam1.png

# 2b/4b. Camera poses in the tank frame, from the long floor board
python calibrate_world.py --board long --cam0 0 --cam1 1 --save-images

# 3. Validate against a tape measure
python test_calibration.py --cam0 0 --cam1 1

# Optional: is the printed board actually flat?
python check_flatness.py --camera 0 --param cam0_intrinsics.npz
```

> Only one process can hold a V4L2 camera at a time — never run two scripts
> (or a script + the browser tools) against the same device simultaneously.

### Browser tools

```bash
# Live tag viewer: both cameras + seen/missing marker chips
python web_tags.py --cam0 0 --cam1 1          # http://<host>:5000/
python web_tags.py --board small              # track the small 6x8 board instead

# Calibration assistant: capture poses in the browser, solve in-browser
# Step A - intrinsics, once per camera, with the SMALL board
python web_calibrate.py --mode individual --board small --camera 0 --output cam0_intrinsics.npz
python web_calibrate.py --mode individual --board small --camera 1 --output cam1_intrinsics.npz

# Step B - cameras relative to each other, SMALL board seen by both
python web_calibrate.py --mode stereo --board small --cam0 0 --cam1 1 \
    --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz

# Step C - cameras relative to the tank, LONG floor board
#   (place the board in the tank; add --tank-t / --tank-rpy if the board
#    frame is not already the tank frame; defaults assume it is)
#   10 pictures are captured/kept per camera by default (--samples 10)
python web_calibrate.py --mode world --board long --cam0 0 --cam1 1 \
    --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz \
    --tank-output tank_alignment.npz
```

The web UI defaults to port **5001** (`--port 5001`), so open
`http://<host>:5001/`. In `web_calibrate.py` press **Capture Now** to store each
pose and **Finish & Calibrate** to solve. It writes the exact same files the CLI
does (`cam0_intrinsics.npz`, `cam1_intrinsics.npz`, `stereo_params.npz`,
`tank_alignment.npz`).

The dry-validation tool has its own browser UIs (ports **5002** for capture and
**5003** for labelling):

```bash
# Capture full-resolution stationary pairs; open the printed URL
python validate_stereo_3d.py capture

# Label correspondences in saved pairs; no cameras are opened
python validate_stereo_3d.py label
```

In the capture page, press **Capture Pair** for each stationary pose. The live
previews show an orange crosshair at the calibrated big-board origin
`(0,0,0)` plus 50 mm board axes: `+X` red, `+Y` green, `+Z` blue (currently
downward). This uses the saved board pose in `tank_alignment.npz`, is drawn
only on the preview stream, and is not written into the saved PNGs. In the
labelling page, click a point in cam0 then the same point in cam1; use the
mouse wheel to zoom, right/shift-drag to pan, and the point table to edit or
delete. **Save** writes raw full-resolution pixel coordinates and also saves
derived geometry for each point:

* `pixel_xyz`: x, y in cam0 raw pixels; z is the horizontal disparity
  `cam0.x - cam1.x` in pixels (not metric depth),
* `X_cam0_mm` and `X_board_mm`: triangulated Camera-0 and big-board coordinates
  in millimetres,
* height above the board, validity flags, reprojection/epipolar errors and
  parallax.

The derived fields are recomputed on every save; the raw `cam0`/`cam1` pixels
remain authoritative. Opening an older pair computes them for display, and
pressing **Save** writes them into that pair's `correspondences.json`.

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
actuals and predicted-to-actual error lines on one equal-scale graph. Board
`+Z` is physically downward, so use the height column for upward height. This
page never edits the calibration files. The two C920s are not
hardware-synchronized, so the target must not move during a capture.

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
