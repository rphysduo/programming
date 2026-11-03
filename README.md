# RPhys Wave Tank: Calibration and Simulation

Tools for the RPhys wave-tank experiment: stereo camera calibration, dry 3D
measurement validation, a browser-based CAD viewer with synthetic waves and
camera rays, and experimental hardware scripts.

Start with the camera tools to establish measurement geometry, or run the web
simulator to explore the assembly visually. The simulator currently uses
manually specified camera positions; it does not load the saved camera
calibration or reconstruct waves from video.

## Repository layout

| Location | Purpose |
| --- | --- |
| [camera/](camera/README.md) | ChArUco detection, lens calibration, stereo geometry, tank alignment, and measurement validation. |
| [wave-cad-sim/](wave-cad-sim/README.md) | Interactive CAD assembly, synthetic wave surface, and stereo ray visualization. |
| `main.py` | Host script sending sinusoidal paddle commands over serial. |
| `servos/main.py` | MicroPython firmware controlling two PWM servos from serial commands. |
| `ultrasonic/main.py` | Host script reading sensor output over serial. |
| `point_graphing/` | HTTP point-data client; see its script arguments and requirements. |

The hardware scripts contain device-specific serial paths and GPIO settings.
Review those settings and the physical setup before running them; the host
paddle script sends commands continuously, and the servo firmware initializes
the motors when started.

## Camera tools

### Set up Python

From the repository root, create a fresh environment with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-contrib-python flask pytest
```

The camera tools need an OpenCV build with `cv2.aruco.CharucoDetector`.
Install one OpenCV distribution in this environment; use the contrib package
above for the ChArUco tools. Live capture is configured for Linux/V4L2 with
C920-class USB cameras at 1920×1080 and 30 fps. Saved-image diagnostics do not
require camera hardware.

### Calibrate the rig

Run camera commands from `camera/` so the default calibration paths resolve:

```bash
cd camera

# 1. Fit each camera's lens parameters using varied poses of the small board.
python calibrate_individual.py --board small --camera 0 --output cam0_intrinsics.npz --samples 20
python calibrate_individual.py --board small --camera 1 --output cam1_intrinsics.npz --samples 20

# 2. Fit the relative camera rotation and translation.
python calibrate_studio.py --board small --cam0 0 --cam1 1 --samples 15

# 3. Locate the fixed cameras relative to the stationary long floor board.
python calibrate_world.py --board long --cam0 0 --cam1 1 --save-images

# 4. Check reconstruction against physically measured distances.
python test_calibration.py --board long --cam0 0 --cam1 1
```

The board definitions are in `camera/charuco_common.py`: the small board is
6×8 squares with 28 mm squares and 21 mm markers; the long board is 4×35
squares with 44 mm squares and 33 mm markers. Verify the physical printed
spacing against these definitions. Keep focus and resolution consistent
between calibration and measurement.

For the browser calibration assistant, run one stage at a time:

```bash
python web_calibrate.py --mode individual --board small --camera 0 --output cam0_intrinsics.npz
```

Open `http://localhost:5001`. Repeat for camera 1 with its own output file,
then use `--mode stereo --board small` and `--mode world --board long`.
See the [camera guide](camera/README.md) for capture controls and other tools.

### Understand the calibration files

| File under `camera/` | Contents |
| --- | --- |
| `cam0_intrinsics.npz`, `cam1_intrinsics.npz` | Lens matrix `K`, distortion `D`, and image size. |
| `stereo_params.npz` | Relative geometry using `X_cam1 = R @ X_cam0 + T`; translation is in millimeters. |
| `tank_alignment.npz` | Board-to-camera poses and the supplied board-to-tank placement. |

The default board-to-tank transform is the identity. Establish the physical
tank origin, longitudinal direction, and upward direction before treating
board coordinates as verified tank coordinates. Saved parameters remain
provisional until checked against independent physical measurements.

### Diagnose a saved image pair

From `camera/`, this example processes a pair already present in the repository:

```bash
python charuco_geometry_diagnostic.py --board long \
  --pair-dir validation/pairs/pair_0018_20261029_142637 \
  --roi-file diagnostics/pair_0018_direct_board_roi.json \
  --output-dir ../outputs/charuco_geometry_diagnostic
```

The diagnostic compares known board coordinates, each camera's independent
ray intersection with the board plane, and unconstrained stereo reconstruction.
It also compares saved stereo `R,T` with the relative geometry implied by the
two board poses, and checks calibration-file hashes against capture metadata.
It writes a JSON summary and per-corner CSV without changing calibration files.
Its classification thresholds are demonstration values.

Follow the [diagnostic lab procedure](camera/diagnostics/README.md) to inspect
calibration hashes, verify the direct-board ROI, and interpret signed errors.

A board corner should reconstruct near its known position and near board
`Z=0`. Water-surface points must be allowed to reconstruct above the floor.
Validate raised reference objects as well as the floor board before measuring
wave heights. The webcams are not hardware synchronized, and saved read
timestamps do not establish simultaneous exposure; dry references should remain
stationary during capture.

## Web simulator

Requires Node.js 22.13 or newer. From the repository root:

```bash
cd wave-cad-sim
```

Copy your original `Full Assembly.stl` to `public/full-assembly.stl`. This large
CAD file is intentionally excluded from Git and must be supplied separately.
The viewer assumes STL coordinates are meters and CAD Z points upward.

```bash
npm ci
npm run dev
```

Open `http://localhost:3000`. Drag to orbit, scroll to zoom, adjust wave
amplitude/frequency/phase speed, change ray density, toggle scene layers, and
pause or resume the animation.

This is a synthetic 2.5D digital twin, not validated free-surface CFD. It does
not simulate overturning, splashing, turbulence, or air entrainment. Camera
placement and dashboard timing/synchronization readouts are illustrative;
they do not report actual camera measurements.

## Verification

From the repository root, with the Python environment active:

```bash
python -m pytest camera/tests/test_charuco_geometry_diagnostic.py
```

From `wave-cad-sim/`, after supplying the CAD asset and installing dependencies:

```bash
npm run lint
npm test
```

`npm test` builds the simulator and checks the rendered interface and asset
wiring. These checks do not establish experimental calibration accuracy or
validate the wave physics. Keep generated reports under `outputs/`, and keep
virtual environments, caches, and generated media out of new commits.
