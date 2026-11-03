# Dry-board geometry diagnostic: findings and lab procedure

## Scope and staged investigation

1. Audit the current repository and all four input files, including their SHA256
   hashes, schemas, image sizes, board definitions, and coordinate conventions.
2. Compare the full camera-to-camera transform from stereo with that implied
   by the two independent board poses. This stage needs no images.
3. Restrict detection to the direct physical long board in each raw image.
   Compare known corner projections, independent ray/plane intersections, and
   unconstrained stereo for the same chess-corner IDs. Inspect exclusions and
   signed residual patterns before interpreting summary numbers.
4. Verify the geometry code with synthetic tests and repeat the dry-board test
   on fresh stationary pairs. Preserve both invalid and successful results.
5. Only then test the independently measured 155 mm blue-dot span, off-plane
   heights, a measured board-to-tank transform, and eventually wet dynamics.

This diagnostic never fits or writes calibration parameters, rescales stereo,
or moves free stereo points onto the floor. It measures internal consistency
against the nominal printed board. Because that board also supplied the world
poses, this is not independent proof of physical millimeter accuracy.

## Confirmed current files (inspection on November 3, 2026)

Source: `rphysduo/programming`, fetched `master` at `2ea435a` before implementation.
The root `camera/` arrays match the reported October 29 stereo and world values;
they are different from the older session snapshot under `validation/calibration/`.
Calibration RMS and capture dates are not stored in these four NPZ schemas, so
the October 29 association follows array agreement with the handoff/screenshots,
not a date field or filesystem modification time.

| File | SHA256 |
| --- | --- |
| `cam0_intrinsics.npz` | `529adac7fd421b70ee4c4e8abc5b853d1395512ddfa12d6adac3bd79c25e8992` |
| `cam1_intrinsics.npz` | `e4736c561d61e1622a163b5e7912f92fc10468f8e4f3fce5c5d82c78b8e5766b` |
| `stereo_params.npz` | `517fab792585cd9ecbf355a8b05f3753303a15b7d08171717be071b279a37b01` |
| `tank_alignment.npz` | `def332fa8a80ab32f46eb34517a1dbb21b8ef6026fb3785523f5592fa27fabea` |

Both intrinsic sizes are 1920×1080; their K/D values match the embedded stereo
K0/D0/K1/D1 exactly. The saved E/F agree with those K/D and R/T. The board is
the original modern 4×35, 44 mm square / 33 mm marker pattern, DICT_5X5_250.
Use `getChessboardCorners()[id]`, retaining original IDs after trimming.
The remaining strip's outer dimensions do not relocate the mathematical origin.

Conventions verified in the implementation:

```text
X_cam1 = R_stereo @ X_cam0 + T_stereo
X_cami = R_board_to_cami @ X_board + t_board_to_cami
X_board = R_board_to_cam0.T @ (X_cam0 - t_board_to_cam0)

R_world_implied = R_board_to_cam1 @ R_board_to_cam0.T
T_world_implied = t_board_to_cam1 - R_world_implied @ t_board_to_cam0
```

Both translations compared above are in camera-1 axes and millimeters.
Triangulation uses normalized undistorted rays with `[I|0]` and `[R|T]`.
There is no image rotation, rectification, resizing, or K multiplication at
that stage. The manual labeller's `rawToScreen` / `screenToRaw` functions map
display zoom/pan back to original pixels; this diagnostic bypasses manual
clicks entirely. Physical camera serial/device ordering still needs checking
in the lab: image size and hashes alone cannot prove which device is cam0.

| Comparison | Result |
| --- | ---: |
| Stereo T, mm | `[2.144401, -34.124649, -1.881317]` |
| World-implied T, mm | `[2.360353, -37.406199, -5.267343]` |
| World minus stereo T, mm | `[0.215952, -3.281550, -3.386026]` |
| Relative rotation disagreement | 0.649562° |
| Translation-vector disagreement | 4.720209 mm |
| Stereo / world-implied baselines | 34.243678 / 37.848908 mm |
| Baseline difference | +3.605231 mm |

The existing validator also prints about 3.786 mm for the error in predicting
the board-origin translation (`R_stereo @ t0 + T_stereo - t1`). That is a
different quantity from the full 4.720209 mm translation-vector disagreement.
Both are labeled separately in the new JSON.

The board-to-tank transform is still the identity placeholder, and
`stereo_scale` is `provisional`. Camera centers have negative board Z (about
-575/-572 mm). Therefore board +Z must not be labeled verified physical up.
No tank coordinates are certified by this diagnostic.

## Reproduce the saved October 29 example

Run from `/home/nicholas/programming/camera` on the lab computer, or your own
checkout's `camera/` directory. Use the existing Python environment with NumPy
and OpenCV's `aruco.CharucoDetector`. These tools use Python 3.10+ syntax; an
original Jetson Nano's stock Python 3.6 is insufficient. Offline analysis can
run on a laptop using the unchanged calibration files and raw image pairs.
Node is not involved. Matplotlib is optional for the plots; JSON/CSV and image
overlays work without it.

```bash
cd /home/nicholas/programming/camera
# Activate your existing camera environment here, if it is not already active.
python -c "import sys, cv2, numpy; print(sys.version); print(cv2.__version__); assert hasattr(cv2.aruco, 'CharucoDetector')"

# Inspect exact current inputs without capturing or solving any calibration.
python charuco_geometry_diagnostic.py --inspect-calibration \
  --param0 ./cam0_intrinsics.npz --param1 ./cam1_intrinsics.npz \
  --stereo ./stereo_params.npz --tank ./tank_alignment.npz \
  --output-dir ../outputs/board_audit_run01

# Review the supplied polygon against the direct board, at raw resolution.
python charuco_geometry_diagnostic.py \
  --pair-dir validation/pairs/pair_0018_20261029_142637 \
  --roi-file diagnostics/pair_0018_direct_board_roi.json \
  --preview-only --output-dir ../outputs/board_preview_run01

# Analyze ALL coherent common corners, including geometrically invalid results.
python charuco_geometry_diagnostic.py \
  --param0 ./cam0_intrinsics.npz --param1 ./cam1_intrinsics.npz \
  --stereo ./stereo_params.npz --tank ./tank_alignment.npz \
  --pair-dir validation/pairs/pair_0018_20261029_142637 \
  --roi-file diagnostics/pair_0018_direct_board_roi.json \
  --max-corners 0 --output-dir ../outputs/board_pair0018_run01
```

Every output directory must be new; existing runs are refused rather than
overwritten. With the hashes above, OpenCV 5.0.0 and the included ROI, the
example produced 42 common corners, 40 homography inliers in both cameras,
and 40 analyzed points. The two rejected candidates remain in the selection
report with their pixels, residuals and exclusion reasons. Detector versions
and polygon boundaries may change the selected sample; they are recorded.

| Error on those 40 points | Result |
| --- | ---: |
| Median cam0/cam1 independent plane error | 0.722 / 0.746 mm |
| World-pose reprojection RMS cam0/cam1 | 1.318 / 1.274 px |
| Stereo reprojection RMS cam0/cam1 on this pair | 3.580 / 3.554 px |
| Median free stereo position error | 141.486 mm |
| Median absolute free stereo board-Z error | 94.194 mm |
| Free stereo position RMSE / maximum | 155.908 / 266.842 mm |
| Existing geometric gate invalids (retained) | 40/40 |

The signed Z bias grows from roughly +52 mm at board Y=264 mm to +140 mm
at Y=836 mm. This is a spatially systematic inconsistency, not just a clicked
origin offset. The ~0.77 px historical calibration RMS refers to the fitting
data; this held image pair has materially larger stereo residuals.

The same pair's capture/label metadata hashes point to the older stereo/world
session files. The diagnostic makes that discrepancy visible but does not
substitute those older arrays. These results explicitly reprocess the raw
images using the newer root files. A fresh pair with recorded current hashes
is required to distinguish rig movement between sessions from fitting/model
errors. Small independent plane errors support local world-pose consistency;
they do not establish independent intrinsic, print-scale or off-plane accuracy.

## Fresh lab checkpoint

1. Keep the dry long board and both cameras stationary. Remove the small board,
   ruler and other occluders for this checkpoint. Check serial/device ordering,
   1920×1080 raw orientation, focus (historically 30), and acquisition settings.
   Do not rotate the hardware to match a projected overlay. Keep original
   calibration files in place and compare their hashes with the audit.
2. Capture three pairs into a **new** session so calibration snapshots cannot
   silently come from a prior session:

   ```bash
   python validate_stereo_3d.py capture --calib-dir . \
     --session ../outputs/board_fresh_run01 \
     --cam0 0 --cam1 1 --width 1920 --height 1080 --fps 30 --focus 30 \
     --auto 3 --no-gui --interval 1
   ```

3. Inspect each saved raw cam0/cam1 PNG. The cameras are not exposure-synchronized;
   stationary targets are essential. First use `--preview-only` without an ROI
   on the chosen fresh pair, then define the direct physical board polygons.
   Read polygon coordinates in a raw-pixel image viewer. No chess-corner clicks
   are required. Create a new JSON with this structure, replacing the vertices
   with coordinates from the **fresh images**:

   ```json
   {
     "cam0": [[205,638], [1620,376], [1680,848], [207,825]],
     "cam1": [[319,194], [1748,361], [1742,566], [320,641]]
   }
   ```

   These numbers are only the old pair-0018 example. Choose at least three
   ordered vertices forming a convex polygon entirely inside each raw image.
   Include complete direct-board markers near its boundaries, exclude the side
   wall reflections, and avoid narrow masks that cut through useful markers.
   Re-run preview with `--roi-file` and inspect the cyan boundary and IDs.
   Do not accept a coherent wall reflection just because RANSAC accepts it.
4. Analyze each fresh pair with `--max-corners 0`, the inspected ROI, and a
   distinct output directory. To use exactly the files snapshotted at capture,
   pass these four explicit paths (replace the pair directory with its actual
   name printed by capture):

   ```bash
   python charuco_geometry_diagnostic.py \
     --param0 ../outputs/board_fresh_run01/calibration/cam0_intrinsics.npz \
     --param1 ../outputs/board_fresh_run01/calibration/cam1_intrinsics.npz \
     --stereo ../outputs/board_fresh_run01/calibration/stereo_params.npz \
     --tank ../outputs/board_fresh_run01/calibration/tank_alignment.npz \
     --pair-dir ../outputs/board_fresh_run01/pairs/PAIR_DIRECTORY \
     --roi-file ../outputs/fresh_direct_board_roi.json \
     --max-corners 0 --output-dir ../outputs/fresh_pair01_report
   ```

5. Compare signed residuals across board Y and image position over all three
   reports. Do not recalibrate based on one scalar RMS. No physical acceptance
   threshold is set here: the default 5 mm / 1° classifications are explicitly
   demonstration limits, and requirements must follow the wave-height task.

## Output and interpretation

- `charuco_geometry_diagnostic.json`: audit, hashes, source/version information,
  raw pair metadata and timings, exclusion ledger, full transforms, per-corner
  coordinates, signed error vectors, validity flags, and statistics by board Y.
- `charuco_geometry_corners.csv`: flattened raw pixels, known/projected/plane/
  free-stereo values and signed residuals. Error convention is **estimated or
  projected minus known or observed**. Missing quantities are null/empty;
  invalid finite XYZ values remain unchanged.
- `summary.txt`: numerical comparison and diagnosis. `valid` only describes the
  existing geometric gates, never physical accuracy.
- `cam0_annotated.png`, `cam1_annotated.png`: original-resolution evidence with
  ROI boundary, detected ID circles, known-point world projections as magenta
  crosses/arrows, and signed stereo dz labels. `!` means inspect the flags.
- With Matplotlib: `error_pattern.png` and `board_3d.png`, with equal metric
  scaling on the 3D axes. Invalid finite reconstructions are included.

Small world projection/plane errors with large free stereo errors implicate
inconsistent relative geometry or its application. Large world errors in one
camera instead implicate its pose/intrinsics, board geometry, detection or
identity. Errors concentrated at page seams suggest a target/model issue.
The diagnostic reports evidence for these alternatives, not a unique physical
cause or a corrected reconstruction. Geometric invalid counts are separate
from the model-consistency diagnosis so failed points cannot disappear.

## Tests and subsequent physical checks

From the repository root:

```bash
python -m pytest camera/tests/test_charuco_geometry_diagnostic.py
python camera/validate_stereo_3d.py selftest
```

Synthetic tests include ideal geometry, opposite rolls with distortion,
incorrect stereo scale despite near-zero reprojection error, camera swaps,
wrong corner identity, repeated board images/duplicate IDs, perturbed world
poses, parallel rays, output preservation and calibration hash preservation.

After diagnosing the board, remeasure the blue-dot center-to-center span
independently (currently reported as **155 mm**). Use a new ruler session,
label the same two dot centers `A` and `B` in both raw images over at least
three stationary pairs, and create a fresh plan:

```bash
python validate_stereo_3d.py capture --session ../outputs/ruler_run01 --calib-dir .
python validate_stereo_3d.py label --session ../outputs/ruler_run01
python validate_stereo_3d.py plan --session ../outputs/ruler_run01
```

Edit only the new `measurement_plan.json`, retaining its other fields, with:

```json
"length_tests": [
  {"name": "blue_dot_span", "p1": "A", "p2": "B", "kind": "euclidean", "truth_mm": 155.0}
]
```

Then run `python validate_stereo_3d.py measure --session ../outputs/ruler_run01`.
The 155 mm value is comparison truth only. Do not rescale results. Repeat
across the width, along the tank, and at independently measured off-plane
heights. Only afterward establish measured tank Y/up-Z axes and a still-water
reference; retain full X/Y/Z when extracting `eta(y,t)`.
