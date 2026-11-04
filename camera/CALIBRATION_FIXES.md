# Calibration capture fixes

These changes repair capture/acceptance defects. They do not recalibrate the
rig, change any saved parameters, or correct previously reconstructed points.
Continue the [dry-board diagnostic procedure](diagnostics/README.md) before
deciding whether a new physical calibration is necessary.

## Interpreting negative Z and invalid points

Negative **board** Z is allowed. The triangulator requires positive depth in
each **camera** frame; a point behind either camera remains invalid. Reprojection
error over 3 px in either camera, Sampson epipolar error over 2 px, undefined
coordinates, or ray angles below 0.01 degrees also reject a point. These gates
are unchanged. The label table now displays the actual rejection explanations,
camera depths, reprojection/epipolar values and ray angle in a visible
**Geometry diagnostics** column. It previously showed flags only on hover.
Reopening a pair recomputes explanations; Save persists them with the derived
geometry. Accepted geometry still does not certify physical accuracy.

The capture crosshair at `(0,+176,0)` mm is a preview-only location, not the
saved board origin. `--preview-origin-y-rows 0` shows the original `(0,0,0)`.
Neither setting changes saved points or point-to-point distances. With board
+Z physically downward, raised points may have negative board Z, but a point
on the saved board plane should still have Z near zero, within measurement
uncertainty. A floor offset near -50 mm needs an independently measured board/
floor relationship or a geometry diagnosis; it cannot be explained by sign
convention alone. Do not subtract that offset, reverse signs, or rescale lengths
to force agreement.

For the November 3 ramp screenshot, the nearby points are **B and D**. Their
cam0 observations differ by about 2.9 px, while cam1 differs by 16.6 px; they
are not identical stereo correspondences. Their reconstructed Y changes by
169.5 mm. A short baseline and mismatched rays can produce large coordinate
changes; the latest session's actual NPZs/raw pairs/labels are needed to report
the exact failing checks and distinguish click sensitivity from model errors.
The screenshot's active baseline is 33.3311 mm, so older root files cannot be
used as substitutes to reproduce these results.

## What changed

- Both CLI and browser world fitting reject collinear or insufficient unique
  object points, nonfinite observations/poses, nonpositive camera depths, and
  excessive reprojection RMS. Pooling stationary repeated observations remains
  supported. The default `--max-world-rms-px 3.0` is a configurable pixel-quality
  gate, **not** a claim of physical millimeter accuracy. The combined fit is
  validated as well as each individual view.
- Browser and CLI world calibration require the requested number of valid
  pictures from **each** camera. Finish/quit before that does not save a partial
  world fit. The browser shows both counts and rejected-pose reasons.
- Intrinsic, stereo, and world capture now use shared direct-board ROI controls
  and exclude all occurrences of ambiguous duplicate chess-corner IDs. Headless
  stereo assembles each shared ID exactly once. Raw pixels remain unrotated,
  uncropped and unscaled; masking is applied before board decoding.
- Stereo/world capture refuses camera/intrinsic image-size mismatches. This
  protects the lens-model coordinate convention.
- New calibration files use exclusive writes: existing NPZs cannot be replaced,
  even if a file appears between startup and save. Choose a fresh candidate path
  under `outputs/`. Nonfinite data cannot be exported. Future candidates also
  record their fit RMS; current calibration artifacts remain byte-identical.
- The validation labeller now displays the **active loaded baseline**, file
  paths and abbreviated SHA256 hashes. A visible warning identifies differences
  from available source files. `/api/calibration` exposes full hashes. Pair
  metadata discrepancies are also reported. Source comparison is made when the
  server starts; restart the labeller after intentional file changes.

## Before a future candidate calibration

Do not rerun calibration merely to make an expected ruler length appear.
First identify the relevant failure through the preserved-file diagnostic.
The following commands are only for a separately justified candidate run.

Run from the `camera/` directory with the existing Python camera environment.
Keep the old four NPZ files and old validation sessions in place.

For stereo/world modes, provide direct-board polygons in original raw pixels:

```json
{
  "cam0": [[205,638], [1620,376], [1680,848], [207,825]],
  "cam1": [[319,194], [1748,361], [1742,566], [320,641]]
}
```

These coordinates describe the old pair-0018 view only. Make and inspect a new
polygon file for the current scene; do not reuse it after camera movement.
Use the diagnostic's `--preview-only --roi-file ...` to inspect a saved raw
pair. The browser calibration previews also show the ROI boundary. For
individual mode, use the key `"camera"` with the polygon for that one stream,
regardless of the physical device index.

For a small movable board, the ROI must include its intended pose/edge coverage
while excluding reflected/other targets. If the target is physically isolated
and no reflections or other board are visible, explicitly choose
`--allow-full-frame` instead. Duplicate-ID removal alone cannot distinguish a
single coherent reflected board.

Example future stereo candidate, keeping the successful active intrinsics:

```bash
python calibrate_studio.py --board small --cam0 0 --cam1 1 \
  --allow-full-frame \
  --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz \
  --output ../outputs/candidates/stereo_candidate_run01.npz
```

Example future world candidate:

```bash
python web_calibrate.py --mode world --board long --cam0 0 --cam1 1 \
  --roi-file direct_board_roi.json --samples 10 --max-world-rms-px 3 \
  --param0 cam0_intrinsics.npz --param1 cam1_intrinsics.npz \
  --world-output ../outputs/candidates/tank_candidate_run01.npz
```

Both require deliberate captures on stationary targets. No new candidate is
automatically promoted to an active file. Validate its geometry and independent
physical distances/heights with explicit paths and a fresh session first.

## Why the screenshot used older parameters

The six saved points in `pair_0018_20261029_142637` reproduce exactly with
`validation/calibration/` (35.9127 mm baseline). The root stereo file has a
34.2437 mm baseline and gives different coordinates and validity flags for the
same raw clicks. Both parameter generations can still be metrically inaccurate.

Sessions intentionally retain their original calibration snapshots. To validate
the current active files, create a **new** session:

```bash
python validate_stereo_3d.py capture --calib-dir . \
  --session ../outputs/dry_current_run01 --auto 3 --no-gui --interval 1
python validate_stereo_3d.py label --session ../outputs/dry_current_run01
```

Check the baseline in the new label-page banner. Expanding it shows paths,
hashes and source-mismatch warnings. Do not overwrite the old session to change
what an earlier result meant. `valid` still means geometric consistency checks,
not verified physical position; all X/Y/Z coordinates remain available.

## Tests

```bash
python -m pytest camera/tests
python camera/validate_stereo_3d.py selftest
```

Run these from the repository root. Regression tests cover NaN/collinear/
behind-camera world fits, corrupted points, duplicate observations, reflection
masks, asymmetric camera sample counts, premature finish, image-size checks,
non-overwriting candidate exports, and the older-snapshot UI warning. Tests
use synthetic observations or read existing files; they do not recalibrate or
replace the real camera parameters.
