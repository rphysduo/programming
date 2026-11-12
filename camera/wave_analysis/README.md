# Recorded-wave inspection and manual annotation

This implements the first two pipeline stages and a diagnostic point-geometry export. The source baseline is `rphysduo/programming` master commit `84869186996338c930c25d07b00d156407bd5956`.

The current tools inspect saved videos, review original frames, store versioned observations, and reproduce board-coordinate XYZ for researcher-verified stereo matches. Physical wave profiles and physics features require a reviewed surface pilot, measured coordinate conventions and a still-water reference.

## Run the lab workflow

Use the existing camera Python environment. For a new laptop environment, install the dependencies from the repository's camera setup guide. This workflow uses NumPy, OpenCV and Flask, with pytest for tests.

Run these commands from the repository root containing `camera/`. The analysis code must be installed in this checkout before running them.

```bash
cd ~/programming
SESSION=outputs/wave_trials/wave_20261110_140220_415247

python camera/analyze_wave_trial.py inspect \
  --session "$SESSION" \
  --output-dir outputs/analysis/feasibility01
```

Choose a new output name for each inspection. The command reads the original files, decodes both complete videos, checks frame counts and dimensions, audits calibration hashes, checks exported pair metadata, and writes nine representative raw views per camera plus a contact sheet. It also checks timestamps for missing indices, duplicates, reordering and inconsistent clock origins. Equal decoded pixel images are flagged as candidates for review, since a static scene can produce identical images. Exposures lost before acquisition are unknown.

Open `feasibility_report.md`, `quality_report.json` and `review/contact_sheet.jpg`. Record a visual assessment in a separate `surface_review.md`. The inspection command requires human review to decide whether the visible features are usable water-surface observations.

```bash
python camera/analyze_wave_trial.py annotate \
  --session "$SESSION" \
  --output-dir outputs/analysis/trial01/annotations \
  --host 0.0.0.0 --port 5005
```

Open `http://JETSON_IP:5005` from the lab laptop. On the computer running the server, use `http://127.0.0.1:5005`. The default server host is localhost when `--host` is omitted. The server uses the trial's own calibration snapshots and original video frames. All frames are accessible through timestamp pairing, including when a frame is reused. There is no need to re-export images first.

1. Start with pairs **0, 23, 45, 68, 91, 113, 136, 159 and 181** for the November 10 trial. Inspect nearby pairs to interpret motion. The new pair indices are zero-based and correspond to the frame numbers displayed in the picker.
2. Choose **Stereo point**, click a physical feature in either view, and then click the same feature in the other. Each candidate starts as **Uncertain**. Inspect it, add a label and confidence, and explicitly select **Verified physical match** when justified. The yellow curve supplies epipolar guidance. Confidence is a reviewer assessment, and has no automatic millimeter interpretation.
3. If the other view is ambiguous, choose **Keep unmatched point**. Uncertain and unmatched observations remain available for review.
4. Choose **Image trace** and click ordered vertices along one visible water boundary. Click **Finish trace / mask**, then start the other view independently. Set **Vertices follow propagation** only when the direction is known. Each trace remains an image observation.
5. Use the reflection, occlusion or glare tools to draw exclusion polygons. Finish each polygon with at least three vertices. Point geometry inside a mask carries its mask IDs.
6. Wheel to zoom independently in each view. Right-drag or Shift-drag to pan. Use **Fit views** to reset. In **Select / edit**, drag an observation or trace vertex to correct it. Moving a verified point returns it to Uncertain. Use **Delete selected** to remove an item from the next revision. Earlier revisions remain available on disk.
7. Use **Earlier along profile** and **Later along profile** to record correspondence order. This order expresses reviewer intent. A later physical profile will need to check actual reconstructed longitudinal and cross-tank coordinates.
8. For breaking, choose **Yes** on the pair containing the first visible onset, select the camera and record the visual criterion. **Breaking location** records an optional raw-pixel location. Add a wave ID or `trial`, confidence and notes. The server derives frame number and time from the original camera log. A **No** label requires that the complete outcome remained observable. Use **Unknown** when the outcome is ambiguous or leaves view.
9. Click **Save revision**, navigate away and reopen the pair. Verify that points, traces, masks, notes and breaking labels return. A concurrent stale save is rejected so another reviewer's revision remains intact.

After saving reviewed points, export their diagnostic geometry into a fresh sibling directory.

```bash
python camera/analyze_wave_trial.py extract \
  --session "$SESSION" \
  --from-annotations outputs/analysis/trial01/annotations \
  --output-dir outputs/analysis/trial01/geometry01
```

This writes `surface_observations.csv`, `breaking_events.csv`, `reconstruction.json` and `analysis_manifest.json`. The JSON contains every included annotation revision, source coordinates, timing and geometry flags. The CSV retains raw pixels, board XYZ, both original elapsed timestamps and frame indices, reviewer confidence, masks and geometry errors. Original read timestamps and camera device identities are also retained in the JSON.

The current `extract` command provides point diagnostics. `measure`, automatic surface detection, `extract --config`, physical profile generation and velocity estimation belong to later checkpoints. Fields for physical height, eta and positional uncertainty remain null. Masked and geometrically invalid observations stay available for diagnostics.

## November 10 feasibility findings

The inspected trial contains 167 cam0 frames and 182 cam1 frames, fully decoded at 1920 by 1080. Observed rates are 8.0495 and 8.7957 fps. Median sample intervals are 125.204 and 114.231 ms. Its 182 candidate pairs have median host-read gap 32.408 ms, 95th percentile 58.991 ms and maximum 61.455 ms. Fifteen cam0 frames are reused once each.

All four recorded calibration hashes agree with the saved snapshots. The baseline is 103.300 mm. Independent world poses imply 104.412 mm, with relative rotation disagreement 0.5356 degrees and translation-vector disagreement 2.5836 mm. The board-to-tank transform is an identity placeholder and the saved scale metadata is provisional. Physical camera serial identities and exposure timestamps are unavailable.

Visual inspection of nine representative pairs shows moving disturbances and boundary features in both views. Transparency, background edges and reflections make continuous physical surface correspondence uncertain. Some features may support a reviewed sparse stereo pilot. A wall waterline may be more observable in particular regions, subject to a verified wall-plane and optical-path calibration. First breaking onset and a reliable longitudinal metric profile remain unestablished by this review.

## Provenance and measurement conventions

- The recorder's nearest-timestamp pairing logic is reused. All frame times come from the original CSV logs, with integer nanosecond differences calculated before unit conversion. Header fps is reported separately.
- Each pair records original camera frame indices and both timestamps. Frame reuse is visible in the interface and reports. A later tracking system must account for shared observations.
- Host read gaps are measured. Exposure skew and its uncertainty remain unknown. A small read gap alone does not establish simultaneous observations.
- The existing `load_calibration`, `triangulate_point`, `cam0_to_board`, `compare_relative_geometry` and `predict_correspondence` functions supply calibration, XYZ, error checks and guidance. The original 3 px reprojection and 2 px epipolar gates are retained.
- XYZ is in the saved board frame and millimeters. Reconstructed board Z retains its original sign. Physical up, the board-to-tank transform, still-water level, bottom geometry and an intended X slice need measured provenance before eta or physical slopes can be interpreted.
- Annotation manifests fingerprint the original videos, timestamp logs, metadata and calibration files. Every revision records the source pair, code revision and source-code hashes. Existing annotation stores reject a different input snapshot.
- Annotation JSON files are published atomically as exclusive revisions. Saved-session `correspondences.json` files belong to the existing labeler. This workflow writes its own separate annotation tree.
- Default breaking criteria are a starting review convention. Researchers must apply them consistently and inspect surrounding frames. Labeling confidence and geometry validity are independent of experimental accuracy.

## Next checkpoints

1. Complete the manual 5–15-frame pilot. Verify physical feature identity, cross-camera timing limitations, visible spatial coverage and reproducibility between reviewers.
2. Establish physical up, measured board-to-tank axes, still-water elevation and the intended X slice. Validate known heights and lengths through the actual wet optical path. Then construct sparse longitudinal profiles, preserve gaps and estimate positional uncertainty.
3. Implement slope, crest/trough height, wavelength and steepness only where profiles support them. Read the Babanin paper's exact shape definitions before implementing geometric skewness and asymmetry. Test direction reversal, noise, incomplete profiles and units.
4. Evaluate simple automatic suggestions against held-out human-reviewed frames after the pilot succeeds.
5. Track crest and envelope observations with the original timestamps and shared-frame provenance. Report group speed only when a distinct envelope can be followed.
6. Join validated measurements, measured experimental settings and reviewed breaking outcomes by trial and wave ID.
7. Compare prediction baselines using whole-trial splits when sufficient independent experimental trials exist.

The visual review is provisional. Synthetic geometry tests establish numerical behavior. Experimental measurement accuracy and breaking prediction require lab validation.

## Verification

```bash
python -m pytest
python camera/validate_stereo_3d.py selftest
node --check camera/wave_analysis/static/annotate.js
```

Node is only an optional JavaScript syntax-check tool during development. The annotation app runs with Python and a browser.

The focused tests use synthetic MJPG videos, irregular timestamps and known camera geometry. They cover malformed metadata, missing files, changed calibration hashes, source immutability, annotation revision history, concurrent saves, bad pixels, masks, unmatched observations, breaking labels, preserved rejection thresholds, Flask endpoints and reproducible reconstruction after reopening.

Generated reports, test recordings, caches and screenshots belong under `outputs/` and should be excluded from source commits.
