# Phases 4, 5, and 7: lab workflow

These stages run offline on existing recordings. They add conservative automatic
track proposals, reviewable image suggestions, and simple breaking/location
prediction tools. They have been tested with synthetic data. Real detection
accuracy, velocity accuracy and predictive skill remain to be measured.

Start in your GitHub checkout's `camera` folder with the existing camera Python
environment active:

```bash
cd ~/programming/camera
python -m pip install -r wave_analysis/requirements.txt
```

No Node server is needed. NumPy, OpenCV and Flask come from the existing camera
environment; the analysis requirements add Matplotlib and scikit-learn.
For a completely synthetic demonstration, with no camera connected:

```bash
python demo_wave_automation.py --output-dir ../outputs/analysis/phases457_demo01
```

Use a new output name for each run. This creates profile plots, automatic tracks,
a synthetic edge overlay, detector errors, artificial trial features, simple
classifier/regression models and evaluation reports. Artificial labels are
explicitly marked; they do not simulate actual breaking physics.

## Phase 5: surface suggestions and human review

Set these to an existing recording and its annotation directory (replace
`YOUR_RUN` with the actual recording name). Existing annotation stores remain
compatible; saves append new revisions.

```bash
SESSION=../outputs/wave_trials/YOUR_RUN
ANN=../outputs/analysis/YOUR_RUN_manual01
python analyze_wave_trial.py annotate --session "$SESSION" --output-dir "$ANN" --port 5005
```

Open `http://localhost:5005` on that computer.

1. Choose a pair. Draw **one rough Image trace per camera** following the visible
   waterline, with vertices progressing consistently left-to-right or
   right-to-left in that image. The trace defines a search band; it does not
   establish stereo identity. Draw reflection/glare/occlusion masks first.
2. Set **Search band ±px** (12 raw pixels initially) and click **Suggest from
   rough traces**. The detector searches for isolated vertical intensity edges
   in that band. Low contrast, competing edges and masked columns leave gaps.
   A folded or near-vertical image contour needs a different method/manual work.
3. New waterline segments and patch matches are machine suggestions. Select a
   segment or point, inspect it, and use **Accept selected suggestion**. Drag
   vertices/points to correct them. Use **Delete selected** to reject a proposal;
   earlier saved revisions preserve history. Add missing observations manually.
4. Check both observations of every stereo point. Matching considers upright
   and 180-degree image patches near the saved epipolar curve, with a mutual
   uniqueness check. This simple matcher can fail under viewpoint changes,
   repeated texture, glare and transparent water. Zero matches is legitimate.
5. **Save revision**. Accepted points still undergo all original stereo validity
   tests. Accepted single-view traces do not become 3-D points. Geometry and
   physical measurement validation remain separate.

Confidence is an edge-margin or normalized patch-correlation score, not a
calibrated probability or millimeter error. Proposal diagnostics retain coverage,
missing columns, patch scores and triangulation flags. Pixel/position uncertainty
is explicitly unknown until evaluated; the existing physical extraction stage
propagates measured pixel uncertainties after correspondence review.

For repeatable batch proposals, save rough traces/masks on each requested pair,
then write proposals to a **new** annotation directory. It can be opened in the
same app without copying files:

```bash
python analyze_wave_trial.py suggest --session "$SESSION" --annotations "$ANN" --pairs 0 5 10 --band-px 12 --output-dir ../outputs/analysis/YOUR_RUN_proposals01
python analyze_wave_trial.py annotate --session "$SESSION" --output-dir ../outputs/analysis/YOUR_RUN_proposals01 --port 5005
```

Pair indices above are examples, not a required sampling interval. Batch output
also includes original-frame overlays. Do not repeatedly generate duplicates in
one pair; remove unwanted proposals or reopen the last saved revision first.

### Held-out detector evaluation

Reserve manually traced reference pairs before tuning the band/thresholds. Keep
reference annotations separate from untouched machine proposals. Evaluation
checks that neither camera frame is shared between tuning and evaluation pairs.
It rejects accepted/edited machine traces. Use the batch proposal directory
*before review*, or regenerate an untouched set in a new folder.

```bash
python analyze_wave_trial.py evaluate-suggestions --session "$SESSION" --reference ../outputs/analysis/YOUR_RUN_reference --suggestions ../outputs/analysis/YOUR_RUN_untouched_proposals --tuning-pairs 0 5 --pairs 10 15 --output-dir ../outputs/analysis/YOUR_RUN_detection_eval01
```

Read `heldout_errors.csv` and `evaluation.json`: mean and 95th-percentile
raw-pixel contour distance, precision and reference coverage within a declared
3-pixel tolerance. This tolerance is configurable with `--tolerance-px`; it is a
review criterion, not a universal accuracy standard. Evaluation is conditional
on manually initialized search bands, not fully autonomous segmentation.
Independent stereo correspondences and known physical objects still validate
metric reconstruction. Ground truth should be traced independently of proposals.

## Phase 4: wave identities and phase/group velocities

First complete `extract --config` and `measure` as described in
[PHYSICAL_ANALYSIS.md](PHYSICAL_ANALYSIS.md). The measured physical frame,
still-water level, slice and timing uncertainty are required lab inputs.

```bash
MEASURED=../outputs/analysis/YOUR_RUN_measurements01
python analyze_wave_trial.py track --measurements "$MEASURED" --kind crest --max-speed-m-s 1.0 --position-gate-m 0.01 --output-dir ../outputs/analysis/YOUR_RUN_crest_proposals01
```

The numeric gates are demonstration settings. Choose/document a plausible
maximum speed and position gate for your tank and measurement resolution.
The tracker uses actual timestamps, position prediction, height and wavelength
compatibility. It joins only unambiguous candidate associations. Competing
identities/possible merges/splits terminate tracks and are flagged; the algorithm
does not claim to identify the physical cause of ambiguity. Short missing
observations can be bridged; larger gaps end identities. Reused camera frames and
excessive host gaps are omitted. A reported breaking onset conservatively breaks
all active proposals at that time when wave identities have not been reviewed.

Inspect `trajectories.svg`, `tracking_report.json` and `track_links.json`. To
accept tracks, edit a copy of `track_links.json`:

- Rename `track_id` to the same wave ID used for your breaking annotations.
- Remove bad associations; split uncertain identities into separate tracks.
- Set `reviewed` to `true` and write actual review evidence in `evidence`.
- Keep at least three independent observations for a fitted speed.

Then rerun measurement into a new directory, using the **original extraction**
folder rather than the measurement folder:

```bash
python analyze_wave_trial.py measure --analysis-dir ../outputs/analysis/YOUR_RUN_profiles01 --tracks ../outputs/analysis/YOUR_RUN_reviewed_tracks.json --output-dir ../outputs/analysis/YOUR_RUN_measurements_reviewed01
```

Until reviewed, proposed speeds are provisional and never enter the accepted
research dataset. Regression residual errors and conditional position/timing
uncertainties are reported separately. Unknown exposure timing leaves combined
uncertainty unavailable; host read gaps do not establish exposure synchronization.

For group velocity, run the same `track` command with `--kind packet` and a new
output folder. It requires at least five measured crests with a single interior
height-envelope maximum and resolved lower flanks. Uniform trains, boundary
peaks and ambiguous envelopes produce no packet. The landmark is a sparse
crest-height-envelope proxy; its position uncertainty includes crest spacing.
Inspect the footage/profile sequence, set `distinct_envelope: true`, supply track
and individual observation evidence, then include reviewed packet links in the
same links list passed to `measure`. Group speed is not substituted for phase
speed, and no packet-to-wave membership is inferred. Linear finite-depth
reference speeds remain distinct from measured speeds.

## Phase 7: breaking and location prediction

The training/evaluation implementation is ready now. Real training requires
independent, validated trials with reviewed wave outcomes, documented timing
uncertainty and a fixed study design. Finish annotations first; there is no need
to fabricate labels or mark provisional measurements as validated.

1. `dataset` joins reviewed measurement folders. Supply a split-policy JSON that
   assigns each **whole trial** to `train`, `validation` or `test` (see the
   physical-analysis guide). Never split adjacent frames of one recording.
2. Choose a fixed early trial-relative cutoff and lead time *before examining
   held-out results*. The model selects the latest eligible observation per wave
   at/before that cutoff. Waves already breaking by cutoff + lead time are
   excluded, not retrospectively moved to earlier frames. Unknown outcomes are
   excluded; complete-observation nonbreaking labels come from the annotation
   workflow. A two-sigma timing allowance is a declared conservative convention,
   not a guarantee of camera synchronization.
3. Train and compare simple logistic models using conditions, steepness,
   steepness/depth, shape, and reliable **causal** phase speed. Conditions may
   include paddle amplitude when units and commanded/measured kind agree across
   trials. Group velocity stays in a separate packet table until membership can
   be measured; it is not silently assigned to a crest. Full-track speed remains
   diagnostic; predictive speed uses only observations available at that time.
4. Preprocessing uses training trials only. Validation selects regularization
   and feature family by Brier score. Test results are reported afterward.
   Baselines include training prevalence and the fixed deep-water 0.44 reference
   (a demonstration, not a universal tank breaking criterion).

Example commands after replacing the dataset and output names:

```bash
python analyze_wave_trial.py dataset --analyses ../outputs/analysis/TRIAL_A_measurements_reviewed01 ../outputs/analysis/TRIAL_B_measurements_reviewed01 ../outputs/analysis/TRIAL_C_measurements_reviewed01 --split-policy ../outputs/analysis/split_policy.json --output-dir ../outputs/analysis/study_dataset01
python analyze_wave_trial.py train-predictor --dataset ../outputs/analysis/study_dataset01 --observation-time-s 1.0 --lead-time-s 0.2 --output-dir ../outputs/analysis/study_model01
```

The three trial paths illustrate syntax; they are **not enough** for fitting.
The implementation requires both outcomes and at least two eligible trials per
split as a structural minimum. This is not an adequate-sample-size claim.
Use your full collection of independent trials and a justified study design.
`--synthetic-demo` is limited to explicitly synthetic dataset manifests; it is
not an override for unvalidated experimental measurements.

`evaluation.json` reports feature-family comparisons, Brier score, ROC AUC,
average precision, balanced accuracy, confusion matrices, a trial-bootstrap
Brier interval and evaluation on unseen paddle/depth configurations when both
outcomes are available. Classifier probabilities are not calibrated uncertainty
bounds. Small test sets remain unreliable. Do not retune from test results;
reserve new independent trials after changes.

### Where breaking occurs

Supply a JSON list of **independently measured tank Y in meters**, bound to the
same reviewed trial/wave IDs. Raw image onset pixels are not metric labels:

```json
[
  {"trial_id":"TRIAL_A", "wave_id":"wave_01", "y_m":1.23,
   "verified":true, "evidence":"Replace with the actual measurement method and uncertainty"}
]
```

Add `--locations ../outputs/analysis/measured_onset_locations.json` to
`train-predictor`. A regularized linear location model is fitted only to breaking
waves, with whole-trial splits and validation selection. It reports held-out MAE,
RMSE and a training-mean baseline. It requires at least two labeled breaking
trials per split as a software minimum; otherwise location remains unavailable.
It predicts location **conditional on breaking**, without clamping to the ramp.
An out-of-region prediction needs investigation, not cosmetic correction.

Apply a saved model to an early-feature CSV (same feature names/units plus
`trial_id`, `wave_id`, `latest_frame_time_s`, `exposure_time_sigma_s`):

```bash
python analyze_wave_trial.py predict --model-dir ../outputs/analysis/study_model01 --features ../outputs/analysis/new_early_features.csv --output-dir ../outputs/analysis/new_predictions01
```

This produces probabilities and conditional onset Y when a location model exists.
Model and input fingerprints are recorded. Use only prebreaking features from
comparable camera calibration, physical definitions, timing and experimental
conditions. No PINN, CFD or experimentally validated breaking model is claimed.
