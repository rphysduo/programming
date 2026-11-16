# Profiles, wave measurements and reviewed tracking

The annotation interface and its storage format remain compatible. Researchers
can continue annotating while these offline stages are developed. This pipeline
reuses NumPy, OpenCV and Flask from the camera tools. Offline plots additionally
use Matplotlib with its headless Agg canvas. No display server or Node is needed.

## What is implemented

* Phase 2: explicit board-to-tank transformation, slice selection, sparse
  elevation profiles, gap/coverage reports, conditional position uncertainty,
  CSV/JSON and standalone profile plots.
* Phase 3: sampled local slopes, crest/trough height, next-crest wavelength,
  steepness, geometric skewness/asymmetry, measured local depth, relative
  depth/height and finite-depth linear speed references.
* Phase 4: reviewed crest and packet associations, signed trajectory speeds,
  actual sampling intervals, shared-frame exclusions and conditional uncertainty.
  Automatic motion/shape proposals and sparse packet-envelope proposals are
  available; ambiguous identities remain subject to human review.
* Phase 6: joined trial/wave observations and settings, conservative reviewed
  outcomes, separate packet tracks, input hashes and whole-trial split checks.
* Phases 5 and 7: edge/patch proposals, held-out contour evaluation and simple
  whole-trial prediction/location models are implemented. See
  [AUTOMATION_AND_PREDICTION.md](AUTOMATION_AND_PREDICTION.md) for commands and
  the distinction between synthetic testing and real experimental validation.

These are numerically tested tools. No new experimental accuracy claim is made.
The synthetic example does not establish the behavior of actual tank water.

## Lab commands

Run from `~/programming` in the existing camera Python environment after the
new code has been installed. These changes may be local until explicitly pushed.

```bash
python -m pip install -r camera/wave_analysis/requirements.txt
SESSION=outputs/wave_trials/wave_20261110_140220_415247
python camera/analyze_wave_trial.py init-config \
  --session "$SESSION" --output outputs/analysis/configs/trial01.json
```

This writes an **incomplete** template containing the session calibration hashes.
Do not replace missing measurements with demonstration numbers. Complete it with:

* `board_to_tank_mm`: measured 4x4 rigid transform, **tank = R board + t**,
  translations in mm. Tank +Z is physically up, +Y longitudinal and X lateral.
  R must be a proper rotation (determinant +1), without scaling. Flipping Z alone
  is a reflection and is rejected. Record measurement evidence in `frame_evidence`
  and set `frame_verified` only after checking known directions/heights.
* `still_water_z_mm`: measured still-water elevation in that tank frame, and its
  uncertainty if known. It is not assumed equal to the board or tank bottom.
* `slice_x_mm`, `slice_halfwidth_mm` and `y_range_mm`: intended observed region.
* `max_gap_mm`: largest neighboring-point separation you consider supported.
  Larger gaps split the profile. No missing region is automatically filled.
* `max_host_gap_ms`: explicit lab acceptance policy. This gate does not measure
  exposure synchronization. `exposure_time_sigma_s` remains null until supported.
* `propagation_direction`: +1 or -1 relative to tank Y.
* `min_wave_height_mm`, `min_samples_per_wave`, `smoothing_window`: analysis
  choices to review against resolution and noise. Minimum samples defaults to 9;
  smoothing defaults to 1 (none). These are analysis settings, not breaking limits.
* `shape_reference`: defaults to `local_wave_mean`, the spatial mean over the
  bracketing trough-to-trough interval for geometric Sk/As. `still_water` is an
  explicit alternative and is labeled in every result. Neither changes XYZ or eta.
* `max_track_gap_s`: largest interval allowed within a reviewed track segment.
* Optional `pixel_sigma_px`, `systematic_position_sigma_mm`,
  `still_water_sigma_mm`, `exposure_time_sigma_s`, and `uncertainty_evidence`.
  Missing uncertainty stays null. Confidence labels are not converted to mm.
* Experimental settings, units, and commanded versus measured paddle amplitude.
  Supply either measured bottom Y/Z points or constant still-water depth. A
  constant depth is not extended beyond a configured ramp start without bottom
  measurements. No bottom extrapolation is performed.
* `experimental_validation`: leave false until known lengths/heights through
  the actual optical path establish measurement performance. Document evidence.

Generate profiles from the annotations you have already saved:

```bash
python camera/analyze_wave_trial.py extract \
  --session "$SESSION" \
  --from-annotations outputs/analysis/trial01/annotations \
  --config outputs/analysis/configs/trial01.json \
  --output-dir outputs/analysis/trial01/profiles01

python camera/analyze_wave_trial.py measure \
  --analysis-dir outputs/analysis/trial01/profiles01
```

Open `profiles01/plots/profile_*.png` or `.svg` for elevation versus longitudinal position.
Each gap-separated segment is drawn independently. The original unsmoothed
measurements remain in `profiles.csv`; rejected observations and reasons remain
in `profile_quality.json` and the original diagnostic reconstruction.

`measure` creates a fresh `profiles01/measurements/` containing
`wave_features.csv`, `slopes.csv`, `wave_tracks.csv`, `measurements.json`,
`analysis_manifest.json`, `track_links_template.json` and `plots/`. Empty tables
retain headers. One isolated crest can have height/shape but wavelength and
steepness remain unavailable without its next supported crest.

Choose a new extraction/measurement name each time annotations or settings change.
Existing output folders are never replaced. Extraction input hashes are checked
before measurements. Running `extract` without `--config` retains the original
board-XYZ diagnostic workflow.

## Reviewed velocity measurements

Inspect successive frames and the generated feature IDs. Make a separate track
JSON file under outputs. The generated template is **not an association result**;
it lists candidates and must be reorganized into independently reviewed tracks.

```json
[
  {
    "track_id": "wave01",
    "kind": "crest",
    "reviewed": true,
    "evidence": "Describe how this crest was identified across these frames",
    "observations": [
      {"pair_index": 0, "feature_id": "s0_p20"},
      {"pair_index": 1, "feature_id": "s0_p22"},
      {"pair_index": 2, "feature_id": "s0_p24"}
    ]
  }
]
```

Replace every example ID with actual exported IDs. A track needs at least three
independent pairs to estimate speed. If either camera frame is reused, the later
observation is excluded and reported. End a track at lost identity, a merge,
split or breaking; do not bridge ambiguous observations just because they are
nearby. The fit uses both cameras' original midpoint host times; it cannot
recover unknown exposure times.

For group speed, a `kind: "packet"` track additionally requires
`distinct_envelope: true`. Each observation contains `pair_index`, measured
`y_m`, optional `sigma_y_m`, and evidence of how the envelope position was
measured. This is a reviewed packet landmark input, not automatic envelope
extraction. A monochromatic train without a distinct packet has no measured
group speed. Packet-to-crest membership is not inferred.

```bash
python camera/analyze_wave_trial.py measure \
  --analysis-dir outputs/analysis/trial01/profiles01 \
  --tracks outputs/analysis/trial01/reviewed_tracks.json \
  --output-dir outputs/analysis/trial01/measurements_with_tracks01

python camera/analyze_wave_trial.py dataset \
  --analyses outputs/analysis/trial01/measurements_with_tracks01 \
  --output-dir outputs/analysis/research_dataset01
```

Use the same explicit wave ID in breaking labels and reviewed crest tracks to
join outcomes. Trial-wide labels are not assigned to every crest automatically.
Conflicting labels, incomplete observation and unlinked waves remain unknown.
The dataset preserves post-onset rows for diagnosis and marks whether each is
prebreaking by the recorded times; no predictor is fitted from these rows.

## Scientific definitions and limits

Reference: [Babanin et al. (2007), Figure 1 and equations 1–2](https://doi.org/10.1029/2006GL029135).
The original figure was visually checked. We use Sk = a1/a2 - 1 and
As = b_front/b_rear - 1, with the two widths measured from the crest's horizontal
position to its neighboring reference-level crossings. The default reference is
the local spatial wave mean, corresponding to the paper's mean-level definition;
the estimated mean and reference choice are exported. Forward leaning has negative
As. These are geometric quantities, not statistical moment or Hilbert asymmetry.
For irregular waves, a2 is explicitly the mean depth of the two neighboring
troughs; H = a1+a2. This averaging is our documented convention. Wavelength is
distance to the next eligible crest along propagation within the same supported
segment. An unresolved intervening peak makes wavelength unavailable; it is not
silently skipped. Steepness = pi H/lambda. Front and rear slopes are sampled maximum
absolute derivatives between crest and trough. Values are in meters and seconds.

Optional smoothing uses a local linear fit on measured nonuniform Y coordinates;
it never spans a missing region. Both raw and used elevations are exported,
along with the change in measured height. Noise can create or obscure extrema.
The minimum height/sample settings must be tested against annotation resolution.
No single-valued profile can represent overturning or disconnected spray.

Position covariance propagates independent raw-pixel noise through existing
triangulation. Explicit systematic-position and still-water uncertainties are
combined conditionally. These approximations do not estimate calibration bias,
refraction, correlations or exposure skew from reprojection RMS. Crest sampling
halfwidth is reported; its uniform-distribution approximation contributes to
crest-position and wavelength uncertainty. Slope/height errors assume independent
point uncertainties and are unavailable after smoothing. Shape-ratio uncertainty
is currently null. Speed regression residual error is separate from conditional
position/timing error; unknown exposure uncertainty leaves the latter null.
Do not interpret a zero synthetic residual as exact experimental speed.

Finite-depth linear cp/cg are reference columns only. There is no automatic
breaking threshold. Comparisons with the deep-water Stokes limit require lab
measurements and interpretation of finite depth and ramp shoaling.

## Remaining data-dependent stages

`dataset` writes `readiness.json`. Optional `--split-policy path.json` accepts:

```json
{
  "splits": {"trialA": "train", "trialB": "validation", "trialC": "test"},
  "min_trials_per_split": 1,
  "rationale": "Replace this demonstration policy with your study design"
}
```

One trial appears in exactly one split. The example minimum is a software
demonstration, not an adequate experimental sample size. Readiness requires
reviewed outcomes of both classes, independent measurement validation and timing
uncertainty evidence. Passing structural checks requires study-design review.
The `train-predictor` command implements simple model comparisons; useful
experimental results still require reviewed independent data. Breaking-location
regression accepts separately measured tank Y labels, never raw onset pixels.
See the automation guide for all new commands.

## Synthetic demonstration and tests

```bash
python camera/demo_wave_analysis.py --output-dir outputs/analysis/synthetic_demo01
python -m pytest
python camera/validate_stereo_3d.py selftest
```

The demo creates known synthetic traveling-wave XYZ, profile plots, wave
measurements and a known-identity crest trajectory. It is not image detection or
experimental validation. A separate integration test projects known XYZ into two
synthetic cameras and exercises the saved-annotation-to-profile pipeline.
