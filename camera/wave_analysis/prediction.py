"""Small, auditable trial-held-out breaking and conditional-location models.

Synthetic runs exercise software only. All preprocessing fits training data;
validation selects models; test trials never select thresholds or features.
"""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .pipeline import digest, table
from .reports import code_version, json_text, output_path

FAMILIES = {
    'conditions': ['paddle_amplitude', 'paddle_sweep_angle_deg', 'driving_frequency_hz', 'still_water_depth_mm'],
    'steepness': ['steepness'],
    'depth': ['steepness', 'kh', 'H_over_h'],
    'shape': ['steepness', 'skewness', 'asymmetry'],
    'velocity': ['steepness', 'kh', 'H_over_h', 'skewness', 'asymmetry', 'causal_phase_speed_m_s'],
}
FEATURES = set(sum(FAMILIES.values(), []))


def numeric(value):
    """Parse a missing CSV value as NaN; reject nonfinite supplied values."""
    if value in ('', None): return np.nan
    try: result = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f'Invalid numeric value: {value}') from exc
    if not np.isfinite(result): raise ValueError('Supplied values must be finite')
    return result


def snapshots(rows, observation_time_s, lead_time_s, *, synthetic=False):
    """Choose one early snapshot per wave at a fixed trial-relative cutoff.

    A wave already breaking at that cutoff is excluded, not retrospectively
    shifted to an earlier frame. Unknown outcomes never become negative labels.
    """
    if not np.isfinite([observation_time_s, lead_time_s]).all() or min(observation_time_s, lead_time_s) < 0:
        raise ValueError('Cutoff and lead time must be finite and nonnegative')
    groups, assignments, excluded = {}, {}, Counter()
    for row in rows:
        trial, wave, split = row.get('trial_id'), row.get('wave_id'), row.get('split')
        if not trial or split not in ('train', 'validation', 'test'):
            raise ValueError('Assign every complete trial to train, validation or test')
        if trial in assignments and assignments[trial] != split:
            raise ValueError('Trial leakage: one trial occurs in multiple splits')
        assignments[trial] = split
        if not wave: excluded['unlinked_wave'] += 1; continue
        groups.setdefault((trial, wave), []).append(row)
    selected = []
    for key, values in groups.items():
        outcomes = {v.get('breaking_observed') for v in values}
        if len(outcomes) != 1 or not outcomes <= {'yes', 'no'}:
            excluded['unknown_or_conflicting_outcome'] += 1; continue
        outcome = next(iter(outcomes))
        if not synthetic and any(str(v.get('experimentally_validated')).lower() != 'true' for v in values):
            excluded['unvalidated_measurement'] += 1; continue
        onsets = {numeric(v.get('onset_time_s')) for v in values if v.get('onset_time_s') not in ('', None)}
        if outcome == 'yes' and (len(onsets) != 1 or next(iter(onsets)) <= observation_time_s+lead_time_s):
            excluded['already_breaking_or_insufficient_lead'] += 1; continue
        eligible = []
        for v in values:
            time = numeric(v.get('latest_frame_time_s'))
            uncertainty = numeric(v.get('exposure_time_sigma_s'))
            if not np.isfinite(time) or not np.isfinite(uncertainty) or uncertainty < 0:
                continue
            # Conservative timing allowance is a declared convention, not an exposure guarantee.
            if time+2*uncertainty <= observation_time_s and str(v.get('prebreaking_by_recorded_times')).lower() == 'true':
                eligible.append(v)
        if not eligible: excluded['no_eligible_early_snapshot'] += 1; continue
        selected.append(max(eligible, key=lambda v: numeric(v['latest_frame_time_s'])))
    return selected, dict(excluded)


def _matrix(rows, names):
    return np.array([[numeric(r.get(k)) for k in names] for r in rows], float)


def _preprocess(rows, names):
    x = _matrix(rows, names)
    names = [n for n, keep in zip(names, np.isfinite(x).any(axis=0)) if keep]
    if not names: return None
    x = _matrix(rows, names)
    median = np.nanmedian(x, axis=0); x = np.where(np.isnan(x), median, x)
    scale = x.std(axis=0); scale[scale < 1e-12] = 1
    return dict(features=names, median=median.tolist(), mean=x.mean(axis=0).tolist(), scale=scale.tolist())


def _transform(rows, model):
    x = _matrix(rows, model['features'])
    return (np.where(np.isnan(x), model['median'], x)-model['mean'])/model['scale']


def infer(rows, model):
    """Apply a portable JSON model; only whitelisted physical inputs are read."""
    if model.get('kind') not in ('logistic', 'ridge') or not model.get('features') or not set(model['features']) <= FEATURES:
        raise ValueError('Unsupported model or leakage-prone feature names')
    if not rows: return []
    n = len(model['features'])
    if any(len(model[k]) != n or not np.isfinite(model[k]).all() for k in ('median', 'mean', 'scale', 'coefficients')) or any(s <= 0 for s in model['scale']) or not np.isfinite(model['intercept']):
        raise ValueError('Malformed model coefficients or preprocessing')
    z = _transform(rows, model) @ np.asarray(model['coefficients']) + model['intercept']
    return (1/(1+np.exp(-np.clip(z, -700, 700)))).tolist() if model['kind'] == 'logistic' else z.tolist()


def _metrics(y, p, threshold=.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score, confusion_matrix
    predicted = np.asarray(p) >= threshold
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist(), threshold=threshold,
        sample_count=len(y))


def _fit(rows, names, y, regularization, kind):
    from sklearn.linear_model import LogisticRegression, Ridge
    preprocessing = _preprocess(rows, names)
    if preprocessing is None: return None
    counts = Counter(r['trial_id'] for r in rows)
    weights = np.array([1/counts[r['trial_id']] for r in rows]); weights *= len(rows)/weights.sum()
    estimator = LogisticRegression(C=regularization, max_iter=2000, random_state=0) if kind == 'logistic' else Ridge(alpha=regularization)
    estimator.fit(_transform(rows, preprocessing), y, sample_weight=weights)
    return dict(preprocessing, kind=kind, coefficients=np.asarray(estimator.coef_).ravel().tolist(),
                intercept=float(np.asarray(estimator.intercept_).ravel()[0]), regularization=regularization)


def train_prediction(dataset_dir, output_dir, *, observation_time_s, lead_time_s=.1, locations_path=None, synthetic=False):
    """Compare simple models with frozen trial splits and optional measured onset Y.

    Locations JSON: [{trial_id,wave_id,y_m,verified:true,evidence:...}].
    Location regression is conditional on breaking; it is not a prediction that
    a nonbreaking wave must eventually break. All artifacts are non-destructive.
    """
    source = Path(dataset_dir).resolve()
    csv_path = source/'research_dataset.csv'
    manifest = json.loads((source/'analysis_manifest.json').read_text())
    if manifest.get('dataset_sha256') != digest(csv_path):
        raise ValueError('Missing/changed dataset fingerprint; rebuild dataset before training')
    if synthetic and manifest.get('data_kind') != 'synthetic':
        raise ValueError('--synthetic-demo only accepts an explicitly synthetic dataset manifest')
    if not synthetic and manifest.get('data_kind') == 'synthetic':
        raise ValueError('Synthetic datasets require --synthetic-demo')
    with csv_path.open(newline='') as stream: original = list(csv.DictReader(stream))
    rows, exclusions = snapshots(original, observation_time_s, lead_time_s, synthetic=synthetic)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    for split, values in splits.items():
        if len({r['trial_id'] for r in values}) < 2 or {r['breaking_observed'] for r in values} != {'yes', 'no'}:
            raise ValueError(f'{split} needs at least two eligible independent trials and both outcomes; this is a software minimum, not sample adequacy')
    amplitude_units = {(r.get('paddle_amplitude_units'), r.get('paddle_amplitude_kind')) for r in rows if r.get('paddle_amplitude') not in ('', None)}
    if amplitude_units and (len(amplitude_units) != 1 or any(not all(v) for v in amplitude_units)):
        raise ValueError('Paddle amplitudes need consistent declared units and commanded/measured kind')
    labels = {s: np.array([r['breaking_observed'] == 'yes' for r in values], int) for s, values in splits.items()}
    models = []
    for family, names in FAMILIES.items():
        for regularization in (.1, 1., 10.):
            model = _fit(splits['train'], names, labels['train'], regularization, 'logistic')
            if model is None: continue
            score = _metrics(labels['validation'], infer(splits['validation'], model))
            models.append(dict(model, family=family, validation_brier=score['brier']))
    if not models: raise ValueError('No usable predictor features')
    best = min(models, key=lambda m: m['validation_brier'])
    comparison = []
    for family in FAMILIES:
        candidates = [m for m in models if m['family'] == family]
        if candidates:
            m = min(candidates, key=lambda v: v['validation_brier'])
            comparison.append(dict(family=family, features=m['features'], regularization=m['regularization'],
                validation=_metrics(labels['validation'], infer(splits['validation'], m)),
                test=_metrics(labels['test'], infer(splits['test'], m))))
    baseline = float(labels['train'].mean())
    baselines = {'training_prevalence': _metrics(labels['test'], [baseline]*len(splits['test']))}
    # A fixed classical reference is explicitly not a universal tank threshold.
    subset = [r for r in splits['test'] if np.isfinite(numeric(r.get('steepness')))]
    if {r['breaking_observed'] for r in subset} == {'yes', 'no'}:
        baselines['deep_water_0_44_demonstration'] = _metrics([int(r['breaking_observed']=='yes') for r in subset],
                                                           [float(numeric(r['steepness']) >= .44) for r in subset])
    location_model, location_report = None, dict(status='unavailable: need independently measured onset locations')
    if locations_path is not None:
        location_values = json.loads(Path(locations_path).read_text()); locations = {}
        if not isinstance(location_values, list): raise ValueError('Locations JSON must be a list')
        for v in location_values:
            key = (v['trial_id'], v['wave_id'])
            if key in locations or v.get('verified') is not True or not v.get('evidence', '').strip() or not np.isfinite(numeric(v.get('y_m'))):
                raise ValueError('Location labels require unique wave IDs, verified measured Y in meters and evidence')
            locations[key] = numeric(v['y_m'])
        location_rows = {s: [r for r in values if r['breaking_observed']=='yes' and (r['trial_id'], r['wave_id']) in locations] for s, values in splits.items()}
        if all(len({r['trial_id'] for r in values}) >= 2 for values in location_rows.values()):
            targets = {s: np.array([locations[(r['trial_id'], r['wave_id'])] for r in values]) for s, values in location_rows.items()}
            choices = [_fit(location_rows['train'], sorted(FEATURES), targets['train'], a, 'ridge') for a in (.1, 1., 10.)]
            choices = [m for m in choices if m is not None]
            if choices:
                location_model = min(choices, key=lambda m: np.mean(np.abs(np.asarray(infer(location_rows['validation'], m))-targets['validation'])))
                pred = np.asarray(infer(location_rows['test'], location_model))
                location_report = dict(status='conditional_on_breaking', test_mae_m=float(np.mean(np.abs(pred-targets['test']))),
                    test_rmse_m=float(np.sqrt(np.mean((pred-targets['test'])**2))),
                    training_mean_baseline_mae_m=float(np.mean(np.abs(targets['test']-targets['train'].mean()))),
                    trials={s: len({r['trial_id'] for r in v}) for s, v in location_rows.items()})
        else: location_report = dict(status='unavailable: need two labeled breaking trials per split; software minimum only')
    train_conditions = {(r.get('paddle_sweep_angle_deg'), r.get('driving_frequency_hz'), r.get('still_water_depth_mm')) for r in splits['train']}
    held_out = [r for r in splits['test'] if (r.get('paddle_sweep_angle_deg'), r.get('driving_frequency_hz'), r.get('still_water_depth_mm')) not in train_conditions]
    test_probabilities = np.asarray(infer(splits['test'], best))
    trial_ids = sorted({r['trial_id'] for r in splits['test']})
    rng = np.random.default_rng(37)
    bootstrap = []
    for _ in range(200):
        selected_trials = rng.choice(trial_ids, len(trial_ids), replace=True)
        indices = [i for trial in selected_trials for i, r in enumerate(splits['test']) if r['trial_id']==trial]
        bootstrap.append(float(np.mean((test_probabilities[indices]-labels['test'][indices])**2)))
    report = dict(test_brier_trial_bootstrap_95_interval=np.percentile(bootstrap, [2.5,97.5]).tolist(),
        amplitude_units=sorted(amplitude_units), status='SYNTHETIC SOFTWARE TEST ONLY' if synthetic else 'experimental model requires independent scientific assessment',
        selected_family=best['family'], observation_time_s=observation_time_s, lead_time_s=lead_time_s,
        exclusions=exclusions, comparison=comparison, baselines=baselines, location=location_report,
        split_trials={s: sorted({r['trial_id'] for r in v}) for s, v in splits.items()},
        unseen_condition_test=_metrics([int(r['breaking_observed']=='yes') for r in held_out], infer(held_out, best))
            if {r['breaking_observed'] for r in held_out} == {'yes','no'} else None,
        limitations=['Test results must not be used to retune this model; reserve new independent trials for further tuning.',
                    'Probabilities are not calibrated uncertainty bounds. Small sample metrics are not reliable evidence.',
                    'No PINN or multiphase breaking simulation; thresholds do not verify the Stokes limit.'])
    output = output_path(source, output_dir); output.mkdir(parents=True, exist_ok=False)
    (output/'model.json').write_text(json_text(dict(schema_version=1, classification=best, location=location_model,
        synthetic=synthetic, amplitude_units=sorted(amplitude_units), observation_time_s=observation_time_s, lead_time_s=lead_time_s)))
    (output/'evaluation.json').write_text(json_text(report))
    predictions = [dict(trial_id=r['trial_id'], wave_id=r['wave_id'], outcome=r['breaking_observed'], probability=p)
                   for r, p in zip(splits['test'], infer(splits['test'], best))]
    table(output/'test_predictions.csv', predictions, [])
    (output/'manifest.json').write_text(json_text(dict(code=code_version(), dataset_sha256=digest(csv_path),
        dataset_manifest_sha256=digest(source/'analysis_manifest.json'), model_sha256=digest(output/'model.json'),
        locations_sha256=None if locations_path is None else digest(locations_path), settings=report)))
    return dict(output_dir=str(output), selected_family=best['family'], location=location_report, status=report['status'])
