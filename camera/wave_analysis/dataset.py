"""Trial-level exports and explicit readiness checks; no premature model fitting."""
import json
from pathlib import Path

from .pipeline import digest, table
from .reports import code_version, json_text, output_path
from .plots import plot_series
from .tracking import fit_track


def reviewed_outcome(events, wave_id):
    """Resolve an explicit wave ID; retain unknown or contradictory outcomes."""
    relevant=[e for e in events if e['wave_id']==wave_id and e['observed']!='unknown']
    states={e['observed'] for e in relevant}
    if len(states)!=1:
        return dict(breaking_observed='unknown',onset_time_s=None,
                    outcome_reason='conflicting_labels' if len(states)>1 else 'no_reviewed_wave_outcome')
    state=next(iter(states))
    if state=='no' and not all(str(e['observation_complete']).lower()=='true' for e in relevant):
        return dict(breaking_observed='unknown',onset_time_s=None,outcome_reason='incomplete_visibility')
    times=[int(e['elapsed_ns'])/1e9 for e in relevant if e.get('elapsed_ns') not in ('',None)]
    return dict(breaking_observed=state,onset_time_s=min(times) if state=='yes' and times else None,
                outcome_reason='reviewed_wave_label')


def build_dataset(analysis_dirs, output_dir, split_policy=None):
    """Join reviewed wave tracks, conditions and outcomes without trial leakage.

    split_policy is optional: {splits:{trial_id:train|validation|test},
    min_trials_per_split:N, rationale:...}. N is a planning requirement supplied
    by researchers, not a scientific sample-size guarantee. This command never
    trains a detector or predictor, even if a readiness check passes.
    """
    datasets, rows, trials, sources, sessions = [], [], {}, [], []
    recording_identities=set()
    for directory in analysis_dirs:
        directory=Path(directory).resolve()
        manifest=json.loads((directory/'analysis_manifest.json').read_text())
        identity=json.dumps(manifest['recording_identity'],sort_keys=True) if manifest.get('recording_identity') else str(Path(manifest['source_session']).resolve())
        if identity in recording_identities:
            raise ValueError('Use one measurement version per independent trial; repeated source recording detected')
        recording_identities.add(identity)
        sessions.append(manifest['source_session'])
        if digest(directory/'measurements.json')!=manifest['measurements_sha256']:
            raise ValueError('Measurement artifact changed; rerun measurement')
        data=json.loads((directory/'measurements.json').read_text())
        config=data['config']
        trial=config['trial_id']
        if trial in trials:
            raise ValueError('Use one measurement version per independent trial')
        trials[trial]=dict(validated=config['experimental_validation']['verified'],
            evidence=config['experimental_validation']['evidence'],
            profiles=len(data['profiles']),features=len(data['features']))
        sources.append(dict(directory=str(directory),sha256=digest(directory/'measurements.json')))
        datasets.append(data)
        tracks={}
        for track in data['tracks']:
            if track['kind']=='crest':
                for obs in track['observations']:
                    tracks[(obs['pair_index'],obs['feature_id'])]=track
        profile_lookup={p['pair_index']:p for p in data['profiles']}
        for feature in data['features']:
            track=tracks.get((feature['pair_index'],feature['feature_id']))
            wave_id=None if track is None else track['track_id']
            outcome=reviewed_outcome(data['breaking_events'],wave_id)
            profile=profile_lookup[feature['pair_index']]
            latest=max(profile['pair']['cameras'][c]['elapsed_ns'] for c in ('cam0','cam1'))/1e9
            prebreaking=None
            if outcome['breaking_observed']=='yes' and outcome['onset_time_s'] is not None:
                prebreaking=latest<outcome['onset_time_s']
            elif outcome['breaking_observed']=='no':
                prebreaking=True
            causal = [] if track is None else [o for o in track['observations'] if o.get('time_s') is not None and o['time_s'] <= profile.get('time_s', -1)]
            causal_speed = fit_track(causal, config['exposure_time_sigma_s'])['speed_m_s']
            condition_keys=('paddle_sweep_angle_deg','paddle_amplitude','paddle_amplitude_units','paddle_amplitude_kind',
                            'driving_frequency_hz','still_water_depth_mm','ramp_start_y_mm')
            rows.append(dict(trial_id=trial,wave_id=wave_id,**feature,**outcome,
                **{key:config[key] for key in condition_keys},
                phase_speed_m_s=None if track is None else track['speed_m_s'],
                latest_frame_time_s=latest, causal_phase_speed_m_s=causal_speed,
                group_speed_m_s=None,  # Packet-to-wave association is not inferred.
                experimentally_validated=config['experimental_validation']['verified'],
                prebreaking_by_recorded_times=prebreaking,
                exposure_time_sigma_s=config['exposure_time_sigma_s'],
                split=None if split_policy is None else split_policy.get('splits',{}).get(trial)))
    if not datasets:
        raise ValueError('Supply at least one measurement directory')
    reasons=[]
    if split_policy is None:
        reasons.append('No independently assigned whole-trial train/validation/test splits')
    else:
        assignments=split_policy.get('splits',{})
        minimum=split_policy.get('min_trials_per_split')
        if type(minimum) is not int or minimum<1 or not split_policy.get('rationale','').strip():
            raise ValueError('Split policy requires positive min_trials_per_split and rationale')
        if set(assignments)!=set(trials) or any(v not in ('train','validation','test') for v in assignments.values()):
            raise ValueError('Assign each supplied trial exactly once to train, validation or test')
        for split in ('train','validation','test'):
            if sum(v==split for v in assignments.values())<minimum:
                reasons.append(f'{split} has fewer independent trials than the declared policy')
    eligible=[r for r in rows if r['experimentally_validated'] and r['wave_id'] and
              r['prebreaking_by_recorded_times'] is True and r['exposure_time_sigma_s'] is not None]
    if not all(t['validated'] for t in trials.values()):
        reasons.append('Some trials lack independent experimental measurement validation')
    if not eligible:
        reasons.append('No validated, linked prebreaking observations with documented timing uncertainty')
    if {r['breaking_observed'] for r in eligible}!={'yes','no'}:
        reasons.append('Both reviewed breaking and nonbreaking outcomes are needed')
    if split_policy:
        for split in ('train','validation','test'):
            subset=[r for r in eligible if r['split']==split]
            if len({r['trial_id'] for r in subset})<split_policy['min_trials_per_split']:
                reasons.append(f'{split} lacks the declared number of eligible independent trials')
            if {r['breaking_observed'] for r in subset}!={'yes','no'}:
                reasons.append(f'{split} lacks eligible observations of both outcomes')
    readiness=dict(prediction=dict(ready_for_study_design_review=not reasons,reasons=reasons,
        model_trained=False,note='Passing these checks is not proof of sample-size adequacy or predictive skill'),
        automatic_detection=dict(ready_for_review=any(t['validated'] and t['profiles']>=5 and t['features'] for t in trials.values()),
            implemented=True,remaining='Evaluate edge/patch proposals on held-out manual frames'),
        breaking_location=dict(implemented=True,reason='Prediction requires separately measured onset tank Y; raw image pixels are not metric labels'))
    output=output_path(sessions[0], output_dir)
    for session in sessions[1:]:
        output_path(session, output)
    # Never write into an input measurement directory or its parents.
    if any(output==Path(d).resolve() or Path(d).resolve() in output.parents or output in Path(d).resolve().parents for d in analysis_dirs):
        raise ValueError('Dataset output must be separate from input analyses')
    output.mkdir(parents=True,exist_ok=False)
    table(output/'research_dataset.csv',rows,['trial_id','wave_id','pair_index','height_m','breaking_observed'])
    table(output/'packet_tracks.csv',[dict(trial_id=d['config']['trial_id'],**t) for d in datasets for t in d['tracks'] if t['kind']=='packet'],
          ['trial_id','track_id','speed_m_s'])
    table(output/'breaking_events.csv',[dict(trial_id=d['config']['trial_id'],**e) for d in datasets for e in d['breaking_events']],
          ['trial_id','wave_id','observed','elapsed_ns'])
    plots=output/'plots'
    plots.mkdir()
    for key,label in (('height_m','Wave height (m)'),('steepness','Steepness'),('skewness','Geometric Sk'),
                      ('asymmetry','Geometric As'),('phase_speed_m_s','Reviewed phase speed (m/s)')):
        states=('yes','no','unknown')
        series=[[(r['crest_y_m'],r[key]) for r in rows if r['breaking_observed']==state
                  and r.get('crest_y_m') is not None and r.get(key) is not None] for state in states]
        plot_series(plots/f'{key}_by_outcome.svg',series,f'{label} by reviewed outcome (conditional)',
                    'Tank Y (m)',label,connect=False,labels=['Breaking','Nonbreaking','Unknown'])
    (output/'readiness.json').write_text(json_text(readiness))
    (output/'analysis_manifest.json').write_text(json_text(dict(code=code_version(),sources=sources,trials=trials,split_policy=split_policy,dataset_sha256=digest(output/'research_dataset.csv'))))
    return dict(output_dir=str(output),observations=len(rows),trials=len(trials),readiness=readiness)
