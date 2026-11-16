"""Reproducible offline profile/measurement stages with immutable output folders."""
import hashlib
import csv
import json
from pathlib import Path

from .config import validate_config
from .plots import plot_series
from .profiles import build_profiles
from .reports import code_version, json_text, output_path, write_csv
from .tracking import reviewed_tracks
from .wave_features import measure_profiles


def digest(path):
    """Hash an exact input artifact for reproducibility."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, rows, empty_columns):
    """Write a tidy CSV even when nothing was measurable."""
    normalized = json.loads(json_text(rows))
    columns = list(dict.fromkeys(k for r in normalized for k in r)) or empty_columns
    write_csv(path, normalized, columns)


def export_profiles(output, documents, config, cal):
    """Append profile artifacts inside a newly allocated extraction directory."""
    profiles, exclusions = build_profiles(documents, config, cal)
    output = Path(output)
    (output/'physical_config.json').write_text(json_text(config))
    (output/'profiles.json').write_text(json_text(profiles))
    (output/'profile_quality.json').write_text(json_text(dict(exclusions=exclusions,
        frame_count=len(profiles), usable_point_count=sum(len(p['points']) for p in profiles),
        status='conditional_on_measured_configuration', exposure_uncertainty='not inferred from host gaps')))
    rows = [dict(pair_index=p['pair_index'],time_s=p['time_s'],coverage_fraction=p['coverage_fraction'],**point)
            for p in profiles for point in p['points']]
    table(output/'profiles.csv',rows,['pair_index','time_s','point_id','x_m','y_m','z_m','eta_m','segment'])
    folder=output/'plots'
    folder.mkdir()
    for p in profiles:
        segments = [[(r['y_m'],r['eta_m']) for r in p['points'] if r['segment']==s]
                    for s in sorted({r['segment'] for r in p['points']})]
        plot_series(folder/f'profile_{p["pair_index"]:05d}.svg',segments,
                    f'Pair {p["pair_index"]}: sparse profile, {p["time_s"]:.4f} s (conditional)',
                    'Tank Y (m)', 'Elevation relative to still water (m)')
    return dict(profile_count=len(profiles), usable_point_count=len(rows))


def measure_analysis(analysis_dir, output_dir=None, tracks_path=None):
    """Measure an extracted physical profile; keep all previous results intact."""
    source=Path(analysis_dir).resolve()
    manifest=json.loads((source/'analysis_manifest.json').read_text())
    if not (source/'profiles.json').is_file():
        raise ValueError('No physical profiles: run extract with a measured --config first')
    if set(manifest.get('profile_inputs_sha256',{})) != {'profiles.json','physical_config.json','reconstruction.json','breaking_events.csv'}:
        raise ValueError('Missing profile provenance; re-extract with --config')
    for filename, expected in manifest.get('profile_inputs_sha256', {}).items():
        if digest(source/filename)!=expected:
            raise ValueError(f'Extraction artifact changed: {filename}; re-extract to a new directory')
    config=validate_config(json.loads((source/'physical_config.json').read_text()),manifest['calibration_hashes'])
    profiles=json.loads((source/'profiles.json').read_text())
    features, slopes, quality=measure_profiles(profiles,config)
    links=[] if tracks_path is None else json.loads(Path(tracks_path).read_text())
    tracks=reviewed_tracks(profiles,features,links,config)
    output=output_path(manifest['source_session'],output_dir or source/'measurements')
    if output==source or output in source.parents:
        raise ValueError('Use a fresh measurement directory')
    output.mkdir(parents=True,exist_ok=False)
    table(output/'wave_features.csv',features,['pair_index','feature_id','height_m','wavelength_m','steepness','skewness','asymmetry'])
    table(output/'slopes.csv',slopes,['pair_index','point_id','y_m','slope'])
    table(output/'wave_tracks.csv',[{k:v for k,v in t.items() if k not in ('observations','rejected')} for t in tracks],
          ['track_id','kind','speed_m_s','conditional_speed_sigma_m_s','status'])
    with (source/'breaking_events.csv').open(newline='') as stream:
        events=list(csv.DictReader(stream))
    (output/'measurements.json').write_text(json_text(dict(config=config,features=features,slopes=slopes,tracks=tracks,
        profiles=profiles,breaking_events=events)))
    provenance=dict(source_analysis=str(source),source_session=manifest['source_session'],code=code_version(),
        recording_identity={name:meta['sha256'] for name,meta in manifest.get('source_files',{}).items()
                            if name in ('video/cam0.avi','video/cam1.avi','video/cam0_frames.csv','video/cam1_frames.csv')},
        input_hashes={name:digest(source/name) for name in ('profiles.json','physical_config.json','reconstruction.json','breaking_events.csv')},
        tracks_input=None if tracks_path is None else dict(path=str(Path(tracks_path).resolve()),sha256=digest(tracks_path)),
        definitions='Babanin 2007 Fig 1; adjacent-trough mean for irregular H/Sk; next crest along propagation for wavelength',
        units='m, s, dimensionless slope/steepness/Sk/As', quality=quality,
        experimental_validation=config['experimental_validation'])
    provenance['measurements_sha256']=digest(output/'measurements.json')
    (output/'analysis_manifest.json').write_text(json_text(provenance))
    template=[dict(track_id='REPLACE_WITH_REVIEWED_WAVE_ID',kind='crest',reviewed=False,evidence='',
        observations=[dict(pair_index=w['pair_index'],feature_id=w['feature_id']) for w in features])]
    (output/'track_links_template.json').write_text(json_text(template))
    plots=output/'plots'
    plots.mkdir()
    for key,label in (('height_m','Wave height (m)'),('steepness','Steepness pi H / wavelength'),
                      ('skewness','Geometric Sk'),('asymmetry','Geometric As')):
        series=[[(w['crest_y_m'],w[key]) for w in features if w['pair_index']==p['pair_index'] and w['segment']==segment and w[key] is not None]
                for p in profiles for segment in sorted({r['segment'] for r in p['points']})]
        plot_series(plots/f'{key}.svg',series,f'{config["trial_id"]}: {label} (conditional)','Tank Y (m)',label)
    for i,track in enumerate(tracks):
        plot_series(plots/f'track_{i:03d}.svg',[[(o['time_s'],o['y_m']) for o in track['observations']]],
                    f'{track["track_id"]}: reviewed {track["kind"]} trajectory','Actual time (s)','Tank Y (m)')
    return dict(output_dir=str(output),waves=len(features),tracks=len(tracks),status='conditional_measurements')
