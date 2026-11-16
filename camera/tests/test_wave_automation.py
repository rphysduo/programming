"""Synthetic regression tests for temporal, image and predictive stages."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from wave_analysis.auto_tracking import propose_tracks, packet_candidates
from wave_analysis.suggestions import suggest_trace, evaluate_traces, suggest_document
from wave_analysis.prediction import snapshots, train_prediction, infer
from wave_analysis.pipeline import table, digest
from wave_analysis.annotation_web import create_app
from wave_analysis.annotations import empty_document, AnnotationStore
from wave_analysis.session_io import WaveSession
from test_wave_physics import configuration


def moving():
    profiles, features = [], []
    for i, t in enumerate([0, .11, .25, .39]):
        profiles.append(dict(pair_index=i, time_s=t, pair=dict(frame_indices=[i, i], host_gap_ms=1)))
        for j in range(2):
            features.append(dict(pair_index=i, feature_id=str(j), segment=0, crest_y_m=.3*j+.1*t,
                height_m=.02, wavelength_m=.3, crest_sigma_y_m=.001))
    return profiles, features


def test_tracking_uses_actual_times_multiple_waves_and_review():
    p, f = moving()
    result = propose_tracks(p, f, configuration(), max_speed_m_s=.2)
    assert len(result['tracks']) == 2
    for track in result['tracks']:
        assert track['provisional_fit']['speed_m_s'] == pytest.approx(.1)
        assert track['provisional_fit']['conditional_speed_sigma_m_s'] > 0
        assert not track['reviewed']
        assert track['provisional_fit']['intervals_s'] == pytest.approx([.11, .14, .14])


def test_tracking_reuse_gaps_missing_breaking_and_ambiguity():
    p, f = moving()
    p[1]['pair']['frame_indices'][0] = 0
    result = propose_tracks(p, f, configuration(), max_speed_m_s=.2)
    assert result['diagnostics'][0]['reason'] == 'shared_camera_frame'
    assert len(result['tracks'][0]['observations']) == 3
    p, f = moving()
    missing = [v for v in f if v['pair_index'] != 1]
    result = propose_tracks(p, missing, configuration(), max_speed_m_s=.2)
    assert 'bridged_missing_observation' in result['tracks'][0]['flags']
    result = propose_tracks(p, f, configuration(), max_speed_m_s=.2, breaking_times=[.2])
    assert len(result['tracks']) == 4
    assert 'breaking_boundary' in result['tracks'][0]['flags']
    result = propose_tracks(p, f, dict(configuration(), max_track_gap_s=.12), max_speed_m_s=.2)
    assert any('timing_gap' in t['flags'] for t in result['tracks'])
    f.append(dict(f[2], feature_id='ambiguous', crest_y_m=.012))
    result = propose_tracks(p, f, configuration(), max_speed_m_s=.2)
    assert 'ambiguous_merge_split_or_identity' in result['tracks'][0]['flags']


def test_packet_requires_distinct_resolved_envelope():
    features = [dict(pair_index=0, segment=0, feature_id=str(i), height_m=.02, crest_y_m=i*.1) for i in range(5)]
    assert packet_candidates(features) == []
    for f, h in zip(features, [.01, .02, .04, .02, .01]): f['height_m'] = h
    candidate = packet_candidates(features)[0]
    assert candidate['crest_y_m'] == .2
    assert candidate['crest_sigma_y_m'] == pytest.approx(.05)
    features[-1]['height_m'] = .08
    assert packet_candidates(features) == []


def edge_image():
    image = np.zeros((60, 100, 3), np.uint8)
    image[30:] = 200
    return image


def test_detector_masks_blank_images_ambiguity_and_coordinate_validation():
    seed = [[5, 29], [95, 29]]
    result = suggest_trace(edge_image(), seed)
    assert result['coverage_fraction'] > .9
    assert all(abs(p[1]-30) <= 1 for s in result['segments'] for p in s)
    masked = suggest_trace(edge_image(), seed, [[[35, 0], [65, 0], [65, 59], [35, 59]]])
    assert len(masked['segments']) == 2
    assert any(35 <= x <= 65 for x in masked['missing_columns'])
    assert suggest_trace(np.zeros((60, 100, 3), np.uint8), seed)['segments'] == []
    double = np.zeros((60, 100, 3), np.uint8); double[24:36] = 255
    assert suggest_trace(double, seed)['segments'] == []
    with pytest.raises(ValueError, match='without folding'):
        suggest_trace(edge_image(), [[5, 29], [50, 29], [20, 29]])
    with pytest.raises(ValueError): suggest_trace(edge_image(), seed, band_px=0)


def seed_document():
    doc = empty_document()
    doc['traces'] = [dict(id=c, camera=c, origin='human', confidence=.8,
        direction='unknown', vertices_px=[[5, 24], [55, 24]], label='rough band', notes='') for c in ('cam0', 'cam1')]
    return doc


def test_suggestion_route_is_non_destructive_and_requires_search_bands(wave_session, tmp_path):
    app = create_app(wave_session, tmp_path/'outputs'/'annotations').test_client()
    before = {str(p): digest(p) for p in wave_session.rglob('*') if p.is_file()}
    response = app.post('/api/suggest/0', json=dict(document=seed_document(), band_px=10))
    assert response.status_code == 200
    assert response.get_json()['points'] == []  # flat synthetic video must not invent matches
    assert app.post('/api/suggest/0', json=dict(document=empty_document())).status_code == 400
    assert app.get('/api/pairs/0').get_json()['document']['revision'] == 0
    assert before == {str(p): digest(p) for p in wave_session.rglob('*') if p.is_file()}


def test_trace_evaluation_measures_geometry_and_empty_coverage():
    truth = [[[0, 10], [100, 10]]]
    result = evaluate_traces(truth, [[[0, 12], [100, 12]]], tolerance_px=3)
    assert result['mean_error_px'] == 2
    assert result['coverage_within_tolerance'] == 1
    assert evaluate_traces(truth, [])['coverage_within_tolerance'] == 0
    assert evaluate_traces(truth, [[[0, 10], [40, 10]]])['coverage_within_tolerance'] < .5


def dataset_rows():
    rng = np.random.default_rng(19)
    rows, locations = [], []
    for split in ('train', 'validation', 'test'):
        for i in range(8):
            y = i % 2
            name = f'{split}_{i}'
            rows.append(dict(trial_id=name, wave_id='w', split=split,
                breaking_observed='yes' if y else 'no', onset_time_s=2 if y else None,
                experimentally_validated=False, prebreaking_by_recorded_times=True,
                latest_frame_time_s=.5, exposure_time_sigma_s=.002,
                steepness=.1+.25*y+rng.normal(0,.02), kh=1+i*.1, H_over_h=.1+.05*y,
                skewness=y*.2, asymmetry=-y*.2, causal_phase_speed_m_s=.2,
                paddle_sweep_angle_deg=10+20*y, driving_frequency_hz=1, still_water_depth_mm=150))
            if y: locations.append(dict(trial_id=name, wave_id='w', y_m=.8+.02*i,
                                        verified=True, evidence='Synthetic location target'))
    return rows, locations


def write_dataset(root):
    rows, locations = dataset_rows()
    source = root/'outputs'/'dataset'; source.mkdir(parents=True)
    table(source/'research_dataset.csv', rows, [])
    (source/'analysis_manifest.json').write_text(json.dumps(dict(data_kind='synthetic', dataset_sha256=digest(source/'research_dataset.csv'))))
    path = source/'locations.json'; path.write_text(json.dumps(locations))
    return source, path


def test_snapshots_never_leak_trials_or_post_breaking_and_unknown_labels():
    rows, _ = dataset_rows()
    assert len(snapshots(rows, 1, .1, synthetic=True)[0]) == 24
    assert snapshots(rows, 1, .1)[0] == []
    leak = dict(rows[0], split='test')
    with pytest.raises(ValueError, match='Trial leakage'): snapshots(rows+[leak], 1, .1, synthetic=True)
    rows[1]['onset_time_s'] = .9
    rows[2]['breaking_observed'] = 'unknown'
    rows[3]['latest_frame_time_s'] = 1.1
    selected, excluded = snapshots(rows, 1, .1, synthetic=True)
    assert len(selected) == 21
    assert excluded['already_breaking_or_insufficient_lead'] == 1
    # Must not select an earlier frame from a wave already breaking at cutoff.
    early = dict(rows[1], latest_frame_time_s=.01)
    assert not any(r['trial_id']==early['trial_id'] for r in snapshots(rows+[early], 1, .1, synthetic=True)[0])


def test_training_location_inference_and_no_overwrite(tmp_path):
    source, locations = write_dataset(tmp_path)
    output = tmp_path/'outputs'/'models'
    result = train_prediction(source, output, observation_time_s=1, locations_path=locations, synthetic=True)
    assert result['location']['status'] == 'conditional_on_breaking'
    model = json.loads((output/'model.json').read_text())
    rows, _ = dataset_rows()
    predictions = infer(rows, model['classification'])
    assert all(0 <= p <= 1 for p in predictions)
    assert np.mean(np.array(predictions)[1::2]) > np.mean(np.array(predictions)[::2])
    assert len(infer(rows, model['location'])) == len(rows)
    report = json.loads((output/'evaluation.json').read_text())
    assert len(report['comparison']) == 5
    assert report['comparison'][1]['test']['roc_auc'] > .9
    with pytest.raises(FileExistsError): train_prediction(source, output, observation_time_s=1, synthetic=True)
    with pytest.raises(ValueError, match='Synthetic datasets require'):
        train_prediction(source, tmp_path/'outputs'/'bad', observation_time_s=1)


def test_preprocessing_and_selection_do_not_read_test_values(tmp_path):
    source, _ = write_dataset(tmp_path)
    first = tmp_path/'outputs'/'first'; second = tmp_path/'outputs'/'second'
    train_prediction(source, first, observation_time_s=1, synthetic=True)
    rows, _ = dataset_rows()
    for row in rows:
        if row['split'] == 'test': row['steepness'] = 900
    replacement = source/'replacement.csv'; table(replacement, rows, [])
    replacement.replace(source/'research_dataset.csv')
    (source/'analysis_manifest.json').write_text(json.dumps(dict(data_kind='synthetic', dataset_sha256=digest(source/'research_dataset.csv'))))
    train_prediction(source, second, observation_time_s=1, synthetic=True)
    assert json.loads((first/'model.json').read_text()) == json.loads((second/'model.json').read_text())


def test_cli_tracking_evaluation_holdout_and_prediction(tmp_path, wave_session):
    from analyze_wave_trial import main
    session = WaveSession(wave_session)
    reference = tmp_path/'outputs'/'manual'
    store = AnnotationStore(session, reference)
    for index in range(len(session.pairs)): store.save(index, seed_document(), 0)
    proposed = tmp_path/'outputs'/'proposals'
    assert main(['suggest', '--session', str(wave_session), '--annotations', str(reference), '--pairs', '0', '2', '--output-dir', str(proposed)]) == 0
    assert main(['evaluate-suggestions', '--session', str(wave_session), '--reference', str(reference), '--suggestions', str(proposed),
                 '--pairs', '2', '--tuning-pairs', '0', '--output-dir', str(tmp_path/'outputs'/'heldout')]) == 0
    assert main(['evaluate-suggestions', '--session', str(wave_session), '--reference', str(reference), '--suggestions', str(proposed),
                 '--pairs', '0', '--tuning-pairs', '0', '--output-dir', str(tmp_path/'outputs'/'overlap')]) == 1
    source, _ = write_dataset(tmp_path)
    model = tmp_path/'outputs'/'model'
    assert main(['train-predictor', '--dataset', str(source), '--output-dir', str(model), '--observation-time-s', '1', '--synthetic-demo']) == 0
    assert main(['predict', '--model-dir', str(model), '--features', str(source/'research_dataset.csv'), '--output-dir', str(tmp_path/'outputs'/'predictions')]) == 0


def test_patch_suggestions_keep_geometry_separate_from_human_identity(wave_session):
    from wave_analysis.suggestions import suggest_matches
    session = WaveSession(wave_session)
    rng = np.random.default_rng(18)
    image = rng.integers(0, 256, (48,64,3), dtype=np.uint8)
    other = np.zeros_like(image); other[:, :-5] = image[:, 5:]
    points = [[20, 15], [40, 25], [50, 35]]
    traces = dict(cam0=[points], cam1=[[[p[0]-5, p[1]] for p in points]])
    matches = suggest_matches([image, other], traces, session.cal)
    assert len(matches) == 3
    assert all(m['status']=='uncertain' and not m['human_reviewed'] for m in matches)
    assert all(m['suggestion_geometry']['valid'] for m in matches)
    assert all(m['positional_uncertainty_mm'] is None for m in matches)


def test_packet_trajectory_is_independent_of_crest_speed():
    p, _ = moving()
    features=[]
    for profile in p:
        for i, height in enumerate([.01,.02,.04,.02,.01]):
            features.append(dict(pair_index=profile['pair_index'], segment=0, feature_id=str(i),
                height_m=height, crest_y_m=i*.1+.07*profile['time_s'], wavelength_m=.1))
    result=propose_tracks(p, features, configuration(), max_speed_m_s=.15, kind='packet')
    assert len(result['tracks'])==1
    assert result['tracks'][0]['provisional_fit']['speed_m_s']==pytest.approx(.07)
    assert result['tracks'][0]['distinct_envelope'] is False


def test_cli_track_emits_reviewable_proposals(tmp_path):
    from analyze_wave_trial import main
    p, f = moving(); source=tmp_path/'outputs'/'measurements'; source.mkdir(parents=True)
    (source/'measurements.json').write_text(json.dumps(dict(profiles=p, features=f, config=configuration(), breaking_events=[])))
    (source/'analysis_manifest.json').write_text(json.dumps(dict(measurements_sha256=digest(source/'measurements.json'))))
    output=tmp_path/'outputs'/'tracks'
    assert main(['track','--measurements',str(source),'--output-dir',str(output),'--max-speed-m-s','.2'])==0
    assert len(json.loads((output/'track_links.json').read_text()))==2
    assert (output/'trajectories.svg').is_file()


def test_synthetic_complete_demo(tmp_path):
    from demo_wave_automation import run
    result=run(tmp_path/'outputs'/'demo')
    assert result['automatic_tracks'] >= 1
    assert result['prediction']['location']['status']=='conditional_on_breaking'
    assert (tmp_path/'outputs/demo/synthetic_edge_proposals.png').is_file()


def test_dataset_exports_past_only_velocity(tmp_path):
    from wave_analysis.dataset import build_dataset
    source=tmp_path/'outputs'/'measured'; source.mkdir(parents=True)
    config=configuration()
    profile=dict(pair_index=2,time_s=.2,pair=dict(cameras={c:dict(elapsed_ns=200_000_000) for c in ('cam0','cam1')}))
    observations=[dict(pair_index=i,feature_id='c',time_s=t,y_m=y,sigma_y_m=.001)
                  for i,(t,y) in enumerate([(0,0),(.1,.01),(.2,.02),(.3,99)])]
    data=dict(config=config,profiles=[profile],features=[dict(pair_index=2,feature_id='c',height_m=.01,crest_y_m=.02)],
              tracks=[dict(kind='crest',track_id='w',speed_m_s=99,observations=observations)],breaking_events=[])
    (source/'measurements.json').write_text(json.dumps(data))
    (source/'analysis_manifest.json').write_text(json.dumps(dict(source_session=str(tmp_path/'recording'),measurements_sha256=digest(source/'measurements.json'))))
    output=tmp_path/'outputs'/'joined'
    build_dataset([source],output)
    import csv
    with (output/'research_dataset.csv').open() as stream: row=next(csv.DictReader(stream))
    assert float(row['phase_speed_m_s'])==99
    assert float(row['causal_phase_speed_m_s'])==pytest.approx(.1)
    assert float(row['latest_frame_time_s'])==.2


def test_prediction_rejects_mixed_amplitude_units(tmp_path):
    source,_=write_dataset(tmp_path)
    rows,_=dataset_rows()
    for i,r in enumerate(rows): r.update(paddle_amplitude=10,paddle_amplitude_units='mm' if i%2 else 'degrees',paddle_amplitude_kind='commanded')
    temporary=source/'new.csv';table(temporary,rows,[]);temporary.replace(source/'research_dataset.csv')
    (source/'analysis_manifest.json').write_text(json.dumps(dict(data_kind='synthetic',dataset_sha256=digest(source/'research_dataset.csv'))))
    with pytest.raises(ValueError,match='consistent declared units'):
        train_prediction(source,tmp_path/'outputs'/'bad',observation_time_s=1,synthetic=True)
