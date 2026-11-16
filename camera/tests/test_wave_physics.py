"""Known synthetic geometry and wave signals; no claim of tank accuracy."""
from copy import deepcopy
import json

import numpy as np
import pytest

from wave_analysis.config import template, validate_config
from wave_analysis.profiles import build_profiles
from wave_analysis.wave_features import measure_segment, linear_reference
from wave_analysis.tracking import fit_track, reviewed_tracks
from wave_analysis.annotations import AnnotationStore, empty_document
from wave_analysis.session_io import WaveSession
from wave_analysis.stereo import extract_annotations
from wave_analysis.pipeline import measure_analysis
from wave_analysis.dataset import build_dataset, reviewed_outcome


def configuration(hashes=None):
    c=template()
    c.update(trial_id='synthetic',calibration_hashes=hashes or {'test':'synthetic'},frame_verified=True,
        frame_evidence='Synthetic known frame, not a physical tank',board_to_tank_mm=np.eye(4).tolist(),
        still_water_z_mm=0,still_water_sigma_mm=0,slice_x_mm=0,slice_halfwidth_mm=2,
        y_range_mm=[-100,2100],max_gap_mm=20,max_host_gap_ms=50,propagation_direction=1,
        min_wave_height_mm=1,min_samples_per_wave=9,max_track_gap_s=.5,smoothing_window=1,
        still_water_depth_mm=200,pixel_sigma_px=.001,systematic_position_sigma_mm=0,
        exposure_time_sigma_s=.001,uncertainty_evidence='Synthetic independent noise assumption')
    return validate_config(c,hashes)


def sampled(y,eta):
    return [dict(point_id=str(i),y_m=float(x),eta_m=float(z),sigma_eta_m=.0001,
                 sigma_y_m=.0001,segment=0) for i,(x,z) in enumerate(zip(y,eta))]


def test_configuration_does_not_invent_physical_frame():
    with pytest.raises(ValueError):
        validate_config(template())
    for transform in (np.diag([1,1,-1,1]),np.diag([1.1,1,1,1])):
        c=configuration()
        c['board_to_tank_mm']=transform.tolist()
        with pytest.raises(ValueError,match='rigid'):
            validate_config(c)
    with pytest.raises(ValueError,match='hashes'):
        validate_config(configuration(),{'test':'changed'})


def test_flat_sine_steepness_depth_and_units():
    c=configuration()
    y=np.linspace(0,2,401)
    points=sampled(y,.01*np.cos(2*np.pi*y/.5))
    waves,slopes=measure_segment(points,c)
    assert len(waves)==3
    for wave in waves:
        assert wave['height_m']==pytest.approx(.02)
        assert wave['skewness']==pytest.approx(0,abs=1e-12)
        assert wave['asymmetry']==pytest.approx(0,abs=1e-12)
        assert wave['H_over_h']==pytest.approx(.1)
    assert waves[0]['wavelength_m']==pytest.approx(.5)
    assert waves[0]['steepness']==pytest.approx(np.pi*.02/.5)
    assert waves[-1]['wavelength_m'] is None
    assert max(abs(s['slope']) for s in slopes)==pytest.approx(.01*2*np.pi/.5,rel=.002)
    assert measure_segment(sampled(y,np.zeros_like(y)),c)[0]==[]
    assert measure_segment(points[:2],c)==([],[])


def test_skewed_and_forward_leaning_waves_and_direction():
    c=configuration()
    y=np.linspace(0,2,401)
    theta=2*np.pi*y/.5
    skewed=.01*(np.cos(theta)+.2*np.cos(2*theta))
    waves,_=measure_segment(sampled(y,skewed),c)
    assert waves[0]['skewness']==pytest.approx(.5)
    # Piecewise triangular crest: zero crossings .25 and .75, crest .60.
    y=np.linspace(-1,2,601)
    eta=np.interp(y,[-1,-.5,0,.6,1,1.5,2],[.01,-.01,-.01,.01,-.01,.01,-.01])
    waves,_=measure_segment(sampled(y,eta),c)
    assert waves
    target=min(waves,key=lambda w:abs(w['crest_y_m']-.6))
    assert target['asymmetry']<0
    reversed_c=dict(c,propagation_direction=-1)
    reversed_waves,_=measure_segment(sampled(-y[::-1],eta[::-1]),reversed_c)
    reverse_target=min(reversed_waves,key=lambda w:abs(w['crest_y_m']+.6))
    assert target['asymmetry']==pytest.approx(reverse_target['asymmetry'])


def test_shape_mean_does_not_change_still_water_elevation():
    c=configuration()
    y=np.linspace(0,2,401)
    waves,_=measure_segment(sampled(y,.003+.01*np.cos(2*np.pi*y/.5)),c)
    assert waves[0]['shape_reference_m']==pytest.approx(.003)
    assert waves[0]['crest_eta_m']==pytest.approx(.013)
    assert waves[0]['skewness']==pytest.approx(0,abs=1e-12)
    c['shape_reference']='still_water'
    waves,_=measure_segment(sampled(y,.003+.01*np.cos(2*np.pi*y/.5)),c)
    assert waves[0]['skewness']==pytest.approx(.013/.007-1)


def test_wavelength_does_not_skip_an_unresolved_intervening_crest():
    c=configuration()
    c['min_wave_height_mm']=15
    y=np.linspace(0,4,401)
    eta=np.interp(y,np.arange(0,4.01,.5),[.01,-.01,.01,-.01,-.009,-.01,.01,-.01,.01])
    waves,_=measure_segment(sampled(y,eta),c)
    assert len(waves)==2
    assert waves[0]['wavelength_m'] is None
    assert 'unresolved_intervening_crest' in waves[0]['flags']


def test_noise_smoothing_and_missing_depth_are_explicit():
    c=configuration()
    y=np.linspace(0,2,401)
    eta=.01*np.cos(2*np.pi*y/.5)+np.random.default_rng(4).normal(0,.0001,len(y))
    c.update(smoothing_window=5,ramp_start_y_mm=0)
    waves,_=measure_segment(sampled(y,eta),c)
    assert waves
    assert all(w['depth_m'] is None for w in waves)
    assert all(w['height_sigma_m'] is None for w in waves)
    assert any(abs(w['smoothing_height_change_m'])>0 for w in waves)
    assert all(np.isfinite(w['height_m']) for w in waves)


def synthetic_document(ys):
    points=[dict(id=str(i),status='verified',mask_ids=[],geometry=dict(valid=True,X_board_mm=[0,y,10]),cam0=[0,0],cam1=[0,0]) for i,y in enumerate(ys)]
    pair=dict(pair_index=0,representative_elapsed_seconds=.1,host_gap_ms=2,frame_indices=[0,0],
              cameras={'cam0':{'elapsed_ns':100000000},'cam1':{'elapsed_ns':102000000}})
    return dict(annotations=dict(pair=pair,revision=1),reconstructed=points)


def test_profiles_keep_gaps_masks_x_slice_and_negative_elevation():
    c=configuration()
    c.update(still_water_z_mm=20)
    record=synthetic_document([0,10,20,100,110,120])
    record['reconstructed'][1]['mask_ids']=['reflection']
    record['reconstructed'][4]['geometry']['X_board_mm'][0]=50
    profiles,excluded=build_profiles([record],c)
    points=profiles[0]['points']
    assert len(points)==4
    assert [p['segment'] for p in points]==[0,0,1,1]
    assert points[0]['eta_m']==-.01
    assert profiles[0]['covered_length_m']==pytest.approx(.04)
    assert {e['reason'] for e in excluded}=={'masked','outside_x_slice'}
    assert all(p['sigma_eta_m'] is None for p in points)
    duplicate=synthetic_document([0,10,10,20])
    assert len(build_profiles([duplicate],c)[1])==2
    record['annotations']['pair']['host_gap_ms']=99
    assert build_profiles([record],c)[0][0]['points']==[]


def test_irregular_time_fit_uncertainty_and_linear_theory_limits():
    times=[0,.13,.28,.41,.62]
    obs=[dict(time_s=t,y_m=.3+1.2*t,sigma_y_m=.001) for t in times]
    fit=fit_track(obs,.002)
    assert fit['speed_m_s']==pytest.approx(1.2)
    assert fit['conditional_speed_sigma_m_s']>0
    assert fit_track(obs)['conditional_speed_sigma_m_s'] is None
    assert fit_track(obs[:2])['speed_m_s'] is None
    shallow=linear_reference(100,.1)
    assert shallow['cp_linear_m_s']==pytest.approx(np.sqrt(.981),rel=1e-4)
    deep=linear_reference(.1,100)
    assert deep['cg_linear_m_s']==pytest.approx(deep['cp_linear_m_s']/2)


def test_tracks_require_review_reject_reuse_and_do_not_infer_packets():
    c=configuration()
    profiles=[dict(pair_index=i,time_s=i*.1,pair=dict(frame_indices=frames,host_gap_ms=2))
              for i,frames in enumerate(([0,0],[0,1],[1,2],[2,3]))]
    features=[dict(pair_index=i,feature_id='c',crest_y_m=i*.1,crest_sigma_y_m=.001) for i in range(4)]
    link=dict(track_id='w1',kind='crest',reviewed=True,evidence='synthetic known identity',
              observations=[dict(pair_index=i,feature_id='c') for i in range(4)])
    track=reviewed_tracks(profiles,features,[link],c)[0]
    assert len(track['observations'])==3
    assert track['rejected'][0]['reason']=='shared_camera_frame'
    assert track['speed_m_s']==pytest.approx(1)
    with pytest.raises(ValueError,match='review'):
        reviewed_tracks(profiles,features,[dict(link,reviewed=False)],c)
    with pytest.raises(ValueError,match='envelope'):
        reviewed_tracks(profiles,features,[dict(link,kind='packet')],c)
    with pytest.raises(ValueError,match='timing gap'):
        reviewed_tracks(profiles,features,[link],dict(c,max_track_gap_s=.05))


def test_end_to_end_annotated_pixels_to_profiles_measurements_dataset(wave_session,tmp_path):
    session=WaveSession(wave_session)
    c=configuration(session.cal.hashes)
    c.update(y_range_mm=[-90,90],max_gap_mm=3,min_wave_height_mm=1)
    store=AnnotationStore(session,tmp_path/'outputs'/'annotations')
    data=empty_document()
    for i,y in enumerate(np.linspace(-90,90,121)):
        z=2*np.cos(2*np.pi*y/30)
        data['points'].append(dict(id=f'p{i}',label='surface',confidence=1,origin='human',order=i,status='verified',
            cam0=[32.,24+90*y/(500+z)],cam1=[32-900/(500+z),24+90*y/(500+z)]))
    store.save(0,data,0)
    out=tmp_path/'outputs'/'extract'
    result=extract_annotations(session,store,out,c)
    assert result['usable_point_count']==121
    profiles=json.loads((out/'profiles.json').read_text())
    assert max(p['eta_m'] for p in profiles[0]['points'])==pytest.approx(.002,abs=1e-10)
    assert profiles[0]['points'][0]['sigma_eta_m']>0
    measured=measure_analysis(out)
    assert measured['waves']>=4
    with pytest.raises(FileExistsError):
        measure_analysis(out)
    dataset=build_dataset([out/'measurements'],tmp_path/'outputs'/'dataset')
    assert not dataset['readiness']['prediction']['ready_for_study_design_review']
    assert (out/'plots/profile_00000.svg').exists()
    assert (out/'measurements/wave_features.csv').exists()
    with pytest.raises(ValueError,match='one measurement version'):
        build_dataset([out/'measurements',out/'measurements'],tmp_path/'outputs'/'duplicate')
    (out/'physical_config.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        measure_analysis(out,tmp_path/'outputs'/'changed')


def test_conflicting_or_unobserved_breaking_remains_unknown():
    base=dict(wave_id='w',observed='no',observation_complete='False',elapsed_ns='')
    assert reviewed_outcome([base],'w')['breaking_observed']=='unknown'
    yes=dict(base,observed='yes',elapsed_ns='123000000')
    assert reviewed_outcome([yes],'w')['onset_time_s']==.123
    assert reviewed_outcome([base,yes],'w')['outcome_reason']=='conflicting_labels'
    assert reviewed_outcome([yes],'other')['breaking_observed']=='unknown'


def test_physical_up_requires_proper_rotation_and_preserves_xyz():
    c=configuration()
    c.update(board_to_tank_mm=np.diag([-1,1,-1,1]).tolist(),still_water_z_mm=-20)
    c=validate_config(c)
    profiles,_=build_profiles([synthetic_document([0,10,20])],c)
    assert profiles[0]['points'][0]['z_m']==-.01
    assert profiles[0]['points'][0]['eta_m']==.01


def test_one_supported_crest_has_no_invented_wavelength():
    y=np.linspace(0,1,201)
    waves,_=measure_segment(sampled(y,.01*np.cos(2*np.pi*y)),configuration())
    assert waves==[]  # Boundary crest lacks two interior troughs.
    y=np.linspace(0,1,201)
    waves,_=measure_segment(sampled(y,-.01*np.cos(4*np.pi*y)),configuration())
    assert waves==[]  # Two boundary troughs cannot establish a complete interior wave.
    y=np.linspace(0,1,201)
    waves,_=measure_segment(sampled(y,.01*np.cos(4*np.pi*y)),configuration())
    assert len(waves)==1
    assert waves[0]['height_m']==pytest.approx(.02)
    assert waves[0]['steepness'] is None


def test_reviewed_packet_speed_and_unknown_reference_rejection():
    c=configuration()
    profiles=[dict(pair_index=i,time_s=i*.1,pair=dict(frame_indices=[i,i],host_gap_ms=0)) for i in range(3)]
    link=dict(track_id='packet1',kind='packet',distinct_envelope=True,reviewed=True,evidence='Synthetic packet',
              observations=[dict(pair_index=i,y_m=.2+.05*i,sigma_y_m=.001,evidence='known packet center') for i in range(3)])
    result=reviewed_tracks(profiles,[],[link],c)[0]
    assert result['speed_m_s']==pytest.approx(.5)
    assert result['conditional_speed_sigma_m_s']>0
    link['observations'][0]['pair_index']=999
    with pytest.raises(ValueError,match='unknown profile'):
        reviewed_tracks(profiles,[],[link],c)


def test_cli_config_template_is_incomplete_and_non_destructive(wave_session,tmp_path):
    from analyze_wave_trial import main
    destination=tmp_path/'outputs'/'config.json'
    assert main(['init-config','--session',str(wave_session),'--output',str(destination)])==0
    config=json.loads(destination.read_text())
    assert config['calibration_hashes']==WaveSession(wave_session).cal.hashes
    assert config['board_to_tank_mm'] is None
    with pytest.raises(ValueError,match='physical-up'):
        validate_config(config)
    before=destination.read_bytes()
    assert main(['init-config','--session',str(wave_session),'--output',str(destination)])==1
    assert destination.read_bytes()==before


def test_dataset_whole_trial_splits_and_post_onset_exclusion(tmp_path):
    from wave_analysis.pipeline import digest
    directories=[]
    assignments={}
    for split in ('train','validation','test'):
        for outcome in ('yes','no'):
            name=f'{split}_{outcome}'
            folder=tmp_path/'outputs'/name
            folder.mkdir(parents=True)
            directories.append(folder)
            assignments[name]=split
            config=configuration()
            config.update(trial_id=name,experimental_validation=dict(verified=True,evidence='Synthetic test declaration'))
            profile=dict(pair_index=0,pair=dict(cameras={c:dict(elapsed_ns=100) for c in ('cam0','cam1')}))
            feature=dict(pair_index=0,feature_id='crest',height_m=.01)
            track=dict(kind='crest',track_id='w',speed_m_s=.1,observations=[dict(pair_index=0,feature_id='crest')])
            event=dict(wave_id='w',observed=outcome,observation_complete=True,elapsed_ns=1000)
            data=dict(config=config,profiles=[profile],features=[feature],tracks=[track],breaking_events=[event])
            (folder/'measurements.json').write_text(json.dumps(data))
            (folder/'analysis_manifest.json').write_text(json.dumps(dict(source_session=str(tmp_path/'recordings'/name),
                measurements_sha256=digest(folder/'measurements.json'))))
    policy=dict(splits=assignments,min_trials_per_split=2,rationale='Synthetic structural test, not sample adequacy')
    result=build_dataset(directories,tmp_path/'outputs'/'joined',policy)
    assert result['readiness']['prediction']['ready_for_study_design_review']
    assert result['readiness']['prediction']['model_trained'] is False
    with pytest.raises(ValueError,match='exactly once'):
        build_dataset(directories,tmp_path/'outputs'/'bad',dict(policy,splits={'train_yes':'train'}))
    # A late measurement of a breaking wave must not become a prebreaking predictor input.
    folder=directories[0]
    data=json.loads((folder/'measurements.json').read_text())
    for camera in data['profiles'][0]['pair']['cameras'].values():
        camera['elapsed_ns']=2000
    (folder/'measurements.json').write_text(json.dumps(data))
    manifest=json.loads((folder/'analysis_manifest.json').read_text())
    manifest['measurements_sha256']=digest(folder/'measurements.json')
    (folder/'analysis_manifest.json').write_text(json.dumps(manifest))
    result=build_dataset(directories,tmp_path/'outputs'/'late',policy)
    assert not result['readiness']['prediction']['ready_for_study_design_review']
    # Renaming a trial must not permit the same recording on both sides of a split.
    other=json.loads((directories[-1]/'analysis_manifest.json').read_text())
    other['source_session']=manifest['source_session']
    (directories[-1]/'analysis_manifest.json').write_text(json.dumps(other))
    with pytest.raises(ValueError,match='repeated source'):
        build_dataset(directories,tmp_path/'outputs'/'leak',policy)
