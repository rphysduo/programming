"""Exercise revision history, input validation and independent annotations."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from wave_analysis.annotations import AnnotationStore, RevisionConflict, empty_document, validate_document
from wave_analysis.annotation_web import create_app
from wave_analysis.session_io import WaveSession
from wave_analysis.stereo import reconstruct_document, extract_annotations
from validate_stereo_3d import sha256_file


def point(status='verified'):
    return dict(id='crest1',label='crest',confidence=.7,origin='human',order=0,
                status=status,cam0=[32.,24.],cam1=[30.2,24.],notes='synthetic projection')


def document():
    data=empty_document()
    data['points']=[point()]
    return data


def test_revisions_reopen_edit_delete_and_do_not_touch_source(wave_session,tmp_path):
    source_before={str(p):sha256_file(p) for p in wave_session.rglob('*') if p.is_file()}
    session=WaveSession(wave_session)
    store=AnnotationStore(session,tmp_path/'outputs'/'annotations')
    first=store.save(0,document(),0)
    first_path=store.path/'pair_00000/revision_000001.json'
    first_bytes=first_path.read_bytes()
    reloaded=AnnotationStore(session,store.path).load(0)
    assert reloaded==first
    assert reloaded['pair']['cameras']['cam0']['read_timestamp_ns']==1794337340444523765
    changed=deepcopy(reloaded)
    changed['points'][0]['cam1']=[30.2,25.]
    second=store.save(0,changed,1)
    assert second['revision']==2
    assert first_path.read_bytes()==first_bytes
    with pytest.raises(RevisionConflict):
        store.save(0,first,1)
    second['points']=[]
    assert store.save(0,second,2)['points']==[]
    assert source_before=={str(p):sha256_file(p) for p in wave_session.rglob('*') if p.is_file()}


def test_concurrent_saves_have_one_winner(wave_session,tmp_path):
    session=WaveSession(wave_session)
    path=tmp_path/'outputs'/'annotations'
    stores=[AnnotationStore(session,path),AnnotationStore(session,path)]
    def save(store):
        try:
            return store.save(0,document(),0)['revision']
        except RevisionConflict:
            return 'conflict'
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(save,stores),key=str)==[1,'conflict']
    assert stores[0].load(0)['revision']==1


def test_unmatched_and_traces_never_generate_geometry(wave_session):
    data=document()
    data['points'][0].update(status='unmatched',cam1=None)
    data['traces']=[dict(id='trace',label='',origin='human',confidence=.3,notes='',camera='cam1',
                         direction='unknown',vertices_px=[[3,3],[20,20],[30,24]])]
    result=reconstruct_document(data,WaveSession(wave_session))
    assert len(result)==1
    assert result[0]['geometry'] is None
    assert result[0]['eta_mm'] is None


@pytest.mark.parametrize('update', [dict(cam0=[-1,24]),dict(cam0=[float('nan'),24]),
    dict(status='verified',cam1=None),dict(confidence=2),dict(origin='machine'),dict(order=-1)])
def test_bad_annotations_rejected(wave_session,update):
    data=document()
    data['points'][0].update(update)
    with pytest.raises(ValueError):
        validate_document(data,WaveSession(wave_session))


def test_breaking_requires_outcome_visibility_and_stores_original_time(wave_session,tmp_path):
    session=WaveSession(wave_session)
    store=AnnotationStore(session,tmp_path/'outputs'/'annotations')
    data=document()
    data['breaking']['observed']='no'
    with pytest.raises(ValueError,match='leaves view'):
        store.save(0,data,0)
    data['breaking'].update(observed='yes',onset_camera='cam1',onset_pixel=[20,30])
    saved=store.save(0,data,0)
    assert saved['breaking']['onset_observation']['elapsed_ns']==4_000_000
    assert saved['breaking']['onset_observation']['frame_index']==0


def test_masked_points_still_keep_geometry_and_flags(wave_session):
    data=document()
    data['masks']=[dict(id='mask',origin='human',confidence=1,kind='reflection',camera='cam0',
                        vertices_px=[[20,20],[40,20],[40,30],[20,30]])]
    result=reconstruct_document(data,WaveSession(wave_session))[0]
    assert result['geometry']['valid']
    assert result['mask_ids']==['mask']
    assert not result['usable_for_profile']


def test_real_app_routes_and_reproducible_reconstruction(wave_session,tmp_path):
    folder=tmp_path/'outputs'/'annotations'
    client=create_app(wave_session,folder).test_client()
    assert client.get('/').status_code==200
    assert client.get('/static/annotate.js').status_code==200
    assert client.get('/images/cam0/0').mimetype=='image/png'
    assert client.get('/images/cam2/0').status_code==400
    assert client.get('/api/pairs/999').status_code==400
    data=client.get('/api/pairs/0').get_json()['document']
    data['points']=[point()]
    saved=client.post('/api/pairs/0',json=data)
    assert saved.status_code==200
    geometry=saved.get_json()['geometry'][0]['geometry']
    assert geometry['valid']
    assert geometry['X_board_mm']==pytest.approx([0,0,0],abs=1e-8)
    assert geometry==create_app(wave_session,folder).test_client().get('/api/pairs/0').get_json()['geometry'][0]['geometry']
    assert client.post('/api/pairs/0',json=data).status_code==409
    assert client.post('/api/guide',json={'source_camera':0,'source_px':[32,24]}).get_json()['curve_segments_px']
    assert client.post('/api/geometry',json={'points':[]}).status_code==400
    session=WaveSession(wave_session)
    result=extract_annotations(session,AnnotationStore(session,folder),tmp_path/'outputs'/'geometry')
    assert result['observations']==1
    assert (tmp_path/'outputs/geometry/surface_observations.csv').exists()
    with pytest.raises(FileExistsError):
        extract_annotations(session,AnnotationStore(session,folder),tmp_path/'outputs'/'geometry')


def test_calibration_change_during_annotation_rejected(wave_session,tmp_path):
    client=create_app(wave_session,tmp_path/'outputs'/'annotations').test_client()
    path=wave_session/'calibration/stereo_params.npz'
    path.write_bytes(path.read_bytes()+b'changed')
    response=client.post('/api/pairs/0',json=document())
    assert response.status_code==400
    assert 'changed' in response.get_json()['error']


def test_geometry_rejects_bad_match_without_relaxing_thresholds(wave_session):
    data=document()
    data['points'][0]['cam1']=[30.2,38]
    result=reconstruct_document(data,WaveSession(wave_session))[0]['geometry']
    assert not result['valid']
    assert 'high_epipolar' in result['flags']
    assert 'high_reproj_cam0' in result['flags']


def test_changed_annotation_provenance_is_rejected(wave_session,tmp_path):
    store=AnnotationStore(WaveSession(wave_session),tmp_path/'outputs'/'annotations')
    saved=store.save(0,document(),0)
    saved['pair']['frame_indices']=[99,99]
    (store.path/'pair_00000/revision_000001.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='provenance'):
        store.load(0)
