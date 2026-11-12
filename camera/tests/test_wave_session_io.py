"""Use actual videos and irregular nanosecond timestamps to test the audit."""
import json
from pathlib import Path

import pytest

from wave_analysis.reports import inspect_trial
from wave_analysis.session_io import WaveSession, read_timestamps, timestamp_summary
from validate_stereo_3d import sha256_file


def test_inspection_is_read_only_and_uses_real_intervals(wave_session, tmp_path):
    before = {str(p):sha256_file(p) for p in wave_session.rglob('*') if p.is_file()}
    report = inspect_trial(wave_session, tmp_path/'outputs'/'audit', samples=3)
    assert report['issues'] == []
    assert report['streams']['cam0']['observed_fps'] == pytest.approx(8)
    assert report['streams']['cam0']['intervals_ms']['median'] == pytest.approx(125)
    assert report['streams']['cam1']['decoded_frames'] == 4
    assert report['candidate_pair_count'] == 4
    assert report['candidate_reuse']['cam0']
    assert (tmp_path/'outputs/audit/review/contact_sheet.jpg').exists()
    assert before == {str(p):sha256_file(p) for p in wave_session.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        inspect_trial(wave_session, tmp_path/'outputs'/'audit')


@pytest.mark.parametrize('rows', [
    ['0,0,1000', '0,10,1010'],
    ['0,0,1000', '2,10,1010'],
    ['0,0,1000', '1,0,1000'],
    ['0,20,1020', '1,10,1010'],
    ['0,0,1000', '1,10,1050'],
    ['bad,0,1000'],
])
def test_corrupted_timestamps_are_reported(tmp_path, rows):
    file=tmp_path/'times.csv'
    file.write_text('frame_index,elapsed_ns,read_timestamp_ns\n'+'\n'.join(rows))
    _, issues=read_timestamps(file)
    assert issues


def test_single_frame_has_unknown_fps():
    summary=timestamp_summary([{'frame_index':0,'elapsed_ns':2,'read_timestamp_ns':1002}])
    assert summary['observed_fps'] is None
    assert summary['intervals_ms']['count'] == 0


def test_missing_video_and_changed_hash_rejected(wave_session, tmp_path):
    (wave_session/'video/cam0.avi').unlink()
    path=wave_session/'calibration/cam0_intrinsics.npz'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='hash'):
        WaveSession(wave_session)
    report=inspect_trial(wave_session,tmp_path/'outputs'/'bad')
    assert any('Missing file' in s for s in report['issues'])
    assert report['streams']['cam0']['decoded_frames'] == 0


def test_output_cannot_be_inside_source(wave_session):
    with pytest.raises(ValueError, match='outside'):
        inspect_trial(wave_session,wave_session/'outputs'/'bad')


def test_timestamps_keep_nanosecond_precision(wave_session):
    session=WaveSession(wave_session)
    assert session.pair_info(0)['host_gap_ms'] == 4.0
    assert session.pair_info(0)['cameras']['cam1']['read_timestamp_ns'] == 1794337340448523765


def test_metadata_count_disagreement(wave_session):
    path=wave_session/'video/recording.json'
    data=json.loads(path.read_text())
    data['streams']['cam0']['frames']=999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='frame count'):
        WaveSession(wave_session)


@pytest.mark.parametrize('key', ['streams','camera_settings','calibration_hashes'])
def test_malformed_metadata_still_produces_audit(wave_session,tmp_path,key):
    path=wave_session/'video/recording.json'
    data=json.loads(path.read_text())
    data[key]=None
    path.write_text(json.dumps(data))
    report=inspect_trial(wave_session,tmp_path/'outputs'/'corrupt',samples=1)
    assert report['issues']
