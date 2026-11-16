#!/usr/bin/env python3
"""Demonstrate offline analysis using synthetic known XYZ, never real tank data."""
import argparse
from pathlib import Path
import numpy as np

from wave_analysis.config import template, validate_config
from wave_analysis.pipeline import export_profiles, measure_analysis, digest
from wave_analysis.reports import json_text, output_path, write_csv
from wave_analysis.wave_features import measure_profiles
from wave_analysis.dataset import build_dataset


def run(output_dir):
    """Create a new synthetic trial, profiles, reviewed truth links and dataset."""
    output=output_path('/synthetic_source_not_an_experiment',output_dir)
    output.mkdir(parents=True,exist_ok=False)
    config=template()
    config.update(trial_id='SYNTHETIC_DEMO',calibration_hashes={'synthetic':'not-a-calibration'},
        frame_verified=True,frame_evidence='Synthetic coordinate definition only',
        board_to_tank_mm=np.eye(4).tolist(),still_water_z_mm=0,still_water_sigma_mm=0,
        slice_x_mm=0,slice_halfwidth_mm=1,y_range_mm=[0,2000],max_gap_mm=10,
        max_host_gap_ms=10,propagation_direction=1,min_wave_height_mm=5,
        min_samples_per_wave=9,max_track_gap_s=.3,still_water_depth_mm=150,
        notes='Synthetic software demonstration. No experimental validation.')
    config=validate_config(config)
    documents=[]
    times=[0,.11,.23,.37,.52]
    for index,time in enumerate(times):
        y=np.linspace(0,2,1001)
        eta=.01*np.cos(2*np.pi*(y-.12*time)/.4)
        pair=dict(pair_index=index,frame_indices=[index,index],host_gap_ms=0,
            representative_elapsed_seconds=time,cameras={c:dict(elapsed_ns=round(time*1e9)) for c in ('cam0','cam1')})
        points=[dict(id=f'p{i}',status='verified',mask_ids=[],cam0=None,cam1=None,
            geometry=dict(valid=True,X_board_mm=[0,float(yy*1000),float(zz*1000)])) for i,(yy,zz) in enumerate(zip(y,eta))]
        documents.append(dict(annotations=dict(pair=pair,revision=1),reconstructed=points))
    summary=export_profiles(output,documents,config,None)
    (output/'reconstruction.json').write_text(json_text(documents))
    write_csv(output/'breaking_events.csv',[],['wave_id','observed','observation_complete','elapsed_ns'])
    manifest=dict(source_session='/synthetic_source_not_an_experiment',calibration_hashes=config['calibration_hashes'],
        status='SYNTHETIC SOFTWARE DEMO',profile_inputs_sha256={name:digest(output/name) for name in
        ('profiles.json','physical_config.json','reconstruction.json','breaking_events.csv')})
    (output/'analysis_manifest.json').write_text(json_text(manifest))
    import json
    profiles=json.loads((output/'profiles.json').read_text())
    features,_,_=measure_profiles(profiles,config)
    observations=[]
    for index,time in enumerate(times):
        # Ground truth is available in this demonstration only.
        crest=min((w for w in features if w['pair_index']==index),key=lambda w:abs(w['crest_y_m']-(.8+.12*time)))
        observations.append(dict(pair_index=index,feature_id=crest['feature_id']))
    links=output/'synthetic_truth_tracks.json'
    links.write_text(json_text([dict(track_id='synthetic_crest',kind='crest',reviewed=True,
        evidence='Known synthetic traveling-wave identity, not an image tracking algorithm',observations=observations)]))
    measurements=measure_analysis(output,tracks_path=links)
    dataset=build_dataset([output/'measurements'],output.parent/(output.name+'_dataset'))
    return dict(profiles=summary,measurements=measurements,dataset=dataset)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    print(run(parser.parse_args().output_dir))
