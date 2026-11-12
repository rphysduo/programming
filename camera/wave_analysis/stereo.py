"""Diagnostic reconstruction of human-verified observations with existing geometry."""
import numpy as np
import cv2

from validate_stereo_3d import cam0_to_board, triangulate_point
from .annotations import validate_document
from .reports import json_text, output_path, write_csv


def reconstruct_document(document, session):
    """Retain raw XYZ and all original geometry flags for verified point matches.

    Physical up, tank axes and still-water reference await measured configuration.
    Annotation confidence and timing remain independent of the geometry gates.
    """
    validate_document(document, session)
    session.assert_calibration_current()
    results = []
    for point in document['points']:
        masked = []
        for mask in document['masks']:
            observation = point.get(mask['camera'])
            if observation is not None and cv2.pointPolygonTest(
                    np.asarray(mask['vertices_px'], np.float32), tuple(observation), False) >= 0:
                masked.append(mask['id'])
        item = dict(id=point['id'], label=point.get('label', ''), order=point['order'],
                    status=point['status'], confidence=point['confidence'],
                    cam0=point.get('cam0'), cam1=point.get('cam1'), mask_ids=masked,
                    geometry=None, usable_for_profile=False, physical_height_mm=None,
                    eta_mm=None, positional_uncertainty_mm=None,
                    physical_accuracy='unvalidated', physical_frame_status='board axes only')
        if point['status'] == 'verified':
            result = triangulate_point(point['cam0'], point['cam1'], session.cal)
            result['X_board_mm'] = cam0_to_board(np.asarray(result['X_cam0_mm']), session.cal).tolist()
            item['geometry'] = result
        results.append(item)
    return results


def extract_annotations(session, store, output_dir):
    """Export each saved pair's latest reviewed geometry to a fresh directory."""
    output = output_path(session.path, output_dir)
    if output == store.path or store.path in output.parents or output in store.path.parents:
        raise ValueError('Choose an extraction directory separate from annotations')
    output.mkdir(parents=True, exist_ok=False)
    documents, rows, events = [], [], []
    for index in range(len(session.pairs)):
        document = store.load(index)
        if not document['revision']:
            continue
        observations = reconstruct_document(document, session)
        documents.append(dict(annotations=document, reconstructed=observations))
        info = document['pair']
        for point in observations:
            geometry = point['geometry'] or {}
            xyz = geometry.get('X_board_mm', [None]*3)
            rows.append(dict(pair_index=index, annotation_revision=document['revision'],
                             point_id=point['id'], label=point['label'], order=point['order'],
                             status=point['status'], confidence=point['confidence'],
                             cam0_frame=info['frame_indices'][0], cam1_frame=info['frame_indices'][1],
                             cam0_elapsed_ns=info['cameras']['cam0']['elapsed_ns'],
                             cam1_elapsed_ns=info['cameras']['cam1']['elapsed_ns'], host_gap_ms=info['host_gap_ms'],
                             cam0_u=point['cam0'][0] if point['cam0'] else None,
                             cam0_v=point['cam0'][1] if point['cam0'] else None,
                             cam1_u=point['cam1'][0] if point['cam1'] else None,
                             cam1_v=point['cam1'][1] if point['cam1'] else None,
                             board_x_mm=xyz[0], board_y_mm=xyz[1], board_z_mm=xyz[2],
                             geometry_valid=geometry.get('valid'), flags='|'.join(geometry.get('flags', [])),
                             masked_by='|'.join(point['mask_ids']),
                             reproj_cam0_px=geometry.get('reproj_error_cam0_px'),
                             reproj_cam1_px=geometry.get('reproj_error_cam1_px'),
                             epipolar_px=geometry.get('epipolar_sampson_px'),
                             physical_height_mm=None, eta_mm=None, positional_uncertainty_mm=None))
        event = document['breaking']
        onset = event.get('onset_observation') or {}
        events.append(dict(pair_index=index, annotation_revision=document['revision'],
                           wave_id=event['wave_id'], observed=event['observed'], confidence=event['confidence'],
                           onset_camera=event.get('onset_camera'), frame_index=onset.get('frame_index'),
                           elapsed_ns=onset.get('elapsed_ns'), read_timestamp_ns=onset.get('read_timestamp_ns'),
                           position_raw_px=event.get('onset_pixel'), observation_complete=event['observation_complete'],
                           criterion=event['criterion'], notes=event['notes']))
    manifest = dict(source_session=str(session.path), calibration_hashes=session.cal.hashes,
                    source_files=session.sources, annotation_directory=str(store.path),
                    coordinate_convention='Board XYZ in mm; physical up, tank axes and eta reference unverified',
                    status='Diagnostic geometry from manual annotations; physical measurements pending validation')
    (output/'analysis_manifest.json').write_text(json_text(manifest))
    (output/'reconstruction.json').write_text(json_text(documents))
    columns = ['pair_index','annotation_revision','point_id','label','order','status','confidence',
               'cam0_frame','cam1_frame','cam0_elapsed_ns','cam1_elapsed_ns','host_gap_ms',
               'cam0_u','cam0_v','cam1_u','cam1_v','board_x_mm','board_y_mm','board_z_mm',
               'geometry_valid','flags','masked_by','reproj_cam0_px','reproj_cam1_px','epipolar_px',
               'physical_height_mm','eta_mm','positional_uncertainty_mm']
    # Strict JSON normalization removes nonfinite diagnostics before CSV export.
    import json
    rows = json.loads(json_text(rows))
    write_csv(output/'surface_observations.csv', rows, columns)
    write_csv(output/'breaking_events.csv', events, ['pair_index','annotation_revision','wave_id','observed',
        'confidence','onset_camera','frame_index','elapsed_ns','read_timestamp_ns','position_raw_px',
        'observation_complete','criterion','notes'])
    return dict(annotated_pairs=len(documents), observations=len(rows), output_dir=str(output))
