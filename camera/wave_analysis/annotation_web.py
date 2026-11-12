"""Browser review of offline trial frames, sharing the existing geometry routines."""
from functools import lru_cache

import cv2
from flask import Flask, Response, jsonify, render_template, request

from correspondence_guide import predict_correspondence
from .annotations import AnnotationStore, RevisionConflict
from .reports import json_text
from .session_io import WaveSession
from .stereo import reconstruct_document


def create_app(session_path, output_dir):
    """Create an offline annotation app bound to one immutable trial snapshot."""
    session = WaveSession(session_path)
    store = AnnotationStore(session, output_dir)
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

    def response(value):
        return Response(json_text(value), mimetype='application/json')

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 409 if isinstance(exc, RevisionConflict) else 400

    @app.errorhandler(OSError)
    def io_error(exc):
        return jsonify(error=str(exc)), 500

    @app.get('/')
    def index():
        return render_template('annotate.html')

    @app.get('/api/session')
    def metadata():
        return response(dict(source_session=str(session.path), output_dir=str(store.path),
            calibration=session.calibration_audit(),
            pairs=[session.pair_info(i) for i in range(len(session.pairs))]))

    @app.route('/api/pairs/<int:index>', methods=['GET', 'POST'])
    def pair(index):
        if request.method == 'GET':
            document = store.load(index)
        else:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON annotation document')
            document = store.save(index, data, data.get('revision'))
        return response(dict(document=document, geometry=reconstruct_document(document, session)))

    @lru_cache(maxsize=8)
    def encoded_frame(camera, frame_index):
        frame = session.frame(camera, frame_index)
        ok, data = cv2.imencode('.png', frame)
        if not ok:
            raise ValueError('Could not encode source image')
        return data.tobytes()

    @app.get('/images/<camera>/<int:frame_index>')
    def image(camera, frame_index):
        return Response(encoded_frame(camera, frame_index), mimetype='image/png')

    @app.post('/api/geometry')
    def geometry():
        return response(reconstruct_document(request.get_json(silent=True), session))

    @app.post('/api/guide')
    def guide():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError('Expected source_camera and source_px')
        session.assert_calibration_current()
        return response(predict_correspondence(session.cal, data.get('source_camera'), data.get('source_px')))

    return app
