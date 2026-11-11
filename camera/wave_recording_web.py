"""Browser controls and live previews for the stereo wave recorder."""
from contextlib import ExitStack
from threading import Event, Lock, Thread
import time

import cv2
from flask import Flask, Response, jsonify, request

from record_stereo_waves import (
    CALIB_NAMES, CAMERAS, NS_PER_SECOND, TIMING_WARNING, _RecordingStream,
    _check_frame, _new_session, _open_recording_cameras, _read_calibration, _write_recording_info,
    ensure_session,
)
from web_tags import LiveView, encode_jpeg


class _Trial:
    """Own both video writers for one session, with an independent lock per writer."""

    def __init__(self, session, cal, settings, sizes, rates):
        self.session = session
        self.cal = cal
        self.settings = settings
        self.video_dir = session / "video"
        self.video_dir.mkdir()
        self.stopped = Event()
        self.locks = {camera: Lock() for camera in CAMERAS}
        with ExitStack() as resources:
            self.streams = {
                camera: _RecordingStream(resources, self.video_dir, camera, size, rate)
                for camera, size, rate in zip(CAMERAS, sizes, rates)
            }
            self.resources = resources.pop_all()
        self.start = time.monotonic_ns()
        self.start_unix = time.time_ns()

    def write(self, camera, frame, now):
        with self.locks[camera]:
            if not self.stopped.is_set() and now >= self.start:
                self.streams[camera].write(frame, now, self.start, self.start_unix)

    def finish(self, *, save, interrupted=False):
        self.stopped.set()
        with self.locks["cam0"], self.locks["cam1"]:
            self.resources.close()
            if not save:
                return None
            streams = {camera: stream.summary() for camera, stream in self.streams.items()}
            _write_recording_info(self.video_dir, self.cal, self.settings, streams,
                                  self.start_unix, interrupted=interrupted)
        print(f"Saved wave recording: {self.session}", flush=True)
        return {"session": str(self.session), "streams": streams}


class WaveRecorder:
    """Keep both cameras running while the browser starts and stops trials."""

    def __init__(self, output_dir, sources, cal, caps, sizes, rates, settings):
        self.output_dir = output_dir
        self.sources = sources
        self.cal = cal
        self.sizes = sizes
        self.rates = rates
        self.settings = settings
        self.views = {camera: LiveView() for camera in CAMERAS}
        self.lock = Lock()
        self.closed = Event()
        self.ready = set()
        self.error = None
        self.active = None
        self.last = None
        self.threads = [Thread(target=self._capture, args=(camera, cap, size),
                               name=f"wave-{camera}", daemon=True)
                        for camera, cap, size in zip(CAMERAS, caps, sizes)]

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc):
        self.closed.set()
        try:
            with self.lock:
                if self.active is not None:
                    self._finish(save=True, interrupted=True)
        finally:
            for thread in self.threads:
                thread.join()

    def _capture(self, camera, cap, size):
        next_preview = 0
        try:
            while not self.closed.is_set():
                ok, frame = cap.read()
                now = time.monotonic_ns()
                if self.closed.is_set():
                    break
                if not ok:
                    raise RuntimeError(f"{camera} stopped delivering frames")
                _check_frame(frame, size, camera)
                with self.lock:
                    trial = self.active
                if trial is not None:
                    trial.write(camera, frame, now)
                if now >= next_preview:
                    jpeg = encode_jpeg(frame, display_width=640, quality=75)
                    if jpeg is not None:
                        self.views[camera].publish(jpeg)
                        with self.lock:
                            self.ready.add(camera)
                    next_preview = now + NS_PER_SECOND // 8
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            self.closed.set()
            with self.lock:
                self.error = str(exc)
                if self.active is not None:
                    self._finish(save=False)
        finally:
            self.views[camera].close()

    def _finish(self, *, save, interrupted=False):
        trial, self.active = self.active, None
        result = trial.finish(save=save, interrupted=interrupted)
        if result is not None:
            self.last = result
        return result

    def start(self):
        """Start a new session using the calibration that initialized the live views."""
        with self.lock:
            if self.closed.is_set() or len(self.ready) != len(CAMERAS):
                raise RuntimeError(self.error or "Wait for both camera previews.")
            if self.active is not None:
                raise RuntimeError("A recording is already running.")
            if _read_calibration(self.sources).hashes != self.cal.hashes:
                raise RuntimeError("Calibration changed on disk. Restart the recorder.")
            session = _new_session(self.output_dir)
            cal = _read_calibration(ensure_session(session, self.sources, quiet=True))
            if cal.hashes != self.cal.hashes:
                raise RuntimeError("Calibration changed while starting. Restart the recorder.")
            self.active = _Trial(session, cal, self.settings, self.sizes, self.rates)
            return {"session": str(session)}

    def stop(self):
        """Finalize the current videos so they can be exported immediately."""
        with self.lock:
            if self.active is None:
                raise RuntimeError("No recording is running.")
            return self._finish(save=True)

    def status(self):
        """Return a consistent view of preview readiness and recording state."""
        with self.lock:
            return {
                "ready": len(self.ready) == len(CAMERAS) and not self.closed.is_set(),
                "error": self.error,
                "active": str(self.active.session) if self.active is not None else None,
                "elapsed_seconds": ((time.monotonic_ns() - self.active.start) / NS_PER_SECOND
                                    if self.active is not None else None),
                "last": self.last,
            }


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stereo wave recorder</title>
<style>
  body { margin: 1.25rem auto; padding: 0 1rem; max-width: 1300px;
         color: #e9eef5; background: #151b23; font: 16px system-ui, sans-serif; }
  h1 { margin-bottom: .5rem; }
  .views { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
  figure { margin: 0; } figcaption { margin: .6rem 0; }
  img { width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #090d12; }
  button { padding: .65rem 1rem; margin-right: .5rem; font: inherit; cursor: pointer; }
  button:disabled { cursor: default; } #status { min-height: 1.5em; }
  #saved, #error { overflow-wrap: anywhere; } #error { color: #ffad9f; }
  @media (max-width: 660px) { .views { grid-template-columns: 1fr; } }
</style>
<h1>Stereo wave recorder</h1>
<button id="start" disabled>Start recording</button>
<button id="stop" disabled>Stop recording</button>
<p id="status" role="status">Waiting for both cameras…</p>
<p id="error" role="alert"></p>
<div class="views">
  <figure><figcaption>cam0</figcaption><img src="/video/cam0" alt="Live cam0 preview"></figure>
  <figure><figcaption>cam1</figcaption><img src="/video/cam1" alt="Live cam1 preview"></figure>
</div>
<p id="saved"></p>
<p>Previews are reduced for the browser. Saved videos use the calibrated resolution.</p>
<script>
  const startButton = document.querySelector('#start');
  const stopButton = document.querySelector('#stop');
  const statusLine = document.querySelector('#status');
  const error = document.querySelector('#error');
  const saved = document.querySelector('#saved');
  let busy = false;
  let revision = 0;

  async function refresh() {
    const requestRevision = revision;
    if (busy) return;
    try {
      const response = await fetch('/status', {cache: 'no-store'});
      if (!response.ok) throw new Error('Could not read recorder status.');
      const state = await response.json();
      if (requestRevision !== revision) return;
      startButton.disabled = !state.ready || Boolean(state.active);
      stopButton.disabled = !state.active;
      statusLine.textContent = state.active
        ? `Recording ${state.elapsed_seconds.toFixed(1)} s · ${state.active}`
        : state.ready ? 'Ready to record' : 'Cameras unavailable or starting';
      if (state.error) error.textContent = state.error;
      if (state.last) {
        const stats = Object.entries(state.last.streams).map(([camera, stream]) =>
          `${camera} ${stream.frames} frames` +
          (stream.observed_fps == null ? '' : ` at ${stream.observed_fps.toFixed(1)} fps`));
        saved.textContent = `Last saved session: ${state.last.session} (${stats.join(', ')})`;
      }
    } catch (failure) {
      if (requestRevision !== revision) return;
      startButton.disabled = stopButton.disabled = true;
      statusLine.textContent = failure.message;
    }
  }

  async function control(action) {
    if (busy) return;
    busy = true;
    revision++;
    startButton.disabled = stopButton.disabled = true;
    error.textContent = '';
    statusLine.textContent = action === 'start' ? 'Starting recording…' : 'Saving both videos…';
    try {
      const response = await fetch(`/recording/${action}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
    } catch (failure) {
      error.textContent = failure.message;
    } finally {
      busy = false;
      await refresh();
    }
  }

  startButton.onclick = () => control('start');
  stopButton.onclick = () => control('stop');
  setInterval(refresh, 500);
  refresh();
</script>
</html>"""


def create_app(recorder):
    """Serve live streams, recording controls and the current session path."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/video/<camera>")
    def video(camera):
        if camera not in recorder.views:
            return "Unknown camera", 404
        return Response(recorder.views[camera].frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/status")
    def status():
        return jsonify(recorder.status())

    @app.post("/recording/<action>")
    def control(action):
        if action not in ("start", "stop"):
            return jsonify(error="Unknown recording action"), 404
        if not request.is_json:
            return jsonify(error="Use a JSON request"), 415
        try:
            result = recorder.start() if action == "start" else recorder.stop()
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(result)

    return app


def serve(args):
    """Run the live recorder with both cameras open until the server stops."""
    sources = {name: args.calib_dir / name for name in CALIB_NAMES}
    cal = _read_calibration(sources)
    with _open_recording_cameras(args, cal) as (caps, sizes, rates, settings):
        with WaveRecorder(args.output_dir, sources, cal, caps, sizes, rates, settings) as recorder:
            print(TIMING_WARNING, flush=True)
            print(f"Open port {args.port} in a browser to start and stop wave recordings.", flush=True)
            create_app(recorder).run(host=args.host, port=args.port, threaded=True,
                                     use_reloader=False)
    return 0
