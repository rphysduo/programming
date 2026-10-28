"""Browser-based ChArUco calibration assistant (live capture + solve).

This is the "watch it while it happens" companion to ``calibrate_individual.py``
and ``calibrate_studio.py``.  It opens the camera(s), streams them into the
browser with the same tag highlighting as ``web_tags.py``, shows the live
seen/not-seen marker panel, and captures a pose whenever you press *Capture Now*
(captures are entirely manual).  When enough poses are captured (or you press
*Finish & Calibrate*) it runs the exact same solve and writes the same output
files as the CLI scripts.

Run::

    python web_calibrate.py --mode individual --board small --camera 0 --output cam0_intrinsics.npz
    python web_calibrate.py --mode stereo --board small --cam0 0 --cam1 1
    python web_calibrate.py --mode world --board long --cam0 0 --cam1 1

``individual`` recovers one camera's intrinsics, ``stereo`` the transform
between the two cameras, and ``world`` each camera's pose relative to the
tank (defined by the long floor board).  The world mode writes
``tank_alignment.npz``.

Only one process can hold a V4L2 camera at a time, so do not run this at the
same time as a CLI calibration script on the same devices.
"""

import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from charuco_common import add_board_argument, board_ids, get_board, open_camera
from calibrate_individual import MIN_CORNERS, is_usable_view
from calibrate_studio import MIN_COMMON
from calibrate_world import (MIN_WORLD_CORNERS, board_to_world,
                             combine_camera_world, pack_world_params,
                             solve_camera_world)
from web_tags import LiveView, annotate_frame, encode_jpeg


# ============================================================
# SHARED STATE
# ============================================================

class CalibState:
    """Mutable status shared between the capture thread and the Flask routes."""

    def __init__(self, mode, target, keys):
        self.lock = threading.Lock()
        self.mode = mode
        self.target = target
        self.captured = 0
        self.dropped = 0
        self.common = 0
        self.message = "starting..."
        self.done = False
        self.result = []
        self.error = None
        self.cams = {k: {"markers": [], "charuco": 0, "valid": False,
                         "image_size": None, "error": None} for k in keys}

    def note(self, text):
        with self.lock:
            self.message = text

    def set_done(self, lines):
        with self.lock:
            self.done = True
            self.result = list(lines)
            self.message = lines[-1] if lines else "done"


# ============================================================
# CALIBRATION RUNNER
# ============================================================

class CalibrationRunner(threading.Thread):
    """Owns the camera(s) and runs the capture -> solve pipeline."""

    def __init__(self, args, state, views, spec):
        super().__init__(daemon=True, name="calibration")
        self.args = args
        self.mode = args.mode
        self.state = state
        self.views = views
        self.spec = spec
        self.board = spec.board
        self.detector = spec.detector
        self.expected_ids = board_ids(spec.board)
        self.capture_event = threading.Event()
        self.finish_event = threading.Event()
        self.stop_event = threading.Event()
        self._sizes = {}
        if self.mode in ("stereo", "world"):
            self.labels = {"cam0": f"cam {args.cam0}", "cam1": f"cam {args.cam1}"}
        else:
            self.labels = {"cam0": f"cam {args.camera}"}

    # -- plumbing -----------------------------------------------------------

    def run(self):
        try:
            if self.mode == "stereo":
                self._run_stereo()
            elif self.mode == "world":
                self._run_world()
            else:
                self._run_individual()
        except Exception as exc:  # noqa: BLE001 - surface to the browser
            import traceback
            traceback.print_exc()
            with self.state.lock:
                self.state.error = str(exc)
                self.state.message = f"ERROR: {exc}"
        finally:
            for v in self.views.values():
                v.close()

    def _stop_all(self):
        return self.stop_event.is_set() or self.finish_event.is_set()

    def _open(self, key, source):
        try:
            cap, size = open_camera(source, self.args.width, self.args.height,
                                    self.args.fps, focus_val=self.args.focus,
                                    exposure_val=self.args.exposure)
        except Exception as exc:  # noqa: BLE001
            with self.state.lock:
                self.state.cams[key]["error"] = str(exc)
                self.state.message = f"ERROR opening camera {source}: {exc}"
            return None
        self._sizes[key] = size
        with self.state.lock:
            self.state.cams[key]["image_size"] = size
        return cap

    def _emit(self, key, frame, corners, ids, mcorners, mids, valid):
        disp = frame.copy()
        annotate_frame(disp, corners, ids, mcorners, mids,
                       markers_only=self.args.markers_only, label=self.labels[key])
        jpeg = encode_jpeg(disp, self.args.display_width)
        if jpeg is not None:
            self.views[key].publish(jpeg)
        with self.state.lock:
            c = self.state.cams[key]
            c["markers"] = ([] if mids is None
                            else sorted({int(i) for i in mids.flatten()}))
            c["charuco"] = 0 if ids is None else int(len(ids))
            c["valid"] = bool(valid)

    def _update_progress(self, captured, dropped, common=None):
        with self.state.lock:
            prev = self.state.captured
            self.state.captured = captured
            self.state.dropped = dropped
            if common is not None:
                self.state.common = common
            if captured > prev:
                noun = "picture" if self.mode == "world" else "pose"
                self.state.message = (f"Captured {noun} {captured}/"
                                      f"{self.state.target}.")

    # -- individual ---------------------------------------------------------

    def _run_individual(self):
        a = self.args
        key = "cam0"
        self.state.note("Opening camera...")
        cap = self._open(key, a.camera)
        if cap is None:
            return
        self.state.note("Move the board to varied poses, then press Capture Now.")

        object_points, image_points = [], []
        dropped = 0
        try:
            while not self._stop_all():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, mcorners, mids = self.detector.detectBoard(gray)
                valid = ids is not None and len(ids) >= MIN_CORNERS

                if self.capture_event.is_set():
                    self.capture_event.clear()
                    if valid and not self._store_individual_direct(
                            corners, ids, object_points, image_points):
                        dropped += 1

                self._emit(key, frame, corners, ids, mcorners, mids, valid)
                self._update_progress(len(object_points), dropped)

                if len(object_points) >= a.samples:
                    break

            self._finish_individual(object_points, image_points, dropped)
        finally:
            cap.release()

    def _store_individual_direct(self, corners, ids, object_points, image_points):
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids, None, None)
        if obj_pts is not None and len(obj_pts) >= MIN_CORNERS and is_usable_view(obj_pts):
            object_points.append(obj_pts)
            image_points.append(img_pts)
            return True
        return False

    def _finish_individual(self, object_points, image_points, dropped):
        a = self.args
        lines = [f"Frames kept: {len(object_points)}  dropped: {dropped}"]
        if len(object_points) < 10:
            lines.append("Too few valid captures (minimum 10); nothing saved.")
            self.state.set_done(lines)
            return

        ret, K, D, _, _ = cv2.calibrateCamera(
            objectPoints=object_points,
            imagePoints=image_points,
            imageSize=self._sizes["cam0"],
            cameraMatrix=None,
            distCoeffs=None)
        lines += [
            f"Reprojection RMS Error: {ret:.4f} px (ideal < 0.5)",
            f"Focal Length (fx, fy): ({K[0, 0]:.2f}, {K[1, 1]:.2f})",
            f"Principal Point (cx, cy): ({K[0, 2]:.2f}, {K[1, 2]:.2f})",
        ]
        np.savez(a.output, camera_matrix=K, dist_coeffs=D, image_size=self._sizes["cam0"])
        lines.append(f"Saved to {a.output}")
        self.state.set_done(lines)

    # -- stereo -------------------------------------------------------------

    def _run_stereo(self):
        a = self.args
        self.state.note("Loading intrinsics...")
        intr0 = np.load(a.param0)
        intr1 = np.load(a.param1)
        K0, D0 = intr0["camera_matrix"], intr0["dist_coeffs"]
        K1, D1 = intr1["camera_matrix"], intr1["dist_coeffs"]
        size0 = tuple(int(x) for x in intr0["image_size"])

        cap0 = self._open("cam0", a.cam0)
        cap1 = self._open("cam1", a.cam1)
        if cap0 is None or cap1 is None:
            for c in (cap0, cap1):
                if c is not None:
                    c.release()
            return
        self.state.note("Position the board where BOTH cameras see it; then press Capture Now.")

        board_3d = self.board.getChessboardCorners()
        obj_points, img_points_0, img_points_1 = [], [], []
        dropped = 0
        try:
            while not self._stop_all():
                ok0, frame0 = cap0.read()
                ok1, frame1 = cap1.read()
                if not (ok0 and ok1):
                    time.sleep(0.02)
                    continue

                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                corners0, ids0, mc0, mi0 = self.detector.detectBoard(gray0)
                corners1, ids1, mc1, mi1 = self.detector.detectBoard(gray1)

                common_ids = np.array([], dtype=np.int32)
                if ids0 is not None and ids1 is not None:
                    common_ids = np.intersect1d(ids0.flatten(), ids1.flatten())
                common_count = int(len(common_ids))
                valid = common_count >= MIN_COMMON

                if self.capture_event.is_set():
                    self.capture_event.clear()
                    if valid and not self._store_stereo_manual(
                            common_ids, corners0, ids0, corners1, ids1,
                            board_3d, obj_points, img_points_0, img_points_1):
                        dropped += 1

                self._emit("cam0", frame0, corners0, ids0, mc0, mi0, valid)
                self._emit("cam1", frame1, corners1, ids1, mc1, mi1, valid)
                self._update_progress(len(obj_points), dropped, common=common_count)

                if len(obj_points) >= a.samples:
                    break

            self._finish_stereo(obj_points, img_points_0, img_points_1,
                                K0, D0, K1, D1, size0, dropped)
        finally:
            cap0.release()
            cap1.release()

    def _store_stereo_manual(self, common_ids, corners0, ids0, corners1, ids1,
                             board_3d, obj_points, img0, img1):
        pts0, pts1, obj = [], [], []
        for cid in common_ids:
            i0 = int(np.where(ids0.flatten() == cid)[0][0])
            i1 = int(np.where(ids1.flatten() == cid)[0][0])
            pts0.append(corners0[i0])
            pts1.append(corners1[i1])
            obj.append(board_3d[int(cid)])
        img0.append(np.array(pts0, dtype=np.float32))
        img1.append(np.array(pts1, dtype=np.float32))
        obj_points.append(np.array(obj, dtype=np.float32))
        return True

    def _finish_stereo(self, obj_points, img0, img1, K0, D0, K1, D1, size0, dropped):
        a = self.args
        lines = [f"Stereo pairs kept: {len(obj_points)}  dropped: {dropped}"]
        if len(obj_points) < 10:
            lines.append("Insufficient stereo pairs (minimum 10); nothing saved.")
            self.state.set_done(lines)
            return

        flags = cv2.CALIB_FIX_INTRINSIC
        ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=obj_points,
            imagePoints1=img0,
            imagePoints2=img1,
            cameraMatrix1=K0, distCoeffs1=D0,
            cameraMatrix2=K1, distCoeffs2=D1,
            imageSize=size0,
            flags=flags)
        lines += [
            f"Stereo Reprojection RMS Error: {ret:.4f} px (ideal < 0.8)",
            f"Baseline: {np.linalg.norm(T):.2f} mm",
        ]
        np.savez(a.output, K0=K0, D0=D0, K1=K1, D1=D1, R=R, T=T, E=E, F=F)
        lines.append(f"Saved to {a.output}")
        self.state.set_done(lines)

    # -- world (camera -> tank) ---------------------------------------------

    def _run_world(self):
        a = self.args
        self.state.note("Loading intrinsics...")
        K, D = {}, {}
        for key, path in (("cam0", a.param0), ("cam1", a.param1)):
            intr = np.load(path)
            K[key] = intr["camera_matrix"].astype(np.float64)
            D[key] = intr["dist_coeffs"].astype(np.float64)

        T_world_board = board_to_world(a.world_t, a.world_rpy)
        board_3d = np.asarray(self.board.getChessboardCorners(), np.float64)

        cap0 = self._open("cam0", a.cam0)
        cap1 = self._open("cam1", a.cam1)
        if cap0 is None or cap1 is None:
            for c in (cap0, cap1):
                if c is not None:
                    c.release()
            return
        self.state.note("Lay the long floor board in the tank; capture when BOTH "
                        "cameras see it.")

        captures = {"cam0": [], "cam1": []}
        captured = 0
        dropped = 0
        try:
            while not self._stop_all():
                ok0, frame0 = cap0.read()
                ok1, frame1 = cap1.read()
                if not (ok0 and ok1):
                    time.sleep(0.02)
                    continue

                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                corners0, ids0, mc0, mi0 = self.detector.detectBoard(gray0)
                corners1, ids1, mc1, mi1 = self.detector.detectBoard(gray1)

                if self.capture_event.is_set():
                    self.capture_event.clear()
                    got = False
                    for key, corners, ids in (("cam0", corners0, ids0),
                                              ("cam1", corners1, ids1)):
                        if len(captures[key]) >= a.samples:
                            continue
                        res = solve_camera_world(corners, ids, K[key], D[key],
                                                 self.board, board_3d, T_world_board)
                        if not res["ok"]:
                            continue
                        # Keep every valid picture -- no "is this a new pose?"
                        # check and no best-RMS replacement.
                        captures[key].append(res)
                        got = True
                    if got:
                        captured += 1
                    else:
                        dropped += 1

                self._emit("cam0", frame0, corners0, ids0, mc0, mi0,
                           ids0 is not None and len(ids0) >= MIN_WORLD_CORNERS)
                self._emit("cam1", frame1, corners1, ids1, mc1, mi1,
                           ids1 is not None and len(ids1) >= MIN_WORLD_CORNERS)
                self._update_progress(captured, dropped,
                                      common=sum(1 for v in captures.values() if v))

                if captured >= a.samples and all(captures[k] for k in captures):
                    break

            self._finish_world(captures, T_world_board, dropped, K, D)
        finally:
            cap0.release()
            cap1.release()

    def _finish_world(self, captures, T_world_board, dropped, K, D):
        a = self.args
        best = {}
        for key in ("cam0", "cam1"):
            combined = combine_camera_world(captures[key], K[key], D[key],
                                            T_world_board)
            best[key] = combined if combined.get("ok") else None

        total = sum(len(v) for v in captures.values())
        lines = [f"World pictures kept: {total} "
                 f"(cam0 {len(captures['cam0'])}, cam1 {len(captures['cam1'])})  "
                 f"dropped: {dropped}"]
        for key in ("cam0", "cam1"):
            res = best[key]
            if res is None:
                lines.append(f"{key}: NO valid board view (need >= "
                             f"{MIN_WORLD_CORNERS} spread corners)")
                continue
            c = res["center_world"]
            lines.append(f"{key}: combined {res.get('frames', 1)} picture(s), "
                         f"rms {res['rms']:.2f}px  centre ("
                         f"{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm  "
                         f"dist {np.linalg.norm(c):.1f} mm")
        if any(best[k] is None for k in best):
            lines.append("Need both cameras; nothing saved.")
            self.state.set_done(lines)
            return

        c0, c1 = best["cam0"]["center_world"], best["cam1"]["center_world"]
        lines.append(f"camera-to-camera baseline: {np.linalg.norm(c0 - c1):.1f} mm")
        out = pack_world_params(best, T_world_board, self.spec,
                                self._sizes["cam0"],
                                stereo_scale=a.stereo_scale)
        np.savez(a.world_output, **out)
        lines.append(f"Saved to {a.world_output}")
        self.state.set_done(lines)


# ============================================================
# FLASK APP
# ============================================================

CALIB_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Calibration View</title>
<style>
  :root { color-scheme: dark; }
  body { background:#121212; color:#eee; font-family: system-ui, sans-serif;
         margin:0; padding:16px; }
  h1 { font-size:1.15rem; margin:0 0 12px; font-weight:600; }
  .hud { position:sticky; top:0; z-index:5; background:#181818; border:1px solid #2c2c2c;
         border-radius:10px; padding:12px 14px; margin-bottom:16px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .bar { flex:1 1 260px; height:16px; background:#2a2a2a; border-radius:8px; overflow:hidden; }
  .fill { height:100%; width:0; background:#2e8b41; transition:width .2s; }
  .kv { font-size:0.85rem; color:#bbb; }
  .kv b { color:#fff; }
  .good { color:#7CFC7C; } .bad { color:#ff8080; }
  button { background:#2b5fa8; color:#fff; border:0; border-radius:6px; padding:9px 14px;
           font-size:0.9rem; cursor:pointer; }
  button.alt { background:#7a4b12; }
  button:disabled { opacity:.5; cursor:default; }
  #msg { font-size:0.9rem; margin-top:8px; color:#ffd479; }
  #result { font-size:0.85rem; margin-top:8px; white-space:pre-wrap; color:#cfe; }
  .streams { display:flex; gap:20px; flex-wrap:wrap; }
  .cam { flex:1 1 480px; background:#1b1b1b; border:1px solid #2c2c2c; border-radius:10px; padding:12px; }
  .cam h2 { font-size:0.95rem; margin:0 0 8px; font-weight:600; color:#9ad; }
  .cam img { width:100%; height:auto; display:block; border-radius:6px; background:#000; }
  .meta { color:#aaa; font-size:0.8rem; margin:4px 0; }
  .err { color:#ff6b6b; font-size:0.8rem; margin:4px 0; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(34px,1fr)); gap:4px; margin-top:6px; }
  .chip { text-align:center; font-size:0.72rem; padding:3px 0; border-radius:4px;
          font-variant-numeric:tabular-nums; }
  .chip.on { background:#1e5b2a; color:#b7ffbf; border:1px solid #2e8b41; }
  .chip.off { background:#2a2a2a; color:#666; border:1px solid #333; }
</style>
</head>
<body>
  <h1>Calibration View &mdash; mode: __MODE__ &mdash; board: __BOARD__</h1>
  <div class="hud">
    <div class="row">
      <div class="kv">captured <b id="captured">0</b> / <b id="target">0</b></div>
      <div class="bar"><div class="fill" id="fill"></div></div>
      <div class="kv">dropped <b id="dropped">0</b></div>
      <button id="btn-capture" onclick="doCapture()">Capture Now</button>
      <button id="btn-finish" class="alt" onclick="doFinish()">Finish &amp; Calibrate</button>
    </div>
    <div class="row" style="margin-top:8px"><div class="kv" id="state"></div></div>
    <div id="msg"></div>
    <div id="result"></div>
  </div>
  <div class="streams">
__PANELS__
  </div>
<script>
async function doCapture() { await fetch('/capture', {method:'POST'}); }
async function doFinish() { await fetch('/finish', {method:'POST'}); }
function chipGrid(exp, seen) {
  let h = '<div class="grid">';
  for (const id of exp) h += '<span class="chip ' + (seen.has(id) ? 'on' : 'off') + '">' + id + '</span>';
  return h + '</div>';
}
async function tick() {
  try {
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();
    document.getElementById('captured').textContent = s.captured;
    document.getElementById('target').textContent = s.target;
    document.getElementById('dropped').textContent = s.dropped;
    document.getElementById('fill').style.width =
      (s.target ? Math.min(100, 100 * s.captured / s.target) : 0) + '%';
    let st = 'manual capture &mdash; click Capture Now to store a pose';
    if (s.mode === 'world') st = 'manual capture &mdash; click Capture Now to store a picture (repeats are kept)';
    if (s.mode === 'stereo') st += ' &middot; common corners: ' + s.common;
    if (s.mode === 'world') st += ' &middot; cameras solved: ' + s.common + '/2';
    document.getElementById('state').innerHTML = st;
    document.getElementById('msg').textContent = s.message || '';
    document.getElementById('result').textContent =
      (s.result && s.result.length) ? s.result.join('\\n') : '';
    if (s.done) {
      document.getElementById('btn-finish').disabled = true;
      document.getElementById('btn-capture').disabled = true;
    }
    for (const key in s.cams) {
      const c = s.cams[key];
      const el = document.getElementById('panel-' + key);
      if (!el) continue;
      const size = c.image_size ? (c.image_size[0] + 'x' + c.image_size[1]) : '?';
      let html = '<div class="meta">' + size + ' &middot; markers ' + c.marker_count +
                 ' &middot; charuco ' + c.charuco + ' &middot; <span class="' +
                 (c.valid ? 'good' : 'bad') + '">' +
                 (c.valid ? 'VALID' : 'not enough') + '</span></div>';
      if (c.error) html += '<div class="err">ERROR: ' + c.error + '</div>';
      html += chipGrid(s.expected, new Set(c.markers));
      el.innerHTML = html;
    }
  } catch (e) { /* keep polling */ }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


def build_calib_html(mode, keys, board_label=""):
    panels = []
    for key in keys:
        panels.append(
            f'<div class="cam">'
            f'<h2 id="title-{key}">{key}</h2>'
            f'<img src="/video/{key}">'
            f'<div class="stat" id="panel-{key}">waiting for frames...</div>'
            f'</div>')
    return (CALIB_TEMPLATE.replace("__PANELS__", "\n".join(panels))
            .replace("__MODE__", mode)
            .replace("__BOARD__", board_label))


def create_app(state, runner, views):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return build_calib_html(state.mode, list(views.keys()),
                                getattr(runner.spec, "label", ""))

    @app.route("/video/<key>")
    def video(key):
        v = views.get(key)
        if v is None:
            return ("unknown camera", 404)
        return Response(v.frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        with state.lock:
            cams = {}
            for key, c in state.cams.items():
                seen = list(c["markers"])
                cams[key] = {
                    "markers": seen,
                    "marker_count": len(seen),
                    "charuco": c["charuco"],
                    "valid": c["valid"],
                    "image_size": c["image_size"],
                    "error": c["error"],
                    "missing": [i for i in runner.expected_ids if i not in set(seen)],
                }
            return jsonify({
                "mode": state.mode,
                "captured": state.captured,
                "target": state.target,
                "dropped": state.dropped,
                "common": state.common,
                "message": state.message,
                "done": state.done,
                "result": state.result,
                "error": state.error,
                "expected": runner.expected_ids,
                "cams": cams,
            })

    @app.route("/capture", methods=["POST"])
    def capture():
        runner.capture_event.set()
        return jsonify({"ok": True})

    @app.route("/finish", methods=["POST"])
    def finish():
        runner.finish_event.set()
        return jsonify({"ok": True})

    return app


# ============================================================
# ENTRY POINT
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Browser ChArUco calibration assistant (live capture + solve).")
    p.add_argument("--mode", choices=["individual", "stereo", "world"],
                   default="individual",
                   help="Calibrate a single camera's intrinsics ('individual'), "
                        "the transform between the two cameras ('stereo'), or each "
                        "camera's pose in the tank frame ('world')")
    p.add_argument("--camera", default="0", help="Camera index for individual mode")
    p.add_argument("--cam0", default="0", help="First camera index for stereo/world mode")
    p.add_argument("--cam1", default="1", help="Second camera index for stereo/world mode")
    p.add_argument("--output", default=None,
                   help="Output .npz path (default: cam_intrinsics.npz / stereo_params.npz)")
    p.add_argument("--param0", default="cam0_intrinsics.npz",
                   help="Intrinsics file for cam0 (stereo mode)")
    p.add_argument("--param1", default="cam1_intrinsics.npz",
                   help="Intrinsics file for cam1 (stereo/world mode)")
    p.add_argument("--world-output", "--tank-output", dest="world_output",
                   default="tank_alignment.npz",
                   help="Output .npz path for world mode (camera -> tank)")
    p.add_argument("--world-t", "--tank-t", dest="world_t", default="0,0,0",
                   help="World mode: board origin in the tank frame (mm), 'x,y,z'")
    p.add_argument("--world-rpy", "--tank-rpy", dest="world_rpy", default="0,0,0",
                   help="World mode: board orientation in the tank frame "
                        "(deg, fixed-axis XYZ roll,pitch,yaw), 'r,p,y'")
    p.add_argument("--stereo-scale", choices=["provisional", "validated"],
                   default="provisional",
                   help="World mode: record whether stereo_params.npz metric "
                        "scale is provisional (default) or validated")
    p.add_argument("--samples", type=int, default=None,
                   help="Target number of captured poses (default 20 individual / "
                        "15 stereo / 10 world pictures; world keeps every valid "
                        "picture, even repeated poses)")
    p.add_argument("--focus", type=int, default=30, help="Manual focus value (0-255)")
    p.add_argument("--exposure", type=int, default=None,
                   help="Freeze manual exposure (disables auto-exposure/WB)")
    p.add_argument("--width", type=int, default=1920, help="Capture width")
    p.add_argument("--height", type=int, default=1080, help="Capture height")
    p.add_argument("--fps", type=int, default=30, help="Capture frame rate")
    p.add_argument("--display-width", type=int, default=960,
                   help="Downscale the streamed frames to this width (0 = full size)")
    p.add_argument("--markers-only", action="store_true",
                   help="Draw only ArUco marker outlines (skip ChArUco corner overlay)")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5001, help="HTTP port")
    add_board_argument(p)
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "world" and args.board != "long":
        raise SystemExit(
            "World/tank mode requires --board long (4x35, 44/33 mm); "
            f"got {args.board!r}. The small board is only for intrinsics.")

    if args.output is None:
        args.output = "cam_intrinsics.npz" if args.mode == "individual" else "stereo_params.npz"
    if args.samples is None:
        args.samples = {"individual": 20, "stereo": 15, "world": 10}[args.mode]

    spec = get_board(args.board)

    keys = ["cam0"] if args.mode == "individual" else ["cam0", "cam1"]
    views = {k: LiveView() for k in keys}
    state = CalibState(args.mode, args.samples, keys)
    runner = CalibrationRunner(args, state, views, spec)
    runner.start()

    app = create_app(state, runner, views)
    print(f"\nCalibration View running ({args.mode}, board: {spec.label}). Open "
          f"http://{args.host}:{args.port}/ in a browser. Captures are manual: "
          f"click 'Capture Now' to store a pose; press 'Finish & Calibrate' to solve "
          f"(Ctrl+C to stop).")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        runner.stop_event.set()


if __name__ == "__main__":
    main()
