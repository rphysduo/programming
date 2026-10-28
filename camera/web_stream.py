import cv2
from flask import Flask, Response

app = Flask(__name__)

# Initialize both camera captures using V4L2
def init_cam(dev_id):
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

cam0 = init_cam(0)
cam1 = init_cam(1)

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video0')
def video0():
    return Response(generate_frames(cam0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video1')
def video1():
    return Response(generate_frames(cam1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Dual Camera Stream</title></head>
        <body style="background-color: #121212; color: white; text-align: center;">
            <h1>Jetson Nano Dual Web Streams</h1>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <div>
                    <h3>Camera 0 (/dev/video0)</h3>
                    <img src="/video0" width="640" height="480">
                </div>
                <div>
                    <h3>Camera 1 (/dev/video1)</h3>
                    <img src="/video1" width="640" height="480">
                </div>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
