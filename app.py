from flask import Flask, render_template, Response, request
import cv2
from yolo import YOLO
import os
from database import init_db, get_analytics

app = Flask(__name__)
yolo = YOLO()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()  # read the frame from the video
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video when end is reached
            continue
        else:
            frame = yolo.detect(frame)  # detect people in the frame using YOLO
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    video_path = 'uploads/video.mp4'
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'
        
        file = request.files['video']
        if file.filename == '':
            return 'No selected file'

        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
            file.save(video_path)
            return 'Video uploaded successfully!'

    uploaded_video = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    if os.path.exists(uploaded_video):
        return render_template('index.html', video_uploaded=True)
    else:
        return render_template('index.html', video_uploaded=False)

@app.route('/analytics')
def analytics():
    data = get_analytics()
    return render_template('analytics.html', data=data, data_count=len(data))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
