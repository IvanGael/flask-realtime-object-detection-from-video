# from flask import Flask, render_template, Response, request
# import cv2
# from yolo import YOLO
# import os
# from database import init_db, get_analytics

# app = Flask(__name__)
# yolo = YOLO()

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def gen_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         success, frame = cap.read()  # read the frame from the video
#         if not success:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video when end is reached
#             continue
#         else:
#             frame = yolo.detect(frame)  # detect people in the frame using YOLO
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     video_path = 'uploads/video.mp4'
#     return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'video' not in request.files:
#             return 'No file part'
        
#         file = request.files['video']
#         if file.filename == '':
#             return 'No selected file'

#         if file:
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
#             file.save(video_path)
#             return 'Video uploaded successfully!'

#     uploaded_video = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
#     if os.path.exists(uploaded_video):
#         return render_template('index.html', video_uploaded=True)
#     else:
#         return render_template('index.html', video_uploaded=False)

# @app.route('/analytics')
# def analytics():
#     data = get_analytics()
#     return render_template('analytics.html', data=data, data_count=len(data))

# if __name__ == '__main__':
#     init_db()
#     app.run(debug=True)


# pip install xlsxwriter
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
from yolo import YOLO
import os
from database import init_db, get_analytics
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

@app.route('/analytics/data')
def analytics_data():
    data = get_analytics()
    return jsonify(data)

@app.route('/download/csv')
def download_csv():
    data = get_analytics()
    df = pd.DataFrame(data, columns=['Datetime', 'Name', 'Count'])
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=detected_objects.csv"}
    )

@app.route('/download/excel')
def download_excel():
    data = get_analytics()
    df = pd.DataFrame(data, columns=['Datetime', 'Name', 'Count'])
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='detected_objects.xlsx',
        as_attachment=True
    )

@app.route('/download/pdf')
def download_pdf():
    data = get_analytics()
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 40, "Detected Objects Report")
    y = height - 80
    for row in data:
        c.drawString(100, y, f"{row[0]}: {row[1]} - {row[2]}")
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='detected_objects.pdf'
    )

if __name__ == '__main__':
    init_db()
    app.run(debug=True)


