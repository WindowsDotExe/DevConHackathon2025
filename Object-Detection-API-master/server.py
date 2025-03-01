import cv2
import time
import threading
import base64
import datetime
import numpy as np
from flask import Flask, Response
from flask_socketio import SocketIO
from openai import OpenAI
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows all origins

# Initialize Flask & SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO Model
yolo = YoloV3(classes=80)
yolo.load_weights('./weights/yolov3.tf')

# Initialize OpenAI Client
client = OpenAI()

# Open video source (webcam or video file)
cap = cv2.VideoCapture(0)

def encode_image(image):
    """Encode image to Base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

def generate_alert_message(image):
    """Send the image to OpenAI GPT-4o and return the alert message."""
    base64_image = encode_image(image)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is happening in this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating AI alert: {str(e)}"

def detect_objects():
    """Continuously detect objects and send alerts via WebSocket."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess image for YOLO
        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(img_in, axis=0)
        img_in = transform_images(img_in, 416)

        # Run YOLO detection
        boxes, scores, classes, nums = yolo.predict(img_in)
        detected_classes = [int(classes[0][i]) for i in range(nums[0])]

        # Simulate detecting a "weapon" (Backpack = class 24)
        if 24 in detected_classes:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert_msg = generate_alert_message(frame)

            # Send alert via WebSockets
            socketio.emit('alert', {'message': f"🚨 {alert_msg} at {timestamp}"})
            print(f"[ALERT] {alert_msg}")

        time.sleep(1)  # Avoid excessive processing

def generate_video_stream():
    """Generate MJPEG video stream."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Serve the video stream."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Video Streaming Server Running!"

if __name__ == '__main__':
    # Start YOLO object detection in a separate thread
    threading.Thread(target=detect_objects, daemon=True).start()
    
    # Run Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)