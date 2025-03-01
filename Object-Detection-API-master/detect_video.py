import time
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import datetime
import base64
from openai import OpenAI
from absl import app, flags, logging
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

# ✅ Initialize OpenAI Client
client = OpenAI()  # Uses the API key from environment variable

# Define command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('classes', './data/labels/coco.names', 'Path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf', 'Path to YOLOv3 weights file')
flags.DEFINE_boolean('tiny', False, 'Use YOLOv3-tiny')
flags.DEFINE_integer('size', 416, 'Image resize size')
flags.DEFINE_string('video', '0', 'Path to video file or webcam index (0 for default webcam)')
flags.DEFINE_string('output', None, 'Path to save output video')
flags.DEFINE_string('output_format', 'XVID', 'Codec for saving video')
flags.DEFINE_integer('num_classes', 80, 'Number of classes in the YOLO model')

def encode_image(image_path):
    """Convert image to Base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_alert_message(image_path):
    """
    Sends the image to OpenAI's GPT-4o for analysis and returns an alert message.
    """
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is happening in the image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=300,
        )

        alert_msg = response.choices[0].message.content.strip()
        return alert_msg

    except Exception as e:
        return f"⚠️ Error generating AI alert: {str(e)}"

def main(argv):
    del argv  # Ignore argv to prevent conflicts

    logging.info("Parsing command-line flags...")

    # ✅ Set GPU configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # ✅ Initialize YOLO Model
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('YOLO weights loaded')

    # ✅ Load class names
    class_names = [c.strip() for c in open(FLAGS.classes, encoding="utf-8").readlines()]
    logging.info('Class names loaded')

    # ✅ Initialize video capture
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))  # Webcam
    except:
        vid = cv2.VideoCapture(FLAGS.video)  # Video file

    if not vid.isOpened():
        logging.error("Error: Unable to open video source.")
        sys.exit()

    # ✅ Get video properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # ✅ Initialize video writer if output is enabled
    out = None
    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # ✅ Passenger counting setup
    count_enter = 0
    count_exit = 0
    line_position = width // 2  # Vertical counting line at the center

    # ✅ Tracking people
    people_tracking = {}
    next_person_id = 0

    # ✅ Weapon detection variables
    weapon_detected = False
    weapon_captured = False
    log_file_path = "alerts.log"

    while True:
        ret, img = vid.read()
        if not ret:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        # ✅ Preprocess image for YOLOv3
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # ✅ Perform YOLO detection
        boxes, scores, classes, nums = yolo(img_in)
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()

        # ✅ Print detections
        print(f"Detections found: {nums[0]}")
        for i in range(nums[0]):
            print(f"Class: {class_names[int(classes[0][i])]}, Confidence: {scores[0][i]}")

        # ✅ Detect people and weapons (backpacks)
        for i in range(nums[0]):
            detected_class = int(classes[0][i])
            confidence = scores[0][i]

            if confidence > 0.3:  # Threshold
                if detected_class == 24:  # Backpack detected (simulated weapon)
                    weapon_detected = True
                    x1, y1, x2, y2 = int(boxes[0][i][1] * width), int(boxes[0][i][0] * height), int(boxes[0][i][3] * width), int(boxes[0][i][2] * height)

                    # ✅ Draw bounding box for detected backpack
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "Weapon Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Capture and analyze frame if weapon is detected
        if weapon_detected and not weapon_captured:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"weapon_detected_{timestamp}.jpg"
            cv2.imwrite(image_filename, img)
            weapon_captured = True  # Avoid duplicate captures

            # ✅ Send image to OpenAI GPT-4o for analysis
            alert_message = generate_alert_message(image_filename)
            print(f"[ALERT] {alert_message}")

            # ✅ Log alert message
            log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file_path, 'a', encoding='utf-8') as logf:
                logf.write(f"{log_time} - {alert_message}\n")

        # ✅ Show video feed
        cv2.imshow("Passenger Counter", img)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
