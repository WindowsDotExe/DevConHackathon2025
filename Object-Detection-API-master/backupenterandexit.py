import time
import cv2
import numpy as np
import tensorflow as tf
import sys
from absl import app, flags, logging
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import openai

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

def main(argv):
    del argv  # Ignore argv to prevent conflicts

    logging.info("Parsing command-line flags...")

    # Set GPU configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # ✅ Initialize YOLO after parsing FLAGS
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('YOLO weights loaded')

    # Load class names
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('Class names loaded')

    # Initialize video capture
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))  # Webcam
    except:
        vid = cv2.VideoCapture(FLAGS.video)  # Video file

    if not vid.isOpened():
        logging.error("Error: Unable to open video source.")
        sys.exit()

    # Get video properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if output is enabled
    out = None
    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # Passenger counting setup
    count_enter = 0
    count_exit = 0
    line_position = width // 2  # Set vertical counting line at the center

    # Dictionary to track people
    people_tracking = {}
    next_person_id = 0

    # Initialize logging
    log_file = open("passenger_count_log.csv", "w")
    log_file.write("timestamp,event,enter_count,exit_count\n")

    while True:
        ret, img = vid.read()
        if not ret:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        # Preprocess image for YOLOv3
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # Perform YOLO detection
        boxes, scores, classes, nums = yolo.predict(img_in)

        # Store detected people in this frame
        current_people = []

        for i in range(nums[0]):
            if int(classes[0][i]) == 0:  # Only detect 'person' class
                box = boxes[0][i]
                x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                center_x = (x1 + x2) // 2
                current_people.append((center_x, y1, x2 - x1, y2 - y1))

        # Update tracking information
        updated_tracking = {}
        for center_x, y1, w, h in current_people:
            matched_id = None

            # Try to match with an existing person
            for person_id, data in people_tracking.items():
                prev_x, prev_y, _, _ = data["bbox"]
                distance = abs(center_x - prev_x)  # Check horizontal movement
                if distance < 50:  # Threshold for being the same person
                    matched_id = person_id
                    break

            if matched_id is None:
                # New person detected
                matched_id = next_person_id
                next_person_id += 1

            updated_tracking[matched_id] = {
                "bbox": (center_x, y1, w, h),
                "previous_x": people_tracking.get(matched_id, {}).get("previous_x", center_x),
                "counted": people_tracking.get(matched_id, {}).get("counted", False),
            }

        # Assign the updated tracking info back
        people_tracking = updated_tracking

        # Check if someone has crossed the line
        for person_id, data in people_tracking.items():
            center_x, _, _, _ = data["bbox"]
            previous_x = data["previous_x"]

            if not data["counted"]:  
                if previous_x < line_position and center_x > line_position:  # LEFT → RIGHT (Enter)
                    count_enter += 1
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},enter,{count_enter},{count_exit}\n")
                    print(f"🔵 Person ENTERED (Total: {count_enter})")  # Debugging log
                    people_tracking[person_id]["counted"] = True  # Mark as counted
                
                elif previous_x > line_position and center_x < line_position:  # RIGHT → LEFT (Exit)
                    count_exit += 1
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},exit,{count_enter},{count_exit}\n")
                    print(f"🔴 Person EXITED (Total: {count_exit})")  # Debugging log
                    people_tracking[person_id]["counted"] = True  # Mark as counted

            people_tracking[person_id]["previous_x"] = center_x  # Update last position

        # Draw detections and line
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.line(img, (line_position, 0), (line_position, height), (255, 0, 0), 2)  # Draw vertical counting line

        # Display counts
        cv2.putText(img, f"Entered: {count_enter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Exited: {count_exit}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show video feed
        cv2.imshow("Passenger Counter", img)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    log_file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
