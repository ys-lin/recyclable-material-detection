import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    classes = ['box', 'book']

    net = cv2.dnn.readNet('yolov4-obj_final.weights', 'yolov4-obj.cfg')
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            frame = cv2.VideoCapture(0)
            continue
        if ret:
            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)

            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            bounding_boxes = []
            confidences = []
            class_ids = []

            # first four elements in the output are bounding box coordinates
            # from 5th element on are scores
            for output in layerOutputs:
                for detection in output:
                    # store all detections for different classes
                    scores = detection[5:]
                    # the classes that has is the most likely
                    class_id = np.argmax(scores)
                    # extract the max scores
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        # center coordinate of the detected object
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        # width and height of the bounding box
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # position of the upper corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        bounding_boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            # keep the most probable boxes because we can have more than 1 box for the same object (?)
            indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(bounding_boxes), 3))

            # display bounding box on the image
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = bounding_boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    # create rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # font, 2, (255, 255, 255) , 2 => size 2, colour white, thickness 2
                    cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
            # cv2.imshow('Image', frame)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




# frame = cv2.imread('image.jpg')

# while True:
#     _, frame = cap.read()
#     height, width, _ = frame.shape
#
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#
#     net.setInput(blob)
#
#     output_layers_names = net.getUnconnectedOutLayersNames()
#     layerOutputs = net.forward(output_layers_names)
#
#     bounding_boxes = []
#     confidences = []
#     class_ids = []
#
#     # first four elements in the output are bounding box coordinates
#     # from 5th element on are scores
#     for output in layerOutputs:
#         for detection in output:
#             # store all detections for different classes
#             scores = detection[5:]
#             # the classes that has is the most likely
#             class_id = np.argmax(scores)
#             # extract the max scores
#             confidence = scores[class_id]
#
#             if confidence > 0.5:
#                 # center coordinate of the detected object
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#
#                 # width and height of the bounding box
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # position of the upper corner
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 bounding_boxes.append([x, y, w, h])
#                 confidences.append((float(confidence)))
#                 class_ids.append(class_id)
#
#     # keep the most probable boxes because we can have more than 1 box for the same object (?)
#     indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
#
#     font = cv2.FONT_HERSHEY_PLAIN
#     colors = np.random.uniform(0, 255, size=(len(bounding_boxes), 3))
#
#     # display bounding box on the image
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = bounding_boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = colors[i]
#             # create rectangle
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             # font, 2, (255, 255, 255) , 2 => size 2, colour white, thickness 2
#             cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
#
#     cv2.imshow('Image', frame)
#     # press esc to terminate
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cap.relaease()
# cv2.destroyAllWindows()
