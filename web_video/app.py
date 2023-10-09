# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import threading
import tensorflow as tf

app = Flask(__name__)

# Load the face detection model and mask detection model
prototxtPath = "../model_face_detector/deploy.prototxt"
weightsPath = "../model_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("../masked_module")

# Open the default camera (camera number 0)
camera = cv2.VideoCapture(0)

# Used to flag if the thread should stop
thread_stop = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global thread_stop  # Declare as a global variable
    while True:
        if thread_stop:
            break

        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=1024)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, confidence_rate=0.5)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (withoutMask, mask) = pred
                label_mask = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label_mask == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label_mask, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_and_predict_mask(frame, faceNet, maskNet, confidence_rate):
    # 1. Get the frame shape (height and weight)
    (h, w) = frame.shape[:2]

    # 2. Image preprocessing to normalize "different lightness of the image" by cv2.dnn.blobFromImage
    #    frame: source of the image
    #    1.0: scale factor to resize the image (1=>100%)
    #    (300, 300): size of the image
    #    (104.0, 177.0, 123.0): mean lightness of RGB
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # 3. Find the position of the face
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # 4. Store the face

    # 4.1 Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # 4.2 Loop over the detections
    for i in range(0, detections.shape[2]):

        # 4.3 Extract the confidence (i.e., probability) associated with
        # the detection of the possible face
        confidence = detections[0, 0, i, 2]

        # 4.4 Pick up face detections only when the confidence is
        # greater than the minimum confidence
        if confidence > confidence_rate:

            # 4.5 Find the position of the face
            # Compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 4.6 Get the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # 4.7 Get the face in the box
            # Extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # 4.8 Add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    pass
    pass
    # Only make predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference, make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        input_batch = len(faces)
        faces = np.array(faces, dtype="float32")
        input_x = tf.constant(faces, shape=[input_batch, 224, 224, 3], dtype=tf.float32)
        preds = maskNet.predict(input_x)
    pass

    # Return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)
pass

if __name__ == '__main__':
    # Run the Flask application using multiple threads
    app.run(threaded=True, debug=True)
