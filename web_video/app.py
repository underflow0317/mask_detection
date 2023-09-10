# 导入所需的库
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

# 加载口罩检测模型和人脸检测模型
prototxtPath = "../model_face_detector/deploy.prototxt"
weightsPath = "../model_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("../masked_module")

# 打开默认摄像头（摄像头编号为0）
camera = cv2.VideoCapture(0)

# 用于标志是否停止线程
thread_stop = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global thread_stop  # 声明为全局变量
    while True:
        if thread_stop:
            break

        # 从摄像头读取一帧图像
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
  # 1. get the frame shape (height and weight)
  (h, w) = frame.shape[:2]
  
  # 2. image preprocess to normalize "different lightness of the image" by cv2.dnn.blobFromImage
  #    frame: source of the image
  #    1.0: scalefactor resize the image (1=>100%)
  #    (300,300): size of the image
  #    (104.0, 177.0, 123.0): mean light of RGB 
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

  # 3. to find the position of the face
  faceNet.setInput(blob)
  detections = faceNet.forward()

  # 4. store the face
  
  # 4.1 initialize our list of faces, their corresponding locations,
  # and the list of predictions from our face mask network
  faces = []
  locs = []
  preds = []

  # 4.2 loop over the detections
  for i in range(0, detections.shape[2]):
    
    # 4.3. extract the confidence (i.e., probability) associated with
    # the detection of the possible face
    confidence = detections[0, 0, i, 2]

    # 4.4. pick up face detections only when the confidence is
    # greater than the minimum confidence
    if confidence > confidence_rate:
    
      # 4.5. find the position of the face
      # compute the (x, y)-coordinates of the bounding box for
      # the object
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      # 4.6. get the bounding boxes fall within the dimensions of
      # the frame
      (startX, startY) = (max(0, startX), max(0, startY))
      (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

      # 4.7. get the face in the box
      # extract the face ROI, convert it from BGR to RGB channel
      # ordering, resize it to 224x224, and preprocess it
      face = frame[startY:endY, startX:endX]
      face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      face = cv2.resize(face, (224, 224))
      face = img_to_array(face)
      face = preprocess_input(face)

      # 4.8. add the face and bounding boxes to their respective
      # lists
      faces.append(face)
      locs.append((startX, startY, endX, endY))
    pass
  pass
  # only make a predictions if at least one face was detected
  if len(faces) > 0:
    # for faster inference we'll make batch predictions on *all*
    # faces at the same time rather than one-by-one predictions
    # in the above `for` loop
    input_batch=len(faces)
    faces = np.array(faces, dtype="float32")
    input_x=tf.constant(faces, shape=[input_batch,224,224,3], dtype=tf.float32) 
    preds = maskNet.predict(input_x)
  pass
  
  # return a 2-tuple of the face locations and their corresponding
  # locations
  return (locs, preds)
pass


if __name__ == '__main__':
    # 使用多线程运行Flask应用
    app.run(threaded=True, debug=True)
