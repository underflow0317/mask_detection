# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def main():

    # set mask confidence rate
    confidence_rate=0.5

    # load our serialized face detector model from disk
    prototxtPath = "model_face_detector/deploy.prototxt"
    weightsPath = "model_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("masked_module")

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    vs.title="press q to leave"
    time.sleep(2.0)

    # loop over anf fetch each frame from the video stream
    while True:
      # 1. take a screenshot of the frame
      # screenshot the frame from the threaded video stream and resize it
      # to have a maximum width of 400 pixels
      frame = vs.read()
      frame = imutils.resize(frame, width=400)
      
      # 2. detect the masked faces by faceNet (detecting face) and maskNet (detecting mask)
      
      # locs: location of the face
      # preds: prediction for maksed faces (percent)
      # the return variable is an array, each element in this array means a face
      (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, confidence_rate)
      

      # loop over the detected face locations and their corresponding
      # locations
      for (box, pred) in zip(locs, preds):
      
        # 1. unpack the bounding box and predictions
        # the face location
        (startX, startY, endX, endY) = box
        # the precent of masked face
        (withoutMask,mask ) = pred

        # 2. determine the class label and color we'll use to draw
        #    the bounding box and text
        label_mask = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label_mask == "Mask" else (0, 0, 255)

        # 3. set label with percent: e.g., Mask: 56.22%
        label = "{}: {:.2f}%".format(label_mask, max(mask, withoutMask) * 100)

        # 4. display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
      pass
      # show the output frame
      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF

      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
        break
      pass
    pass
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
pass


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
  main()
pass
