import numpy as np
import cv2 as cv
import time
from helper import *
from keras_facenet import FaceNet


path = 0

cap = cv.VideoCapture(path)
print("[INFO] Opening camera...")
time.sleep(1.0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
font = cv.FONT_HERSHEY_SIMPLEX

detector = FaceNet()

frameCounter = 0
fps = 0
print("[INFO] Starting video...")

while True:
    ret, frame = cap.read()
    frameCounter += 1
    if not ret:
        print('No frames grabbed!')
        break

    faces, crops = detector.crop(frame, threshold=0.99)
    face_images = None

    if len(faces) > 0:

        for face in faces:
            x, y, w, h = face['box']

            (startX, startY, endX, endY) = (x, y, x + w, y + h)

            cropped_face, angle, centre = visualize_face(frame, face)
            img = rotate_face(cropped_face, angle, face['box'], centre)

            image = cv.resize(img, (120, 160), interpolation=cv.INTER_CUBIC)

            if face_images is None:
                face_images = image
            else:
                face_images = np.concatenate((image, face_images), axis=1)

            fps = cap.get(cv.CAP_PROP_FPS)
            cv.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)

    if face_images is not None:
        cv.imshow("image", face_images)

    info = "No. of Frames : {}".format(frameCounter)
    cv.putText(frame, info, (10, frame.shape[0]-60), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    info = "FPS : {}".format(fps)
    cv.putText(frame, info, (10, frame.shape[0]-40), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Cam", frame)

    # visualize(frame, faces)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

print("[INFO] Closing camera...")
cap.release()
cv.destroyAllWindows()

