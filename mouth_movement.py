import numpy as np
import cv2 as cv
import time
from helper import *
import numpy as np
from matplotlib import pyplot as plt
from statistics import median
from keras_facenet import FaceNet


path = 0

def visualize(input, faces, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    # cv.putText(input, (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def gaussian(left_r=-80, right_r=80, sigma=75):
    x = np.array([i for i in range(left_r, right_r)])

    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x ** 2) / (2 * sigma ** 2))
    y = y.reshape((y.shape[0], 1))

    return y

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

activity = []
avg = []
med = []
X = []
prev_ratio = 0
frameCounter = 0
fps = 0
print("[INFO] Starting video...")

while True:
    ret, frame = cap.read()
    frameCounter += 1
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 20, 3, 5, 1.1, 1)
    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)
    old_gray = frame_gray.copy()

    faces, crops = detector.crop(frame, threshold=0.99)
    face_images = None

    ratio = 0
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face['box']

            (startX, startY, endX, endY) = (x, y, x + w, y + h)

            cropped_face, angle, centre = visualize_face(frame, face)
            img = rotate_face(cropped_face, angle, face['box'], centre)
            img = cv.resize(img, (120, 160), interpolation=cv.INTER_CUBIC)

            cv.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            heat_map = np.zeros_like(frame[startY:endY, startX:endX])
            heat_map[..., 1] = 0
            heat_map[..., 0] = 0
            heat_map[..., 2] = cv.normalize(abs(fy[startY:endY, startX:endX]), None, 0, 255, cv.NORM_MINMAX)

            m = heat_map[..., 2]
            image = cv.resize(m, (120, 160), interpolation=cv.INTER_CUBIC)
            (h, w) = image.shape
            y_vertical = gaussian(left_r=-60, right_r=60, sigma=30)
            y_horizontal = gaussian(left_r=-50, right_r=30, sigma=20)

            score2 = np.sum(y_vertical * image[0:int(h / 2), 0:w].T)
            score1 = np.sum(y_vertical * image[int(h / 2):h, 0:w].T)

            score2 += np.sum(y_horizontal * image[0:int(h / 2), 0:w])
            score1 += np.sum(y_horizontal * image[int(h / 2):h, 0:w])

            # print(min(15, score1 / score2), score2, score1)
            ratio = min(15, score1 / score2)
            fps = cap.get(cv.CAP_PROP_FPS)

            face_images = np.concatenate((image, cv.cvtColor(img, cv.COLOR_BGR2GRAY)), axis=1)

    if face_images is not None:
        cv.imshow("image", face_images)

    activity.append(ratio)
    if frameCounter % 10 == 0:
        X.append(frameCounter)
        avg.append(sum(activity[-int(10):]) / (10))
        med.append(median(activity[-int(10):]))

    if sum(activity[-int(10):]) > 30:
        cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

    info = "No. of Frames : {}".format(frameCounter)
    cv.putText(frame, info, (10, frame.shape[0]-60), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    info = "FPS : {}".format(fps)
    cv.putText(frame, info, (10, frame.shape[0]-40), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    info = "Activity for 10 frames : {}".format(int(sum(activity[-int(fps+10):])))
    cv.putText(frame, info, (10, frame.shape[0]-20), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Cam", frame)

    # visualize(frame, faces)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

print("[INFO] Closing camera...")
cap.release()
cv.destroyAllWindows()

plt.plot(activity, linewidth=1.0)
plt.title("Activity at each frame")
plt.ylabel('Disturbance in pixels')
plt.xlabel('Frame number')
plt.show()

plt.plot(X, avg, linewidth=1.0)
plt.title("Average (window size = 40)")
plt.ylabel('Disturbance in pixels')
plt.xlabel('Frame number')
plt.show()

plt.plot(X, med, linewidth=1.0)
plt.title("Median (window size = 40)")
plt.ylabel('Disturbance in pixels')
plt.xlabel('Frame number')
plt.show()
