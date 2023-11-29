import cv2
import numpy as np


def visualize_face(img, detection, thickness=2, pad=0.3, draw=False):
    points = detection['keypoints']
    x, y, w, h = detection['box']
    mid_eye_y = int((points['left_eye'][1] + points['right_eye'][1]) / 2)
    mid_eye_x = int((points['left_eye'][0] + points['right_eye'][0]) / 2)
    mid_mouth_y = int((points['mouth_left'][1] + points['mouth_right'][1]) / 2)

    if draw:
        img = cv2.circle(img, points['left_eye'], 2, (255, 0, 0), thickness)
        img = cv2.circle(img, points['right_eye'], 2, (255, 0, 0), thickness)
        img = cv2.circle(img, points['nose'], 2, (0, 255, 0), thickness)
        img = cv2.circle(img, points['mouth_left'], 2, (0, 0, 255), thickness)
        img = cv2.circle(img, points['mouth_right'], 2, (0, 0, 255), thickness)

    w = w + 2 * int(w * pad)
    h = h + 2 * int(h * pad)

    x = points['nose'][0] - int(w / 2)
    y = int((mid_eye_y + mid_mouth_y) / 2 - h / 2)

    slope = (points['right_eye'][1] - points['left_eye'][1]) / (points['right_eye'][0] - points['left_eye'][0])
    angle = np.arctan(slope) * 180 / np.pi

    centre = (mid_eye_x - x, int(h / 2))
    crop = img[y:y + h, x:x + w]

    return crop, angle, centre


def rotate(angle, img, centre):

    y, x, channels = img.shape

    M = cv2.getRotationMatrix2D(centre, angle, 1)
    img = cv2.warpAffine(img, M, (x, y))

    return img


def rotate_face(face_img, angle, bounding_box, centre):

    face = rotate(angle, face_img, centre)

    x, y, w, h = bounding_box
    x = centre[0] - int(w/2)
    y = centre[1] - int(h/2)

    return face[y:y+h, x:x+w]


def gaussian(left_r=-80, right_r=80, sigma=75):
    x = np.array([i for i in range(left_r, right_r)])

    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x ** 2) / (2 * sigma ** 2))
    y = y.reshape((y.shape[0], 1))

    return y
