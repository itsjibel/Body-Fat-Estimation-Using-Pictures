import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp

def calculateFrontalImagePixelSize(personHeightInCm):
    image = cv2.imread('input/Front.jpg')
    image_height, image_width, _ = image.shape
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = results.pose_landmarks.landmark

    head_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
    foot_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * image_height
    pixel_size = abs(head_y - foot_y) / personHeightInCm
    pixel_size = 1 / pixel_size

    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
    left_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
    left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
    left_shank_length_in_cm = math.sqrt((left_knee_y - left_ankle_y) * (left_knee_y - left_ankle_y) + (left_knee_x - left_ankle_x) * (left_knee_x - left_ankle_x)) * pixel_size

    return pixel_size, left_shank_length_in_cm

def calculateLeftImagePixelSize(personHeightInCm):
    image = cv2.imread('input/Left.jpg')
    image_height, image_width, _ = image.shape
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = results.pose_landmarks.landmark

    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
    left_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
    left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
    left_shank_length_in_pixel = math.sqrt((left_knee_y - left_ankle_y) * (left_knee_y - left_ankle_y) + (left_knee_x - left_ankle_x) * (left_knee_x - left_ankle_x))
    _, left_shank_length_in_cm = calculateFrontalImagePixelSize(personHeightInCm)

    pixel_size = left_shank_length_in_pixel / left_shank_length_in_cm
    pixel_size = 1 / pixel_size

    return pixel_size