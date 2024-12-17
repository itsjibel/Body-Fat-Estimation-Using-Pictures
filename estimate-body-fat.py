print('Importing libraries...', flush=True, end=' ')
from Pixel_Size_Estimation_In_cm import estimate_pixel_size as PixelSizeEstimaion
from Body_Part_Segmentation import body_parts_segmentation as BodyPartSegmentation
from Parametric_Body_Fat_Estimation.src import parametric_body_fat_estimation as ParametricBodyFatEstimation
import math
import cv2
import mediapipe as mp
print('Done!')

HEIGHT = 190
SEX = 'male' # male or female

print("Calculating the pixel size in cm (frontal  picture)...", flush=True, end=' ')
frontal_image_pixel_size = PixelSizeEstimaion.calculateFrontalImagePixelSize(HEIGHT)
print("Done!")

print("Calculating the pixel size in cm (left side picture)...", flush=True, end=' ')
left_image_pixel_size = PixelSizeEstimaion.calculateLeftImagePixelSize(HEIGHT)
print("Done!")

print("Segmenting the body parts (frontal picture)...", flush=True, end=' ')
frontal_image_segments = BodyPartSegmentation.get_segmentation_of_frontal_image()
print("Done!")

print("Segmenting the body parts (left side picture)...", flush=True, end=' ')
left_image_segments = BodyPartSegmentation.get_segmentation_of_left_image()
print("Done!")

print("Calculating neck circumference...", flush=True, end=' ')
neck_bottom_left_x = 0
neck_bottom_left_y = 0
neck_bottom_right_x = 0
neck_bottom_right_y = 0

max_y_left = 0
max_y_right = 0

for y in range(frontal_image_segments.shape[0]):
    for x in range(frontal_image_segments.shape[1]):
        if frontal_image_segments[y][x] == 2:
            break

        if frontal_image_segments[y][x] == 1 and y > max_y_left:
            neck_bottom_left_x = x
            neck_bottom_left_y = y
            break

    for x in range(frontal_image_segments.shape[1] - 1, -1, -1):
        if frontal_image_segments[y][x] == 2:
            break

        if frontal_image_segments[y][x] == 1 and y > max_y_right:
            neck_bottom_right_x = x
            neck_bottom_right_y = y
            break

neck_frontal_radius_in_pixel = math.sqrt((neck_bottom_left_x - neck_bottom_right_x) * (neck_bottom_left_x - neck_bottom_right_x) + (neck_bottom_left_y - neck_bottom_right_y) * (neck_bottom_left_y - neck_bottom_right_y)) / 2
neck_frontal_radius_in_cm = neck_frontal_radius_in_pixel * frontal_image_pixel_size[0]

neck_bottom_left_x = 0
neck_bottom_left_y = 0
neck_bottom_right_x = 0
neck_bottom_right_y = 0

max_y_left = 0
max_y_right = 0

for y in range(left_image_segments.shape[0]):
    for x in range(left_image_segments.shape[1]):
        if left_image_segments[y][x] == 2:
            break

        if left_image_segments[y][x] == 1 and y > max_y_left:
            neck_bottom_left_x = x
            neck_bottom_left_y = y
            break

    for x in range(left_image_segments.shape[1] - 1, -1, -1):
        if left_image_segments[y][x] == 2:
            break

        if left_image_segments[y][x] == 1 and y > max_y_right:
            neck_bottom_right_x = x
            neck_bottom_right_y = y
            break

neck_side_radius_in_pixel = math.sqrt((neck_bottom_left_x - neck_bottom_right_x) * (neck_bottom_left_x - neck_bottom_right_x) + (neck_bottom_left_y - neck_bottom_right_y) * (neck_bottom_left_y - neck_bottom_right_y)) / 2
neck_side_radius_in_cm = neck_side_radius_in_pixel * left_image_pixel_size

neck_circumference = math.pi * (3.33 * (neck_side_radius_in_cm + neck_frontal_radius_in_cm) - math.sqrt((3.33 * neck_side_radius_in_cm + neck_frontal_radius_in_cm) * (neck_side_radius_in_cm + 3.33 * neck_frontal_radius_in_cm)))
print(f"({int(neck_circumference)}cm) Done!")

print("Calculating abdomen circumference...", flush=True, end=' ')
image = cv2.imread('input/Left.jpg')
image_height, image_width, _ = image.shape
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
landmarks = results.pose_landmarks.landmark

left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width

left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width

abdomen_point_y = int((left_shoulder_y + left_hip_y) / 2)

start_point_x = 0
end_point_x = 0

for x in range(image.shape[1]):
    if left_image_segments[abdomen_point_y][x] == 2:
        start_point_x = x
        break

for x in range(image.shape[1] - 1, -1, -1):
    if left_image_segments[abdomen_point_y][x] == 2:
        end_point_x = x
        break

abdomen_side_radius_in_pixel = (end_point_x - start_point_x) / 2
abdomen_side_radius_in_cm = abdomen_side_radius_in_pixel * left_image_pixel_size

image = cv2.imread('input/Front.jpg')
image_height, image_width, _ = image.shape
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
landmarks = results.pose_landmarks.landmark

left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width

left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width

abdomen_point_y = int((left_shoulder_y + left_hip_y) / 2)

start_point_x = 0
end_point_x = 0

for x in range(image.shape[1]):
    if frontal_image_segments[abdomen_point_y][x] == 2:
        start_point_x = x
        break

for x in range(image.shape[1] - 1, -1, -1):
    if frontal_image_segments[abdomen_point_y][x] == 2:
        end_point_x = x
        break

abdomen_front_radius_in_pixel = (end_point_x - start_point_x) / 2
abdomen_front_radius_in_cm = abdomen_front_radius_in_pixel * frontal_image_pixel_size[0]

abdomen_circumference = math.pi * (3.33 * (abdomen_front_radius_in_cm + abdomen_side_radius_in_cm) - math.sqrt((3.33 * abdomen_front_radius_in_cm + abdomen_side_radius_in_cm) * (abdomen_front_radius_in_cm + 3.33 * abdomen_side_radius_in_cm)))

print(f"({int(abdomen_circumference)}cm) Done!")

print("Predicting the body fat percentage...", flush=True, end=' ')
bodyFatPercentage = ParametricBodyFatEstimation.parametricBodyFatEstimation(SEX, HEIGHT / 100, neck_circumference, abdomen_circumference)
print("Done!")

print(f'Body fat percentage of this person is {int(bodyFatPercentage)}%')