from Pixel_Size_Estimation_In_cm import estimate_pixel_size as PixelSizeEstimaion
from Body_Part_Segmentation import body_parts_segmentation as BodyPartSegmentation
import math
HEIGHT = 190

frontal_image_pixel_size = PixelSizeEstimaion.calculateFrontalImagePixelSize(HEIGHT)
left_image_pixel_size = PixelSizeEstimaion.calculateLeftImagePixelSize(HEIGHT)

frontal_image_segments = BodyPartSegmentation.get_segmentation_of_frontal_image()
left_image_segments = BodyPartSegmentation.get_segmentation_of_left_image()

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

neck_frontal_radius_in_pixel = math.sqrt((neck_bottom_left_x - neck_bottom_right_x) * (neck_bottom_left_x - neck_bottom_right_x) + (neck_bottom_left_y - neck_bottom_right_y) * (neck_bottom_left_y - neck_bottom_right_y))
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

neck_side_radius_in_pixel = math.sqrt((neck_bottom_left_x - neck_bottom_right_x) * (neck_bottom_left_x - neck_bottom_right_x) + (neck_bottom_left_y - neck_bottom_right_y) * (neck_bottom_left_y - neck_bottom_right_y))
neck_side_radius_in_cm = neck_side_radius_in_pixel * left_image_pixel_size

neck_circumference = 2 * (neck_frontal_radius_in_cm + neck_side_radius_in_cm)