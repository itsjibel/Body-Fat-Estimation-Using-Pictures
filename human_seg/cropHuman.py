from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolov8x-seg.pt")


def cropHuman(img):
    # Make a mask of all zeros (black), same size as the image
    mask_img = np.zeros_like(img)

    # Set confidence threshold
    conf = 0.5

    # Get the prediction results
    results = model.predict(img, conf=conf, verbose=False)

    # Get the class ID for 'person'
    person_class_id = list(model.names.values()).index("person")

    person_count = 0

    # Iterate over the results
    for result in results:
        if result.masks != None:
            for mask, box in zip(result.masks.xy, result.boxes):
                # Check if the detected class is 'person'
                if int(box.cls[0]) == person_class_id:
                    points = np.int32([mask])
                    cv2.fillPoly(mask_img, points, (255, 255, 255))  # White for person
                    person_count += 1

    # Create a final image where the person remains and other pixels are black
    final_img = cv2.bitwise_and(img, mask_img)

    if person_count == 1:
        return final_img, mask_img
    elif person_count == 0:
        return "There is no person in the picture.", 0
    else:
        return "There is more than one person in the picture.", 0