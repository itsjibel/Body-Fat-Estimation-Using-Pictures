import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from Body_Part_Segmentation.config_reader import config_reader
from Body_Part_Segmentation.model_simulated_RGB101 import get_testing_model_resnet101
from Body_Part_Segmentation.human_seg.human_seg_gt import human_seg_combine_argmax
from Body_Part_Segmentation.human_seg.cropHuman import cropHuman

MODEL_PATH = 'Body_Part_Segmentation/weights/body_part_model.h5'
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = './output'
SCALE = [1.0]

human_part = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
human_ori_part = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
seg_num = 15

def recover_flipping_output2(oriImg, part_ori_size):
    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:, :, human_ori_part] = part_ori_size[:, :, human_part]
    return part_flip_size

def process(input_image, params, model_params):
    input_scale = 1.0

    oriImg = cv2.imread(input_image)
    oriImg, mask_img = cropHuman(oriImg)

    if (type(oriImg) == str):
        return oriImg, 0

    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]*input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]

        output_blobs = model.predict(input_img, verbose=False)
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        output_blobs = model.predict(input_img, verbose=False)

        # extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = recover_flipping_output2(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    return seg_avg, mask_img

def integration_depth_and_segmentation(depth_img, seg_argmax):
    body_pixels_depth_sum = 0
    num_of_image_pixels = depth_img.shape[0] * depth_img.shape[1]

    for x in range(depth_img.shape[0]):
        for y in range(depth_img.shape[1]):
            # If depth_img[x][y] is 0, the YOLO model predicted that this pixel is not part of a human.
            if depth_img[x][y] == 0:
                num_of_image_pixels -= 1

            body_pixels_depth_sum += depth_img[x][y]

    # Calculate the average depth of pixels of the person's body in the picture
    body_pixels_depth_avg = body_pixels_depth_sum / num_of_image_pixels

    # Class 1: Head, Class 2: torso
    class_numbers = [1, 2]
    for class_number in class_numbers:
        sum = 0
        count = 0

        # For each class, we iterate through the picture pixels for accurate segmentation
        for x in range(seg_argmax.shape[1]):
            for y in range(seg_argmax.shape[0]):
                class_predicted = seg_argmax[y][x]
                depth_of_pixel = depth_img[y][x]
                
                # We calculate the average depth of the pixels predicted as "class I" (I is class number) and their depth is slightly different from the body's average depth
                if class_predicted == class_number and abs(body_pixels_depth_avg - depth_of_pixel) < 25:
                    sum += depth_of_pixel
                    count += 1

        if count != 0:
            avg = sum / count
            for x in range(seg_argmax.shape[1]):
                for y in range(seg_argmax.shape[0]):
                    depth_of_pixel = depth_img[y][x]

                    # If this pixel didn't segment to anything and has a low depth difference with an average depth of "class I" and also is a neighbor with one of the pixels which segmented as "class I", then we segment it as "class I".
                    if abs(depth_of_pixel - avg) < 10 and seg_argmax[y][x] == 0 and (seg_argmax[y][x - 1] == class_number or seg_argmax[y][x + 1] == class_number or seg_argmax[y - 1][x] == class_number or seg_argmax[y + 1][x] == class_number):
                        seg_argmax[y][x] = class_number

                    # If this pixel is segmented as "class I" and the depth of it has a big difference from the average depth of "class I", so we segment it as "background".
                    if abs(depth_of_pixel - avg) > 35 and seg_argmax[y][x] == class_number:
                        seg_argmax[y][x] = 0

    return seg_argmax

def get_segmentation_of_image(filename):
    keras_weights_file = MODEL_PATH

    global model
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()

    scale_list = []
    for item in SCALE:
        scale_list.append(float(item))

    params['scale_search'] = scale_list

    seg, human_mask = process(INPUT_FOLDER + '/' + filename, params, model_params)
    seg_argmax = np.argmax(seg, axis=-1)

    # Load the depth image from the person's picture and apply the mask of the "human mask" (predicted from YOLO) to ignore the background pixels
    depth_img = cv2.imread('Depth_Estimation/output/' + filename.split('.')[0] + '_depth.png', cv2.IMREAD_GRAYSCALE)
    human_mask_gray = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
    depth_img = cv2.bitwise_and(depth_img, human_mask_gray)

    # Improve the segmentation accuracy
    seg_argmax = integration_depth_and_segmentation(depth_img, seg_argmax)

    return seg_argmax

def get_segmentation_of_frontal_image():
    return get_segmentation_of_image("Front.jpg")

def get_segmentation_of_left_image():
    return get_segmentation_of_image("Left.jpg")