import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from config_reader import config_reader
from model_simulated_RGB101 import get_testing_model_resnet101
from human_seg.human_seg_gt import human_seg_combine_argmax
from human_seg.cropHuman import cropHuman

MODEL_PATH = './weights/body_part_model.h5'
INPUT_FOLDER = './input'
OUTPUT_FOLDER = './output'
SCALE = [1.0]

# 1: Background, 2: Head, 3: Torso, 4: Right arm, 5: Left arm, 6: Right forearm,
# 7: Left forearm, 8: Right hand, 9: Left hand, 10: Right thigh, 11: Left thigh
# 12: Right shank, 13: Left shank, 14: Right feet, 15: Left feet
torso_confidence_rate = 0.65
right_arm_confidence_rate = 0.6
left_arm_confidence_rate = 0.6
right_forearm_confidence_rate = 0.6
left_forearm_confidence_rate = 0.6
right_thigh_confidence_rate = 0.55
left_thigh_confidence_rate = 0.55
right_shank_confidence_rate = 0.55
left_shank_confidence_rate = 0.55


classes_confidence_rate = [0, 0, torso_confidence_rate, right_arm_confidence_rate, left_arm_confidence_rate,
                           right_forearm_confidence_rate, left_forearm_confidence_rate, 0, 0,
                           right_thigh_confidence_rate, left_thigh_confidence_rate, right_shank_confidence_rate, left_shank_confidence_rate, 0, 0]
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


if __name__ == '__main__':
    keras_weights_file = MODEL_PATH

    print('Loding model...', end=' ', flush=True)
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()

    scale_list = []
    for item in SCALE:
        scale_list.append(float(item))

    params['scale_search'] = scale_list
    print('Done!')

    # generate image with body parts
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            print('Segmenting', filename + '...', end=' ', flush=True)
            seg, mask_img = process(INPUT_FOLDER + '/' + filename, params, model_params)
            if type(seg) == str:
                print(seg)
                continue
            else:
                print('Done!')

            depth_img = cv2.imread('Depth Estimation/output/' + filename.split('.')[0] + '_depth.png', cv2.IMREAD_GRAYSCALE)
            
            # Find the depth of pixels of a person's body and then find the average value of them
            mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            depth_img = cv2.bitwise_and(depth_img, mask_img_gray)

            sum = 0
            image_pixels = depth_img.shape[0] * depth_img.shape[1]

            for x in range(depth_img.shape[0]):
                for y in range(depth_img.shape[1]):
                    if depth_img[x][y] == 0:
                        image_pixels -= 1
                    sum += depth_img[x][y]

            avg = sum / image_pixels

            # Normalize the depth values with average depth values of the person's body to get a better prediction on segmentation
            depth_img = depth_img / avg

            seg_argmax = np.argmax(seg, axis=-1)

            # Check the confidence rate of the prediction
            for x in range(seg_argmax.shape[1]):
                for y in range(seg_argmax.shape[0]):
                    class_predicted = seg_argmax[y][x]
                    depth_of_pixel = depth_img[y][x]

                    # Multiply the depth normalized value of this pixel by the prediction confidence,
                    # if the depth normalized value is < 0, then the confidence rate decreases and increases otherwise.
                    confidence_rate = seg[y][x][class_predicted] * depth_of_pixel
                    if confidence_rate < classes_confidence_rate[class_predicted]:
                        seg_argmax[y][x] = 0

            seg_canvas = human_seg_combine_argmax(seg_argmax)
            cur_canvas = cv2.imread(INPUT_FOLDER + '/' + filename)
            canvas = cv2.addWeighted(seg_canvas, 0.35, cur_canvas, 1, 0)
            file_output_location = '%s/%s'%(OUTPUT_FOLDER, 'seg_' + filename)

            # Draw body joints
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

            # convert the frame to RGB format
            RGB = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            # process the RGB frame to get the result
            results = pose.process(RGB)

            # draw detected skeleton on the frame
            mp_drawing.draw_landmarks(canvas, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imwrite(file_output_location, canvas)
