from inference.functions.crop_utils import *
import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.cluster import KMeans

def kernel_length_width(img=None, inst_map=None, pos=None, ans=None):
    degree = 0
    y1, y2, x1, x2 = pos

    # start = time.time()
    # crop_img = img.copy()
    crop_img = img[y1:y2, x1:x2]
    # temp_inst = inst_map.copy()
    temp_inst = inst_map[y1:y2, x1:x2]
    # print(time.time() - start)

    mask_img = np.zeros_like(crop_img)
    mask_img[np.where(temp_inst == 1)] = crop_img[np.where(temp_inst == 1)]

    crop_img = mask_img
    crop_thres = temp_inst.astype("uint8")

    # crop_thres = inst_map[y1:y2, x1:x2].astype('uint8')
    # mask_img = np.zeros_like(img)
    # mask_img[np.where(inst_map==1)] = img[np.where(inst_map==1)]
    # Crop Image from origin picture and Rotate
    # crop_img = mask_img[y1:y2, x1:x2]
    # crop_thres = inst_map[y1:y2, x1:x2].astype('uint8')

    # start = time.time()
    # Find Most contour area
    contours = contour_area(crop_thres)

    # Find Degree to rotate
    degree = rotated_degree(degree, contours)

    rotated = ndimage.rotate(crop_img, degree - 90)
    rotated_thres = ndimage.rotate(crop_thres, degree - 90)
    contours1 = contour_area(rotated_thres)

    # Mask only Grain
    topx, bottomx, topy, bottomy, out = mark_only_grain(rotated, contours1)
    out2 = out[topx:bottomx, topy:bottomy]

    if out2.shape[0] > 224:
        temp_out2 = cv2.cvtColor(out2, cv2.COLOR_RGB2GRAY)
        first_out2 = temp_out2[:int(temp_out2.shape[0]/2)]
        sec_out2 = temp_out2[int(temp_out2.shape[0]/2):]
        if first_out2[first_out2>0].shape[0] < sec_out2[sec_out2>0].shape[0]:
            out2 = ndimage.rotate(out2, 180)
        out2 = out2[:224]
        
    size = 224
    dst = np.zeros((size, size, 3))
    pt2 = [out2.shape[1] / 2, out2.shape[0] / 2]
    pt1 = [size / 2, size / 2]

    # (2) Calc offset
    dx = int(pt1[0] - pt2[0])
    dy = int(pt1[1] - pt2[1])

    h, w = out2.shape[:2]

    dst[dy: dy + h, dx: dx + w] = out2
    kernel_pic = dst.astype(int)

    # Calculate Sizing
    length = (bottomx - topx) * ans
    width = (bottomy - topy) * ans

    return kernel_pic, length, width

def rotate_image(img_to_rotate):
    image = img_to_rotate.copy()
    image = cv2.cvtColor(img_to_rotate,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])

    # (horizontal,vertical)
    horizontal_center = left[0]+((right[0]-left[0])/2)
    diff_top = abs(top[0]-horizontal_center)
    diff_bottom = abs(bottom[0]-horizontal_center)
    if diff_top > diff_bottom:
        img_to_rotate = cv2.rotate(img_to_rotate, cv2.ROTATE_180)
    return img_to_rotate

def save_each_class(pred_inst, inst_id, each_y_pred, img, path_save_image, request_id, color_map):

    mask_image = visualize_instances_class(
        pred_inst, inst_id, each_y_pred, img, color_map)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    try:
        os.makedirs(path_save_image)
    except:
        print("path is exist")
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    mask_image = cv2.resize(mask_image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("{}{}.jpg".format(path_save_image, request_id), mask_image)
    

def get_colors(image, number_of_colors):
    temp_inst = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2GRAY)
    temp_inst = cv2.threshold(temp_inst, 127, 255, cv2.THRESH_BINARY)[1]
    image =  cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2Lab)
    reshaped_image = image[np.where(temp_inst == 255)]
    clf = KMeans(n_clusters = number_of_colors)
    clf.fit(reshaped_image)
    labels = clf.predict(reshaped_image)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    return rgb_colors  

def get_colors_GMM(image, number_of_colors):
    reshaped_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = GaussianMixture(n_components=number_of_colors, random_state=0)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    center_colors = clf.means_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    return rgb_colors


def pred_green_paddy(img):
    # imgHSV = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2Lab)
    try:
        pred = np.mean(get_colors(img, 3), axis=0)
        return pred[1] < 133
    except:
        return False


def find_labels_pic(pred_inst, pred_type, img, size):
    images = []
    list_inst_id = []
    pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
    pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
    for idx, inst_id in enumerate(pred_id_list):
        inst_type = pred_type[pred_inst == inst_id]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]

        try:
            inst_map = np.array(pred_inst == inst_id, np.uint8)
            y1, y2, x1, x2 = bounding_box(inst_map)

            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= pred_inst.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= pred_inst.shape[0] - 1 else y2

            degree = 0

            mask_img = np.zeros_like(img)
            mask_img[np.where(inst_map == 1)] = img[np.where(inst_map == 1)]

            # Crop Image from origin picture and Rotate
            crop_img = mask_img[y1:y2, x1:x2]
            crop_thres = inst_map[y1:y2, x1:x2].astype("uint8")

            # Find Most contour area
            contours = contour_area(crop_thres)
            #             print(contours)
            # Find Degree to rotate
            degree = rotated_degree(degree, contours)
            #             print(degree)

            rotated = ndimage.rotate(crop_img, degree - 90)
            rotated_thres = ndimage.rotate(crop_thres, degree - 90)
            contours1 = contour_area(rotated_thres)

            # Mask only Grain
            topx, bottomx, topy, bottomy, out = mark_only_grain(
                rotated, contours1)
            out2 = out[topx:bottomx, topy:bottomy]
            
            if out2.shape[0] > 224:
                temp_out2 = cv2.cvtColor(out2, cv2.COLOR_RGB2GRAY)
                first_out2 = temp_out2[:int(temp_out2.shape[0]/2)]
                sec_out2 = temp_out2[int(temp_out2.shape[0]/2):]
                if first_out2[first_out2>0].shape[0] < sec_out2[sec_out2>0].shape[0]:
                    out2 = ndimage.rotate(out2, 180)
                out2 = out2[:224]
    

            img2 = np.zeros((size, size, 3))
            pt2 = [out2.shape[1] / 2, out2.shape[0] / 2]
            pt1 = [size / 2, size / 2]

            # (2) Calc offset
            dx = int(pt1[0] - pt2[0])
            dy = int(pt1[1] - pt2[1])

            h, w = out2.shape[:2]

            dst = img2.copy()
            dst[dy: dy + h, dx: dx + w] = out2
            dst = dst.astype("int64")

            images.append(dst)
            list_inst_id.append(inst_id)
        except Exception as e:
            print(e)
            print(idx)

    return images, list_inst_id

def get_params_from_path(purity_weights_path: str):
    str_split = purity_weights_path.split('/').pop()
    str_split.split('class')[0].split('_').pop()
    label_map = {
        'overall_mali_v1': ['MALI', 'NON-MALI', ''],
        'NFAVOR_ML1': ['MALI', 'NON-MALI', ''],
        'FH_ML1': ['MALI', 'NON-MALI', ''],
        'overall_PT1': ['PT1', 'NON-PT1', ''],
        'dy_gk6': ['GK6', 'NON-GK6'],
        'dy_gk79': ['GK79', 'NON-GK79']
    }
    color_map_default = [
        [54.0, 255.0, 47.0],  
        [255.0, 255, 255.0],  
        [187.0, 0.0, 0.0], 
        [187.0, 107.0, 217.0], 
        [105.0, 246.0, 255.0], 
        [0.0, 255.0, 255],  
        [194.0, 0.0, 255.0], 
        [255.0, 0.0, 0.0], 
        [0.0, 255.0, 134.62371722], 
        [255.0, 0.0, 255.0],
        [0.0, 164.0, 255.0],
        [164.0, 0.0, 164.0]
    ]
    color_map_default_2_class = [
        [54.0, 255.0, 47.0],
        [255.0, 18.0, 18.0],
        [187.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0]
    ]
    color_map = {
        'overall_mali_v1': color_map_default_2_class,
        'NFAVOR_ML1': color_map_default_2_class,
        'FH_ML1': color_map_default_2_class,
        'overall_PT1': color_map_default_2_class,
        'dy_gk6': color_map_default_2_class,
        'dy_gk79': color_map_default_2_class
    }
    model_key = str_split.split('class')[0].split(
        str_split.split('class')[0].split('_').pop())[0][7:-1]
    return int(str_split.split('class')[0].split('_').pop()), label_map[model_key], color_map[model_key]

    
