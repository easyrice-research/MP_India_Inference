import os
import matplotlib.pyplot as plt
import colorsys
import random
from inference.misc.viz_utils import visualize_instances

## config.py Libraries
import importlib

## infer.py Libraries
import tensorflow as tf
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader
import cv2
import numpy as np
import math
from collections import deque
import time
import pandas as pd
import seaborn as sns
from joblib import load, dump

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from skimage.morphology import remove_small_objects, watershed
from scipy import ndimage

model_type = "np_hv"


def __gen_prediction(x, predictor):
    """
    Using 'predictor' to generate the prediction of image 'x'

    Args:
        x : input image to be segmented. It will be split into patches
            to run the prediction upon before being assembled back
    """
    step_size = [80, 80]
    msk_size = [80, 80]
    win_size = [270, 270]
    inf_batch_size = 16

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = x.shape[0]
    im_w = x.shape[1]

    last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
    last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

    diff_h = win_size[0] - step_size[0]
    padt = diff_h // 2
    padb = last_h + win_size[0] - im_h

    diff_w = win_size[1] - step_size[1]
    padl = diff_w // 2
    padr = last_w + win_size[1] - im_w

    x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    #### TODO: optimize this
    sub_patches = []
    # generating subpatches from orginal
    for row in range(0, last_h, step_size[0]):
        for col in range(0, last_w, step_size[1]):
            win = x[row : row + win_size[0], col : col + win_size[1]]
            sub_patches.append(win)

    pred_map = deque()
    while len(sub_patches) > inf_batch_size:
        mini_batch = sub_patches[:inf_batch_size]
        sub_patches = sub_patches[inf_batch_size:]
        mini_output = predictor(mini_batch)[0]
        mini_output = np.split(mini_output, inf_batch_size, axis=0)
        pred_map.extend(mini_output)
    if len(sub_patches) != 0:
        mini_output = predictor(sub_patches)[0]
        mini_output = np.split(mini_output, len(sub_patches), axis=0)
        pred_map.extend(mini_output)

    #### Assemble back into full image
    output_patch_shape = np.squeeze(pred_map[0]).shape
    ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

    #### Assemble back into full image
    pred_map = np.squeeze(np.array(pred_map))
    pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
    pred_map = (
        np.transpose(pred_map, [0, 2, 1, 3, 4])
        if ch != 1
        else np.transpose(pred_map, [0, 2, 1, 3])
    )
    pred_map = np.reshape(
        pred_map,
        (
            pred_map.shape[0] * pred_map.shape[1],
            pred_map.shape[2] * pred_map.shape[3],
            ch,
        ),
    )
    pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

    return pred_map


def proc_np_hv(pred, marker_mode=2, energy_mode=2, rgb=None):
    """
    Process Nuclei Prediction with XY Coordinate Map
    Args:
        pred: prediction output, assuming
                channel 0 contain probability map of nuclei
                channel 1 containing the regressed X-map
                channel 2 containing the regressed Y-map
    """
    assert marker_mode == 2 or marker_mode == 1, "Only support 1 or 2"
    assert energy_mode == 2 or energy_mode == 1, "Only support 1 or 2"

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    ##### Processing
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1

    blb[blb < 0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # back ground is 0 already
    #####

    if energy_mode == 2 or marker_mode == 2:
        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        if energy_mode == 2:
            dist = (1.0 - overall) * blb
            ## nuclei values form mountains so inverse to get basins
            dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        if marker_mode == 2:

            overall[overall >= 0.3] = 1
            overall[overall < 0.3] = 0

            marker = blb - overall
            marker[marker < 0] = 0
            marker = binary_fill_holes(marker).astype("uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            marker = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel)
            marker = measurements.label(marker)[0]
            marker = remove_small_objects(marker, min_size=10)

    if energy_mode == 1:
        dist = h_dir_raw * h_dir_raw + v_dir_raw * v_dir_raw
        dist[blb == 0] = np.amax(dist)
        # nuclei values are already basins
        dist = filters.maximum_filter(dist, 7)
        dist = cv2.GaussianBlur(dist, (3, 3), 0)

    if marker_mode == 1:
        h_marker = np.copy(h_dir_raw)
        v_marker = np.copy(v_dir_raw)
        h_marker = np.logical_and(h_marker < 0.075, h_marker > -0.075)
        v_marker = np.logical_and(v_marker < 0.075, v_marker > -0.075)
        marker = np.logical_and(h_marker > 0, v_marker > 0) * blb
        marker = binary_dilation(marker, iterations=2)

        marker = binary_fill_holes(marker)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, marker, mask=blb)

    return proced_pred, dist, marker, overall


def get_model():
    if model_type == "np_hv":
        model_constructor = importlib.import_module("inference.model.graph")
        model_constructor = model_constructor.Model_NP_HV
    elif model_type == "np_dist":
        model_constructor = importlib.import_module("inference.model.graph")
        model_constructor = model_constructor.Model_NP_DIST

    return model_constructor


def contour_area(img):
    print(img.shape)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def image_resizing(topx, bottomx, bottomy, topy, out, i, img_dim):
    try:
        out_crop = out[topx:bottomx, topy:bottomy]
        img_resize = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)

        width = int(out_crop.shape[1] * 2)
        height = int(out_crop.shape[0] * 2)
        dim = (width, height)
        out_crop = cv2.resize(out_crop, dim, interpolation=cv2.INTER_AREA)

        resize_cropx = out_crop.shape[0]
        resize_cropy = out_crop.shape[1]
        runx = int((img_dim - ((bottomx - topx) * 2)) / 2)
        runy = int((img_dim - ((bottomy - topy) * 2)) / 2)

        img_resize[runx : resize_cropx + runx, runy : resize_cropy + runy, :] = out_crop
        # img_resize[runy:resize_cropx+runy, 42:97, :] = out_crop

    except:
        img_dim = 350
        out_crop = out[topx:bottomx, topy:bottomy]
        img_resize = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)

        width = int(out_crop.shape[1] * 2)
        height = int(out_crop.shape[0] * 2)
        dim = (width, height)
        out_crop = cv2.resize(out_crop, dim, interpolation=cv2.INTER_AREA)

        resize_cropx = out_crop.shape[0]
        resize_cropy = out_crop.shape[1]
        runx = int((img_dim - ((bottomx - topx) * 2)) / 2)
        runy = int((img_dim - ((bottomy - topy) * 2)) / 2)

        img_resize[runx : resize_cropx + runx, runy : resize_cropy + runy, :] = out_crop
        img_resize = cv2.resize(img_resize, (224, 224), interpolation=cv2.INTER_AREA)

    return img_resize


def rotated_degree(degree, contours, i):
    # Use Convexhull to find longest distance width and calculate with Mathematics
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    x = hull[:, 0, 0]
    y = hull[:, 0, 1]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x = x - mean_x
    y = y - mean_y
    nTheta = 180
    fd = np.zeros(nTheta)
    theta = np.arange(0, 180, 1)
    spacing = np.array([1, 1])

    for t in range(nTheta):
        theta2 = (-theta[t] * math.pi) / 180
        x2 = (x * math.cos(theta2)) - (y * math.sin(theta2))
        x2_min = np.min(x2)
        x2_max = np.max(x2)

        dl = spacing[0] * (abs(math.cos(theta2) + spacing[1]) * abs(math.sin(theta2)))
        fd[t] = (x2_max - x2_min) + dl

    index_theta = int(theta.shape[0] / 2)
    feret_area = fd[:index_theta] * fd[index_theta:]
    indminarea = np.argmin(feret_area)
    indmin90 = indminarea + 90

    if fd[indminarea] < fd[indmin90]:
        degree[i] = theta[indmin90]
    else:
        degree[i] = theta[indminarea]

    return degree


def mark_only_grain(rotated_img, contour):
    mask = np.zeros_like(
        rotated_img
    )  # Create mask where white is what we want, black otherwise
    cv2.drawContours(
        mask, contour, 0, (255, 255, 255), -1
    )  # Draw filled contour in mask
    out = np.zeros_like(
        rotated_img
    )  # Extract out the object and place into output image
    out[mask == 255] = rotated_img[mask == 255]

    (x, y, _) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    return topx, bottomx, topy, bottomy, out


def crop_full_hover(pathread, pathout, filename, dpi, img_dim):
    ## Initial ##
    model_path = "./success_models/270x270_3508_RED/00/model-601524.index"
    eval_inf_input_tensor_names = ["images"]
    eval_inf_output_tensor_names = ["predmap-coded"]

    model_constructor = get_model()
    pred_config = PredictConfig(
        model=model_constructor(),
        session_init=get_model_loader(model_path),
        input_names=eval_inf_input_tensor_names,
        output_names=eval_inf_output_tensor_names,
    )

    predictor = OfflinePredictor(pred_config)

    # Read Image
    img = cv2.imread("/{}/{}".format(pathread, filename))
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (4960, 7015), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred_map = __gen_prediction(img, predictor)

    marker_mode = 2
    energy_mode = 2

    pred = np.squeeze(pred_map)
    pred_inst = pred[..., 5:]
    pred_type = pred[..., :5]

    pred_inst = np.squeeze(pred_inst)
    pred_type = np.argmax(pred_type, axis=-1)
    pred_inst = pred
    b = pred.copy()

    pred_inst, dist, marker, overall = proc_np_hv(
        pred_inst, marker_mode=marker_mode, energy_mode=energy_mode, rgb=img
    )

    list_contour = []

    # Intial_Values
    inch = pred_inst.shape[1] / dpi
    mm = inch * 25.4
    result = mm / pred_inst.shape[1]

    img_copy = img.copy()
    dict_img = dict()

    lbl = pred_inst
    all_loc = ndimage.find_objects(lbl)
    degree = np.zeros(len(list(all_loc)))

    for idx, i in enumerate(range(len(list(all_loc)))):
        try:
            # Arrange Label
            loc = all_loc[i]
            print(loc)
            mask_img = np.zeros_like(img)
            mask_thres = np.zeros_like(lbl)
            mask_img[np.where(lbl == (i + 1))] = img[np.where(lbl == (i + 1))]
            mask_thres[np.where(lbl == (i + 1))] = 255
            mask_thres = cv2.dilate(
                mask_thres.astype("uint8"),
                kernel=np.ones((3, 3), np.uint8),
                iterations=4,
            )

            lbl[np.where(mask_thres == 255)] = i + 1
            mask_img[np.where(lbl == (i + 1))] = img[np.where(lbl == (i + 1))]
            mask_thres[np.where(lbl == (i + 1))] = 255

            # Crop Image from origin picture and Rotate
            crop_img = mask_img[loc[0].start : loc[0].stop, loc[1].start : loc[1].stop]
            crop_filter = crop_img[:, :, 0] > 80
            crop_filter = cv2.medianBlur(crop_filter.astype("uint8"), 5)
            crop_img_filter = np.zeros_like(crop_img)
            crop_img_filter[np.where(crop_filter)] = crop_img[np.where(crop_filter)]

            crop_img_gray = cv2.cvtColor(crop_img_filter, cv2.COLOR_BGR2GRAY)
            crop_img_lab = cv2.cvtColor(crop_img_filter, cv2.COLOR_BGR2LAB)
            _, crop_red = cv2.threshold(
                crop_img_lab[:, :, 1], 139, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, crop_thres = cv2.threshold(
                crop_img_gray, 139, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            crop_thres = np.logical_or(crop_red, crop_thres)
            crop_thres = crop_thres.astype("uint8")

            crop_img = crop_img_filter

            # Find Most contour area
            contours = contour_area(crop_thres)
            list_contour.append(contours)

            # Find Degree to rotate
            degree = rotated_degree(degree, contours, i)

            rotated = ndimage.rotate(crop_img, degree[i] - 90)
            rotated_thres = ndimage.rotate(crop_thres, degree[i] - 90)
            contours1 = contour_area(rotated_thres)

            # Mask only Grain
            topx, bottomx, topy, bottomy, out = mark_only_grain(rotated, contours1)
            out2 = out[topx:bottomx, topy:bottomy]

            # Calculate Sizing
            length = (bottomx - topx) * result
            width = (bottomy - topy) * result

            # Image Resizing
            img_resize = image_resizing(topx, bottomx, bottomy, topy, out, i, img_dim)

            # Write Text
            cv2.rectangle(
                img_copy,
                (loc[1].start - 5, loc[0].start - 5),
                (loc[1].stop + 5, loc[0].stop + 7),
                (0, 0, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_copy,
                str(i),
                (loc[1].start, loc[0].start - 20),
                font,
                2,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )

            dict_img[i] = {"image": img_resize}
            dict_img[i]["image_crop"] = out2
            dict_img[i]["loc_start"] = (loc[1].start - 5, loc[0].start - 5)
            dict_img[i]["loc_stop"] = (loc[1].stop + 5, loc[0].stop + 7)
            dict_img[i]["loc_text"] = (loc[1].start, loc[0].start - 20)
            dict_img[i]["size"] = length

        except Exception as e:
            print("i is :", i)
            print(e)

    return dict_img, img, img_copy
