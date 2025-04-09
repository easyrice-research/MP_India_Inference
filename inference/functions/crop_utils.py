import cv2
import numpy as np
from scipy import ndimage
import math
import time
from skimage.transform import rescale, resize, downscale_local_mean
from inference.misc.utils import bounding_box

start_time = time.time()


def image_resizing(topx, bottomx, bottomy, topy, out, i, img_dim, pathread, pathout):
    out_crop = out[topx:bottomx, topy:bottomy]
    img_resize = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)

    resize_cropx = out_crop.shape[0]
    resize_cropy = out_crop.shape[1]
    runx = int((img_dim - (bottomx - topx)) / 2)
    runy = int((img_dim - (bottomy - topy)) / 2)

    try:
        img_resize[runx : resize_cropx + runx, runy : resize_cropy + runy, :] = out_crop
        # img_resize[runy:resize_cropx+runy, 42:97, :] = out_crop

    except:
        print("Error Picture No: ", i)
        print("Increae Value 'img_dim'")
        with open("{}/log_error.txt".format(pathout), "a+") as f:
            f.write("Problem in Pic No: {}".format(i))
            f.write("\n")
            f.close

    return img_resize


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


def contour_area(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def arrange_label(loc):
    lista = []
    listb = []
    for b in range(len(loc)):
        lista.append(loc[b])

        if len(lista) == 18:
            lista = sorted(lista, key=lambda x: x[1].start)
            listb = listb + lista
            lista = []

        if b == (len(loc) - 1):
            listb = listb + lista
            lista = []
    return listb


def connected_component(thres):
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        thres, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 300

    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    img_zero = np.zeros((labels.shape))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_zero[labels == i + 1] = 255

    lbl, nlbl = ndimage.label(img_zero, s)
    return lbl, nlbl, img_zero


def image_processing(img, i=0):
    img_blur = cv2.medianBlur(img, 13)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    _, thres_red = cv2.threshold(
        img_lab[:, :, 1], 139, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, thres_gray = cv2.threshold(
        img_gray, 139, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thres = np.logical_or(thres_red, thres_gray)
    thres = thres.astype("uint8")
    # cv2.imwrite('/{}/akernel_test_threshold{}.jpg'.format('output/test_output_yellow5', i), thres_gray)

    return thres, thres_gray


def rotated_degree(degree, contours):
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
        degree = theta[indmin90]
    else:
        degree = theta[indminarea]

    return degree


def crop_full(pathread, pathout, filename, dpi, img_dim):
    # Read Image
    img = cv2.imread("/{}/{}".format(pathread, filename))
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (4960, 7015), interpolation=cv2.INTER_AREA)

    thres, thres_gray = image_processing(img)
    # thres[thres == 1] = 255

    # Intial_Values
    inch = img.shape[1] / dpi
    mm = inch * 25.4
    result = mm / img.shape[1]

    # Connected Component
    lbl, nlbl, img_zero = connected_component(thres)

    xx = []
    degree = np.zeros(nlbl)
    img_copy = img.copy()
    dict_img = dict()

    for i in range(nlbl):
        print(str(i) + "/" + str(nlbl))
        loc = ndimage.find_objects(lbl)
        listb = arrange_label(loc)
        # Arrange Label
        loc = listb[i]

        # Crop Image from origin picture and Rotate
        crop_img = img[loc[0].start : loc[0].stop, loc[1].start : loc[1].stop]
        crop_thres = thres[loc[0].start : loc[0].stop, loc[1].start : loc[1].stop]

        # Find Most contour area
        contours = contour_area(crop_thres)
        if contours <= 10:
            continue

        # Find Degree to rotate
        degree = rotated_degree(degree, contours, i)
        print(degree)

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
        img_resize = image_resizing(
            topx, bottomx, bottomy, topy, out, i, img_dim, pathread, pathout
        )

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

    print("--- %s seconds ---" % (time.time() - start_time))
    return dict_img, img, img_copy


def crop_half(pathread, pathout, filename, dpi, img_dim, idx, no_pic, aspect):
    img = cv2.imread("/{}/{}".format(pathread, filename))
    img = resize(img, (7015, 4960), anti_aliasing=True)

    dict_img = dict()
    inch = img.shape[1] / dpi
    mm = inch * 25.4
    result = mm / img.shape[1]

    thres, thres_gray = image_processing(img)

    degree = np.zeros(1)
    # Find Most contour area
    contour = contour_area(thres_gray)

    # Find Degree to rotate
    degree = rotated_degree(degree, contour, 0)
    rotated = ndimage.rotate(img, degree[0] - 90)

    thres_rotated, thres_rotated_gray = image_processing(rotated)
    contours1 = contour_area(thres_rotated_gray)
    topx, bottomx, topy, bottomy, out = mark_only_grain(rotated, contours1)

    length = (bottomx - topx) * result
    width = (bottomy - topy) * result
    print(length)

    img_resize = image_resizing(
        topx, bottomx, bottomy, topy, out, idx, img_dim, pathread, pathout
    )
    cv2.imwrite("/{}/kernel_{}".format(pathout, filename), img_resize)

    dict_idx = "{}_{}".format(idx, aspect)
    dict_img[dict_idx] = {"image": img_resize}
    dict_img[dict_idx]["size"] = length
    dict_img[dict_idx]["aspect"] = aspect

    return dict_img, dict_idx, img_resize


def visualize_instances_class(mask, inst_ids, pred_type, canvas=None, color_map=[], color=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    canvas = (
        np.full(mask.shape + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = inst_ids

    for idx, inst_id in enumerate(insts_list):
        if int(pred_type[idx]) != 99:
            inst_map = np.array(mask == inst_id, np.uint8)
            y1, y2, x1, x2 = bounding_box(inst_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]
            inst_canvas_crop = canvas[y1:y2, x1:x2]
            inst_color = np.array(color_map[int(pred_type[idx])])
            contours, hierarchy = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(inst_canvas_crop, contours, -1, inst_color, 4)
            canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas



def visualize_instances_shape(
    mask, inst_ids, pred_shape, kernel_poses, canvas=None, color=None
):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    canvas = (
        np.full(mask.shape + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = inst_ids

    # ['White','Red','Paddy','Yellow','Glutinous','Chalky','Damaged','Undermilled','Whole_shape','Broken_Head_shape']
    color_map = [
        [2.0, 192.0, 47.0],  # green Whole Grain
        [255.0, 0.0, 0.0],  # red Red
        [0.0, 0.0, 0.0],
    ]

    for idx, inst_id in enumerate(insts_list):
        y1, y2, x1, x2 = kernel_poses[idx]
        if int(pred_shape[idx]) == 1:
            inst_map = np.array(mask[y1:y2, x1:x2] == inst_id, np.uint8)
            inst_map_crop = inst_map
            inst_canvas_crop = canvas[y1:y2, x1:x2]
            inst_color = np.array(color_map[int(pred_shape[idx]) - 1])
            contours, hierarchy = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(inst_canvas_crop, contours, -1, inst_color, 2)
            canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas


def visualize_instances_class_length(mask, pred_type, canvas=None, length_list=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    canvas = (
        np.full(mask.shape + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = list(np.unique(mask))
    insts_list.remove(0)  # remove background

    color_map = [
        [255.0, 0.0, 40.12542759],  # red 1.White'
        [86.06613455, 0.0, 255.0],  # navy '2.Red'
        [255.0, 148.28962372, 0.0],  # orange '3.Paddy'
        [0.0, 255.0, 134.62371722],  # light green '4.Yellow'
        [0.0, 255.0, 255],  # light blue '5.Glutinous'
        [255.0, 255.0, 225.0],  # white '6.Chalky'
        [255.0, 0.0, 255.0],  # pink '7.Damaged'
        [255.0, 255.0, 0.0],  # yellow '8.Undermilled'
        [143.0, 0.0, 255.0],  # purple
        [0.0, 164.0, 255.0],  # blue
        [255.0, 170.0, 255.0],
    ]  #

    for idx, inst_id in enumerate(insts_list):
        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        # edit
        mask_temp = np.zeros_like(mask)
        mask_temp[np.where(mask == inst_id)] = pred_type[np.where(mask == inst_id)]
        inst_pred_type = mask_temp[y1:y2, x1:x2]
        inst_pred_type = inst_pred_type[inst_pred_type > 0]
        counts = np.bincount(inst_pred_type)

        if len(counts) > 0:
            inst_color = np.array(color_map[np.argmax(counts) - 1])
            contours, hierarchy = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(inst_canvas_crop, contours, -1, inst_color, 2)
            cv2.putText(
                canvas,
                str(length_list[idx])[:5],
                (x1 - 30, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                [255, 255, 255],
            )
            canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas
