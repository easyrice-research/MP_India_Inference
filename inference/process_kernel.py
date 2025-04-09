from numpy.core.fromnumeric import shape
import cv2
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
from inference.misc.utils import bounding_box
from inference.functions import *
from inference.functions.crop_utils import *
from scipy import ndimage

from inference.Integrated import submodel
import os
from multiprocessing import get_context
from functools import partial
import time


def weight_feature(
    model_type, l_channel, pred_inst, kernel_pos, kernel, kernel_type, shape_type
):
    inst_id, _, _, pixel_count, _ = kernel
    y1, y2, x1, x2 = kernel_pos
    l_channel = l_channel[y1:y2, x1:x2]
    lab = l_channel[np.where(pred_inst[y1:y2, x1:x2] == inst_id)]
    b = [i for i in range(0, 280, 30)]
    hist, _ = np.histogram(lab, bins=b)
    feature = hist / pixel_count

    w_bases = [
        9.888136660434709e-06,
        9.218318742716537e-06,
        8.151976981269914e-06,
        9.943307462734612e-06,
        9.435210959200047e-06,
    ]

    # ข้าวเต้ม ขาวหัก ข้าวเปลือก แดง glu

    # 1.White','2.Red','3.Paddy','4.Yellow','5.Glutinous','6.Chalky','7.Damaged','8.Undermilled'
    if model_type == "QA":
        try:
            if shape_type == 2:
                w_base = w_bases[1]
            elif kernel_type in [2]:
                w_base = w_bases[3]
            elif kernel_type in [7]:
                w_base = w_bases[2]
            elif kernel_type in [5]:
                w_base = w_bases[4]
            else:
                w_base = w_bases[0]
        except:
            print((kernel_type, shape_type))

    elif model_type == "QC":
        if kernel_type in [4]:
            w_base = 0
        else:
            w_base = w_bases[kernel_type - 1]
    feature = np.concatenate([feature, [pixel_count * w_base]])

    return feature


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

    size = 224
    dst = np.zeros((size, size, 3))
    pt2 = [out2.shape[1] / 2, out2.shape[0] / 2]
    pt1 = [size / 2, size / 2]

    ## (2) Calc offset
    dx = int(pt1[0] - pt2[0])
    dy = int(pt1[1] - pt2[1])

    h, w = out2.shape[:2]

    dst[dy : dy + h, dx : dx + w] = out2
    kernel_pic = dst.astype(int)

    # Calculate Sizing
    length = (bottomx - topx) * ans
    width = (bottomy - topy) * ans

    return kernel_pic, length, width


def calculate_percentyield(result=None, percent_yield_type="pixel" or "kernel"):

    yield_paddy = (
        result["WholeGrain_" + percent_yield_type]
        + result["RedWhole_" + percent_yield_type]
    ) / (
        result["WholeGrain_" + percent_yield_type]
        + result["RedWhole_" + percent_yield_type]
        + result["Broken_Red_" + percent_yield_type]
        + result["Broken_" + percent_yield_type]
        + result["Head_WholeGrain_" + percent_yield_type]
    )
    yield_Broken = result["Broken_" + percent_yield_type] / (
        result["WholeGrain_" + percent_yield_type]
        + result["Broken_" + percent_yield_type]
    )

    return {"yield_paddy": yield_paddy, "yield_Broken": yield_Broken}


def preprocess(pred_inst, pred_type, img, ans, args):
    inst_id, pixel_count, pred_inst_centroid = args
    inst_map = np.array(pred_inst == inst_id, np.uint8)
    y1, y2, x1, x2 = bounding_box(inst_map)

    y1 = y1 - 2 if y1 - 2 >= 0 else y1
    x1 = x1 - 2 if x1 - 2 >= 0 else x1
    x2 = x2 + 2 if x2 + 2 <= pred_inst.shape[1] - 1 else x2
    y2 = y2 + 2 if y2 + 2 <= pred_inst.shape[0] - 1 else y2

    # edit
    inst_pred_type = pred_type[y1:y2, x1:x2]
    inst_pred_type = inst_pred_type[inst_pred_type > 0]
    counts_type = np.bincount(inst_pred_type)
    kernel_type = 0
    try:
        kernel_type = np.argmax(counts_type)
    except:
        # kernel pic too big
        return
    try:
        kernel_pic, length, width = kernel_length_width(
            img=img, inst_map=inst_map, pos=[y1, y2, x1, x2], ans=ans
        )
    except:
        # kernel pic too big
        return

    return (
        kernel_pic,
        kernel_type,
        inst_id,
        length,
        width,
        pixel_count,
        pred_inst_centroid,
        [y1, y2, x1, x2],
    )


def process_kernel(
    pred_inst=None,
    pred_type=None,
    pred_inst_centroid=None,
    img=None,
    model_type=None,
    request_id=None,
):
    start_time = time.time()
    start_timex = time.time()
    pred_inst_centroids = pred_inst_centroid

    result = dict()
    each_kernel = []
    if model_type == "QA":
        each_kernel_count = np.zeros((10, 2))
    else:
        each_kernel_count = np.zeros((5, 2))
    _, counts_pixel = np.unique(pred_inst, return_counts=True)
    counts_pixel = np.delete(counts_pixel, 0)
    insts_list = list(np.unique(pred_inst))
    insts_list.remove(0)

    c_name = [
        "White",
        "Red",
        "Paddy",
        "Yellow",
        "Glutinous",
        "Chalky",
        "Damaged",
        "Undermilled",
        "Whole_grain",
        "Broken_shape",
    ]

    if len(insts_list) > 0:

        # load model
        if model_type == "QA":
            if img.shape[1] == 1240:
                model_size = "subsequence_quar/"
                dpi = 150
            else:
                model_size = "subsequence_half/"
                dpi = 290

        inch = pred_inst.shape[1] / dpi
        mm = inch * 25.4
        ans = mm / pred_inst.shape[1]

        start = time.time()
        func = partial(preprocess, pred_inst, pred_type, img, ans)
        kernel_attrs = []
        kernel_pics = []
        kernel_types = []
        kernel_poses = []
        cpu_count = os.cpu_count() // 2 or 1
        with get_context("spawn").Pool(cpu_count) as p:
            start_loop_time = time.time()
            for preprocessed in p.map(
                func, zip(insts_list, counts_pixel, pred_inst_centroids)
            ):
                if preprocessed is None:
                    continue
                (
                    kernel_pic,
                    _,
                    inst_id,
                    length,
                    width,
                    pixel_count,
                    pred_inst_centroid,
                    kernel_pos,
                ) = preprocessed
                kernel_pics.append(kernel_pic)
                kernel_poses.append(kernel_pos)
                # kernel_types.append(kernel_type)
                kernel_attrs.append(
                    (inst_id, length, width, pixel_count, pred_inst_centroid)
                )
            print("loop end :  --- %s seconds ---" % (time.time() - start_loop_time))

        del pred_inst_centroids
        del counts_pixel

        print("preprocess end :  --- %s seconds ---" % (time.time() - start_time))

        lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, _, _ = cv2.split(lab_image)
        weight_func = partial(
            weight_feature, model_type, l_channel, pred_inst, kernel_pos
        )

        if model_type == "QA":
            # 1.White','2.Red','3.Paddy','4.Yellow','5.Glutinous','6.Chalky','7.Damaged','8.Undermilled'

            start = time.time()
            batch_size = 1024
            defect_shape_model = submodel.defect_shape_model(model_size)
            kernel_types = np.array([])
            shape_preds = np.array([])
            for i in range(0, len(kernel_pics), batch_size):
                batch_kernel_types, batch_shape_preds = defect_shape_model.predict(
                    np.array(kernel_pics[i : i + batch_size])
                )
                kernel_types = np.concatenate(
                    [kernel_types, np.argmax(batch_kernel_types, axis=1) + 1]
                )
                shape_preds = np.concatenate(
                    [shape_preds, np.argmax(batch_shape_preds, axis=1) + 1]
                )
                del batch_kernel_types
                del batch_shape_preds

            del kernel_pics
            print("Classification Time", time.time() - start)

            # kernel weight
            start_time = time.time()
            with get_context("spawn").Pool(cpu_count) as p:
                features = [
                    feature
                    for feature in p.starmap(
                        weight_func, zip(kernel_attrs, kernel_types, shape_preds)
                    )
                    if feature is not None
                ]
            print("pred end :  --- %s seconds ---" % (time.time() - start_time))

            start = time.time()
            weight_model = submodel.weight_model(model_size)
            weights = np.array([])
            for i in range(0, len(features), batch_size):
                weights = np.concatenate(
                    [
                        weights,
                        weight_model.predict(np.array(features[i : i + batch_size])),
                    ]
                )

            del weight_model
            del features
            print("Weight predict Time", time.time() - start)

            start = time.time()
            for kernel, kernel_type, shape_pred, weight in zip(
                kernel_attrs, kernel_types, shape_preds, weights
            ):
                inst_id, length, width, pixel_count, pred_inst_centroid = kernel
                each_kernel.append(
                    dict(
                        zip(
                            [
                                "Type",
                                "Shape",
                                "Pixel",
                                "length",
                                "width",
                                "ratio",
                                "centroid",
                                "weight",
                            ],
                            [
                                c_name[int(kernel_type) - 1],
                                c_name[int(shape_pred) - 1 + 8],
                                pixel_count,
                                length,
                                width,
                                length / width,
                                pred_inst_centroid,
                                weight,
                            ],
                        )
                    )
                )
                each_kernel_count[int(kernel_type) - 1][0] += pixel_count  # pixel type
                each_kernel_count[int(shape_pred) - 1 + 8][
                    0
                ] += pixel_count  # pixel shape

                each_kernel_count[int(kernel_type) - 1][1] += 1  # kernel type
                each_kernel_count[int(shape_pred) - 1 + 8][1] += 1  # kernel shape
            print("Labeled Time", time.time() - start)

    start = time.time()
    # Export visualized image
    # Defect
    filtered_inst_list = [attr[0] for attr in kernel_attrs]
    overlaid_class_output = visualize_instances_class(
        pred_inst, filtered_inst_list, kernel_types, kernel_poses, img
    )
    overlaid_class_output = cv2.cvtColor(overlaid_class_output, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('/tmp/{}.png'.format(request_id), overlaid_class_output)
    cv2.imwrite("/tmp/{}.png".format(request_id), overlaid_class_output)
    # Shape
    overlaid_shape_output = visualize_instances_shape(
        pred_inst, filtered_inst_list, shape_preds, kernel_poses, img
    )
    overlaid_shape_output = cv2.cvtColor(overlaid_shape_output, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('/tmp/{}_shape.png'.format(request_id), overlaid_shape_output)

    # plt.imshow(overlaid_class_output)
    # plt.savefig('/tmp/{}.png'.format(request_id))

    # Replace class number with name
    if model_type == "QA":
        classes = [name + "_kernel" for name in c_name]
    elif model_type == "QC":
        classes = [
            "WholeGrain_kernel",
            "Broken_kernel",
            "Paddy_kernel",
            "Husk_kernel",
            "RedWhole_kernel",
        ]
    print(dict(zip(classes, each_kernel_count[:, 1])))
    result.update(dict(zip(classes, each_kernel_count[:, 1])))

    if model_type == "QA":
        classes = [name + "_pixel" for name in c_name]
    elif model_type == "QC":
        classes = [
            "WholeGrain_pixel",
            "Broken_pixel",
            "Paddy_pixel",
            "Husk_pixel",
            "RedWhole_pixel",
        ]

    result.update(dict(zip(classes, each_kernel_count[:, 0])))
    result.update(dict(zip(["Inferenced_img"], img)))
    result.update(dict(zip(["Est"], [each_kernel])))
    print("proc end :  --- %s seconds ---" % (time.time() - start_timex))

    return result


if __name__ == "__main__":
    m_type = "QA"
    pred_inst = np.load("temp/pred_inst_13443903045787.npy")
    pred_type = np.load("temp/pred_type_13443903045787.npy")
    pred_inst_centroid = np.load("temp/pred_inst_centroid_13443903045787.npy")
    img = np.load("temp/test_13443903045787.npy")

    result = process_kernel(
        pred_inst=pred_inst,
        pred_type=pred_type,
        pred_inst_centroid=pred_inst_centroid,
        img=img,
        model_type=m_type,
    )


#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img', help="image path")
#     parser.add_argument('--model_type', help="QA QC")
#     parser.add_argument('--hover_model', help="checkpoint path") #ex ..logs/qa14/model-196992.index
#     parser.add_argument('--size', help="image size ex. 1240,1754  2480,3508")

#     args = parser.parse_args()
#     result = inference_hover(args.img,args.hover_model,args.model_type,args.size)

#     pickle.dump(result,open('../results/'+args.img.split('/')[-1][:-4]+'_pickle_point2.pickle', 'wb'))

#     percent_yield_type = 'pixel'
#     percent_yield = calculate_percentyield(result,percent_yield_type)

# cmd run

# docker run --rm -v /home/easyrice/Documents/tf-gpu:/notebooks nuttheguitar/tf-gpu-realtime-od:update-10.02.20 python ../notebooks/Research/M100/Hover-net-git/src/test.py

# export INEGRATED_MODEL_PATH='Integrated/Models/'
