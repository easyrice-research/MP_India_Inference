import cv2
import numpy as np
from scipy.misc import imresize
import pandas as pd

img_dim = 224


def hist_rgb_selected(img):
    hist_rgb = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = "b"
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [0], None, [50], [0, 255])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr = histr / np_result
        hist_rgb.extend(histr[1])

    return hist_rgb


def hist_lab_selected(img):
    hist_lab = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color = ("b", "g")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i + 1], None, [50], [0, 255])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr = histr / np_result
        hist_lab.append(np.sort(histr.reshape(50))[-2])

    return hist_lab


def hist_hsv_selected(img):
    hist_hsv = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = "b"
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i + 1], None, [50], [0, 255])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr = histr / np_result
        hist_hsv.append(np.sort(histr.reshape(50))[-2])

    return hist_hsv


def yellow(img_yellow1, model, model_ext, dict_img):
    X = list()
    for key in list(dict_img.keys()):
        img = imresize(dict_img[key]["image"], (img_dim, img_dim))
        feature_block = list()
        rgb_list = hist_rgb_selected(img)
        feature_block.extend(rgb_list)
        lab_list = hist_lab_selected(img)
        feature_block.extend(lab_list)
        hsv_list = hist_hsv_selected(img)
        feature_block.extend(hsv_list)

        X.append(feature_block)
        X_np = np.array(X)
        print(X_np.shape)
        X_np = X_np.reshape(int(X_np.shape[0]), int(X_np.shape[1]))

    df = pd.DataFrame(X_np, columns=["R_2nd", "A_P2", "B_P2", "S_P2"])
    features = ["R_2nd", "A_P2", "B_P2", "S_P2"]
    df[features] = np.log10(df[features] + 1)
    df = (df - df.mean()) / (df.max() - df.min())

    pred = model.predict(df)
    pred_prob = model.predict_proba(df)
    pred_prob = pred_prob.tolist()
    print("predict probability : ", pred_prob)
    print("Predict Yellow : {}, {}".format(pred, type(pred)))
    print(pred)

    for idx, key in enumerate(list(dict_img.keys())):
        if pred[idx] == -1:
            print("Kernel {} Yellow Grain".format(key))
            dict_img[key]["yellow"] = "WH"
            cv2.rectangle(
                img_yellow1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (255, 255, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_yellow1,
                "{}".format("WH"),
                dict_img[key]["loc_text"],
                font,
                2,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

        # elif(pred[0] == 2):
        #     print('Kernel {} Yellow Grain'.format(key))
        #     dict_img[key]['yellow'] = 'lv2'
        #     cv2.rectangle(img_yellow, dict_img[key]['loc_start'], dict_img[key]['loc_stop'], (0,215,255), 6)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_yellow, '{}'.format('lv2'), dict_img[key]['loc_text'], font, 2, (0,215,255), 5, cv2.LINE_AA)

        elif pred[idx] == 0:
            print("Kernel {} Yellow Grain".format(key))
            dict_img[key]["yellow"] = "YW"
            cv2.rectangle(
                img_yellow1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (2, 106, 253),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_yellow1,
                "{}".format("YW"),
                dict_img[key]["loc_text"],
                font,
                2,
                (2, 106, 253),
                5,
                cv2.LINE_AA,
            )
