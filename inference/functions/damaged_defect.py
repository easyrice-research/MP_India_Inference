import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.misc import imresize

img_dim = 224


def hist_hsv(img):
    hist_hsv = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = ("r", "g", "b")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [64], [0, 255])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr /= np_result
        hist_hsv.extend(histr)

    return hist_hsv


def damaged(img_damaged1, model, model_ext, dict_img):
    for key in list(dict_img.keys()):
        img = imresize(dict_img[key]["image"], (img_dim, img_dim))
        feature_block = list()
        X = list()
        hsv_list = hist_hsv(img)
        feature_block.extend(hsv_list)

        X.append(feature_block)
        X_np = np.array(X)
        print(X_np.shape)

        hsv_hist_features = []
        hist_temp = [arr for list_arr in X_np[0, 0:192] for arr in list_arr]
        max_x = max(hist_temp)
        min_x = min(hist_temp)
        hist_temp = [x - min_x / max_x - min_x for x in hist_temp]
        hsv_hist_features.append(hist_temp)
        X_np = np.array(hsv_hist_features)

        pred = model.predict(X_np)
        pred_prob = model.predict_proba(X_np)
        pred_prob = pred_prob.tolist()
        print("predict probability : ", pred_prob)
        print("Predict Chalky : {}, {}".format(pred, type(pred)))
        print(pred)

        if pred == "w":
            print("Kernel {} Damaged Grain".format(key))
            dict_img[key]["damaged"] = "no"
            # cv2.rectangle(img_chalky1, dict_img[key]['loc_start'], dict_img[key]['loc_stop'], (255, 128, 255), 6)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img_chalky1, '{}'.format('Glutinous'), dict_img[key]['loc_text'], font, 2, (255, 128, 255), 5, cv2.LINE_AA)

        elif pred == "d":
            print("Kernel {} Damaged Grain".format(key))
            dict_img[key]["damaged"] = "yes"
            cv2.rectangle(
                img_damaged1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (0, 0, 255),
                5,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_damaged1,
                "{}".format("D"),
                dict_img[key]["loc_text"],
                font,
                2,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )

        else:
            print("Kernel {} Damaged Grain".format(key))
            dict_img[key]["damaged"] = "no"
