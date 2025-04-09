import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.misc import imresize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import Model

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


def flatten(l):
    print(np.shape(l))
    if len(np.shape(l)) > 2:
        return np.reshape(l, (len(l), l.shape[1] * l.shape[2] * l.shape[3]))
    else:
        return l


def mix(img, model, model_yd_deep, model_ext, dict_img):
    img_chalky = img.copy()
    img_damaged = img.copy()
    img_glutinous = img.copy()
    img_yellow = img.copy()

    for key in list(dict_img.keys()):

        dict_img[key]["chalky"] = "null"
        dict_img[key]["damaged"] = "null"
        dict_img[key]["glutinous"] = "null"
        dict_img[key]["yellow"] = "null"

        if dict_img[key]["paddy"] == "w":
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

            img = dict_img[key]["image"]
            extract_model = Model(
                inputs=model_ext.input,
                outputs=model_ext.get_layer("{}".format("fc2")).output,
            )
            data = np.reshape(img, (1, img_dim, img_dim, 3))
            features = flatten(extract_model.predict(data))
            pred_yd_deep = model_yd_deep.predict(features)
            pred_prob_yd_deep = model_yd_deep.predict_proba(features)
            pred_prob_yd_deep = pred_prob_yd_deep.tolist()
            print("predict probability : ", pred_yd_deep)

            print("predict probability : ", pred_prob)
            print("Predict Mix : {}, {}".format(pred, type(pred)))
            print(pred)

            ######################### Predict 5 Classes #########################
            if pred[0] == "w":
                print("Kernel {} White Grain".format(key))
                dict_img[key]["chalky"] = "null"
                dict_img[key]["damaged"] = "null"
                dict_img[key]["glutinous"] = "null"
                dict_img[key]["yellow"] = "null"

            else:
                if pred[0] == "c":
                    print("Kernel {} Chalky Grain".format(key))
                    dict_img[key]["chalky"] = "c"
                    cv2.rectangle(
                        img_chalky,
                        dict_img[key]["loc_start"],
                        dict_img[key]["loc_stop"],
                        (255, 255, 255),
                        5,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img_chalky,
                        "{}".format("C"),
                        dict_img[key]["loc_text"],
                        font,
                        1,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA,
                    )

                elif pred[0] == "d":
                    print("Kernel {} Damaged Grain".format(key))
                    dict_img[key]["yellow"] = "y"
                    dict_img[key]["damaged"] = "d"
                    cv2.rectangle(
                        img_yellow,
                        dict_img[key]["loc_start"],
                        dict_img[key]["loc_stop"],
                        (255, 255, 255),
                        5,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img_yellow,
                        "{}".format("Y"),
                        dict_img[key]["loc_text"],
                        font,
                        1,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.rectangle(
                        img_damaged,
                        dict_img[key]["loc_start"],
                        dict_img[key]["loc_stop"],
                        (255, 255, 255),
                        5,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img_damaged,
                        "{}".format("D"),
                        dict_img[key]["loc_text"],
                        font,
                        1,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA,
                    )

                elif pred[0] == "g":
                    print("Kernel {} Glutinous Grain".format(key))
                    dict_img[key]["glutinous"] = "g"
                    cv2.rectangle(
                        img_glutinous,
                        dict_img[key]["loc_start"],
                        dict_img[key]["loc_stop"],
                        (255, 255, 255),
                        5,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img_glutinous,
                        "{}".format("G"),
                        dict_img[key]["loc_text"],
                        font,
                        1,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA,
                    )

                elif pred[0] == "y":
                    print("Kernel {} Yellow Grain".format(key))
                    dict_img[key]["yellow"] = "y"
                    cv2.rectangle(
                        img_yellow,
                        dict_img[key]["loc_start"],
                        dict_img[key]["loc_stop"],
                        (255, 255, 255),
                        5,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img_yellow,
                        "{}".format("Y"),
                        dict_img[key]["loc_text"],
                        font,
                        1,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA,
                    )

                else:
                    print("Kernel {} White Grain".format(key))
                    dict_img[key]["chalky"] = "null"
                    dict_img[key]["damaged"] = "null"
                    dict_img[key]["glutinous"] = "null"
                    dict_img[key]["yellow"] = "null"

            ######################### Predict 2 Classes - Damaged Yellow #########################
            print(dict_img[0].keys())
            # if dict_img[key]['yellow'] == 'y':
            #     if(pred_yd_deep == [1]):
            #         print('Kernel {} Damaged Grain'.format(key))
            #         dict_img[key]['damaged'] = 'd'
            #         cv2.rectangle(img_damaged, dict_img[key]['loc_start'], dict_img[key]['loc_stop'], (255, 255, 255), 5)
            #         font = cv2.FONT_HERSHEY_SIMPLEX
            #         cv2.putText(img_damaged, '{}'.format('D'), dict_img[key]['loc_text'], font, 1, (255, 255, 255), 5, cv2.LINE_AA)
            #     else:
            #         print('Kernel {} Damaged Grain'.format(key))
            #         dict_img[key]['damaged'] = 'null'

            # else:
            #     continue

        else:
            continue

    return img_chalky, img_damaged, img_glutinous, img_yellow, dict_img
