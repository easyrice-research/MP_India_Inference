import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.misc import imresize

img_dim = 224


def hist_rgb(img):
    hist_rgb = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = ("r", "g", "b")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [64], [0, 255])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr /= np_result
        hist_rgb.extend(histr)

    return hist_rgb


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


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def dominant_hsv(img, clt):
    dom_hsv = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt.fit(img)
    hist = find_histogram(clt)
    dom_hsv.extend(hist)
    dom_hsv.sort(reverse=True)

    return dom_hsv


def dominant_rgb(img, clt):
    dom_rgb = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt.fit(img)
    hist = find_histogram(clt)
    dom_rgb.extend(hist)
    dom_rgb.sort(reverse=True)

    return dom_rgb


def chalky(img_chalky1, model, model_ext, dict_img):
    for key in list(dict_img.keys()):
        clt = KMeans(n_clusters=4)
        img = imresize(dict_img[key]["image"], (img_dim, img_dim))
        feature_block = list()
        X = list()

        dom_hsv = dominant_hsv(img, clt)
        feature_block.extend(dom_hsv)
        dom_rgb = dominant_rgb(img, clt)
        feature_block.extend(dom_rgb)
        rgb_list = hist_rgb(img)
        feature_block.extend(rgb_list)
        hsv_list = hist_hsv(img)
        feature_block.extend(hsv_list)

        X.append(feature_block)
        X_np = np.array(X)
        print(X_np.shape)
        X_np = X_np.reshape(int(X_np.shape[0]), int(X_np.shape[1]))

        pred = model.predict(X_np)
        pred_prob = model.predict_proba(X_np)
        pred_prob = pred_prob.tolist()
        print("predict probability : ", pred_prob)
        print("Predict Chalky : {}, {}".format(pred, type(pred)))
        print(pred)

        if pred == [1]:
            print("Kernel {} Chalky Grain".format(key))
            dict_img[key]["chalky"] = "no"
            cv2.rectangle(
                img_chalky1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (255, 128, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_chalky1,
                "{}".format("Glutinous"),
                dict_img[key]["loc_text"],
                font,
                2,
                (255, 128, 255),
                5,
                cv2.LINE_AA,
            )

        elif pred == [2]:
            print("Kernel {} Chalky Grain".format(key))
            dict_img[key]["chalky"] = "no"
            cv2.rectangle(
                img_chalky1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (128, 255, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_chalky1,
                "{}".format("Whole"),
                dict_img[key]["loc_text"],
                font,
                2,
                (128, 255, 255),
                5,
                cv2.LINE_AA,
            )

        # elif(pred[0] == '2'):
        else:
            print("Kernel {} Chalky Grain".format(key))
            dict_img[key]["chalky"] = "yes"
            cv2.rectangle(
                img_chalky1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (0, 0, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_chalky1,
                "{}".format("Chalky"),
                dict_img[key]["loc_text"],
                font,
                2,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )
