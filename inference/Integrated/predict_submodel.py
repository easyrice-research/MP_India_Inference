import cv2
import numpy as np
import xgboost as xgb
from joblib import load, dump

## RGB Channel
def hist_rgb_selected(img, i):
    hist_rgb = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = ("r", "g")  # r, g

    for idx, col in enumerate(color):
        histr = cv2.calcHist([img], [idx], None, [50], [5, 255])
        hist_rgb.extend(histr.reshape(50))

    return hist_rgb


## LAB Channel
def hist_lab_selected(img, i):
    hist_lab = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color = "b"
    for idx, col in enumerate(color):
        histr = cv2.calcHist([img], [idx + 2], None, [50], [5, 255])
        hist_lab.extend(histr.reshape(50))

    return hist_lab


## HSV Channel
def hist_hsv_selected(img, i):
    hist_hsv = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = ("r", "g", "b")
    for idx, col in enumerate(color):
        histr = cv2.calcHist([img], [idx], None, [50], [5, 255])
        hist_hsv.extend(histr.reshape(50))

    return hist_hsv


def feature_extraction(np_image):
    X = []
    img = np_image

    feature_block = list()
    rgb_list = hist_rgb_selected(img, 50)
    feature_block.extend(rgb_list)

    lab_list = hist_lab_selected(img, 50)
    feature_block.extend(lab_list)

    hsv_list = hist_hsv_selected(img, 50)
    feature_block.extend(hsv_list)

    X.append(feature_block)

    X_np = np.array(X)
    X_np = X_np.reshape(1, int(X_np.shape[1]))

    return X_np


def predict(np_image, model):
    X_np = feature_extraction(np_image.astype("uint8"))
    pred = model.predict_proba(X_np)[0][0]
    pred = 0 if pred >= 0.9 else 1
    return pred
