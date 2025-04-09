import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
import tensorflow as tf


def read_numpy(np_path):
    X = []
    np_images = np.load("{}".format(np_path))

    for idx, img in enumerate(np_images):
        img = img.astype(np.uint8)
        X.append(img)

    X_np = np.array(X)
    return X_np


# def flatten(l):
#     return np.reshape(l, (len(l), l.shape[1] * l.shape[2] * l.shape[3]))


def flatten(x):
    predict = tf.reshape(x, [len(x), -1])
    return np.array(predict)


# def binary_pixel(data):
#     data[data>0] = 255
#     return data


def binary_pixel(data):
    for idx in range(data.shape[0]):
        data[idx][data[idx] > 0] = 255
    return np.array(data)


def predict2(np_image, deep_model, model):
    X_temp = []
    X_np = np_image

    # Feature Extraction
    data = np.reshape(X_np, (1, 224, 224, 3))
    features = flatten(deep_model.predict(data))
    pred = model.predict_proba(features)[0][0]
    return pred


def predict(np_image, deep_model, model):
    # X_np = np_image
    # Feature Extraction
    # data = np.reshape(X_np, (1, 224, 224, 3))
    features = flatten(deep_model.predict(np_image))
    pred = model.predict(features)
    del features
    return pred
