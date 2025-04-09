from keras.applications.vgg16 import VGG16
from scipy.misc import imresize
from keras.models import Model
import cv2
import numpy as np

img_dim = 224


def flatten(l):
    print(np.shape(l))
    if len(np.shape(l)) > 2:
        return np.reshape(l, (len(l), l.shape[1] * l.shape[2] * l.shape[3]))
    else:
        return l


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


def paddy(img_paddy, model, model_color, model_ext, dict_img):
    list_pred = []
    for key in list(dict_img.keys()):
        # img = imresize(dict_img[key]['image'], (img_dim, img_dim))
        img = dict_img[key]["image"]
        extract_model = Model(
            inputs=model_ext.input,
            outputs=model_ext.get_layer("{}".format("fc2")).output,
        )
        data = np.reshape(img, (1, img_dim, img_dim, 3))
        features = flatten(extract_model.predict(data))
        pred = model.predict(features)
        pred_prob = model.predict_proba(features)
        pred_prob = pred_prob.tolist()
        print("predict probability : ", pred_prob)
        print("Predict Paddy : {}, {}".format(pred, type(pred)))

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
        pred_color = model_color.predict(X_np)
        pred_prob_color = model_color.predict_proba(X_np)
        pred_prob_color = pred_prob_color.tolist()

        print(pred)
        print(pred_prob)
        print(pred_color)
        print(pred_prob_color)
        if pred_prob[0][1] <= 0.95 and pred_prob_color[0][0] >= 0.30:
            print("Kernel {} Paddy Grain".format(key))
            dict_img[key]["paddy"] = "p"
            cv2.rectangle(
                img_paddy,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (204, 0, 0),
                4,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][0]*100, 2)), dict_img[key]['loc_text'], font, 2, (2, 106, 253), 5, cv2.LINE_AA)

        else:
            print("Kernel {} Whole Grain".format(key))
            dict_img[key]["paddy"] = "w"
            # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][1]*100, 2)), dict_img[key]['loc_text'], font, 2, (34, 139, 34), 5, cv2.LINE_AA)

        list_pred.extend(pred_prob)

    return dict_img
