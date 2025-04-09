import cv2
import numpy as np
from scipy.misc import imresize

img_dim = 224


def hist_hsv(img):
    hist_hsv = list()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = ("r", "g", "b")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [64], [0, 256])
        np_result = np.sqrt(np.sum(np.square(histr), axis=0))
        histr = histr / sum(histr)
        hist_hsv.extend(histr)

    return hist_hsv


def red_ml(img_red1, model, model_ext, dict_img):

    for key in list(dict_img.keys()):
        img = imresize(dict_img[key]["image_crop"], (img_dim, img_dim))
        # img = dict_img[key]['image_crop']
        feature_block = list()
        X = list()
        hsv_list = hist_hsv(img)
        feature_block.extend(hsv_list)
        print(len(feature_block))
        X.append(feature_block)
        X_np = np.array(X)
        print(X_np.shape)
        X_np = X_np.reshape(int(X_np.shape[0]), int(X_np.shape[1]))

        pred = model.predict(X_np)
        pred_prob = model.predict_proba(X_np)
        pred_prob = pred_prob.tolist()
        print("predict probability : ", pred_prob)
        print("Predict Red : {}, {}".format(pred, type(pred)))
        print(pred)

        if pred == 0:
            print("Kernel {} Red Grain".format(key))
            dict_img[key]["red"] = "no"
            cv2.rectangle(
                img_red1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (255, 255, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_red1,
                "{}".format("No"),
                dict_img[key]["loc_text"],
                font,
                2,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

        # elif(pred[0] == '2'):
        else:
            print("Kernel {} Red Grain".format(key))
            dict_img[key]["red"] = "yes"
            cv2.rectangle(
                img_red1,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (0, 0, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_red1,
                "{}".format("Yes"),
                dict_img[key]["loc_text"],
                font,
                2,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )
