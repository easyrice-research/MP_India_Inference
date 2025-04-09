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


def yellow2(img_yellow, model, model_ext, dict_img):
    for key in list(dict_img.keys()):
        img = dict_img[key]["image"]
        # img = imresize(dict_img[key]['image'], (img_dim, img_dim))
        extract_model = Model(
            inputs=model_ext.input,
            outputs=model_ext.get_layer("{}".format("block5_conv2")).output,
        )
        data = np.reshape(img, (1, img_dim, img_dim, 3))
        features = flatten(extract_model.predict(data))
        pred = model.predict(features)
        pred_prob = model.predict_proba(features)
        pred_prob = pred_prob.tolist()
        print("predict probability : ", pred_prob)
        print("Predict Yellow : {}, {}".format(pred, type(pred)))

        if pred[0] == 1:
            print("Kernel {} Yellow Grain".format(key))
            dict_img[key]["yellow"] = "lv1"
            cv2.rectangle(
                img_yellow,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (255, 255, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_yellow,
                "{}".format("lv1"),
                dict_img[key]["loc_text"],
                font,
                2,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

        elif pred[0] == 2:
            print("Kernel {} Yellow Grain".format(key))
            dict_img[key]["yellow"] = "lv2"
            cv2.rectangle(
                img_yellow,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (0, 215, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_yellow,
                "{}".format("lv2"),
                dict_img[key]["loc_text"],
                font,
                2,
                (0, 215, 255),
                5,
                cv2.LINE_AA,
            )

        elif pred[0] == 3:
            print("Kernel {} Yellow Grain".format(key))
            dict_img[key]["yellow"] = "lv3"
            cv2.rectangle(
                img_yellow,
                dict_img[key]["loc_start"],
                dict_img[key]["loc_stop"],
                (2, 106, 253),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_yellow,
                "{}".format("lv3"),
                dict_img[key]["loc_text"],
                font,
                2,
                (2, 106, 253),
                5,
                cv2.LINE_AA,
            )
