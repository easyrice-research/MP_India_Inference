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


def broken(img_broken, model, model_ext, dict_img):
    list_pred = []

    for key in list(dict_img.keys()):
        if dict_img[key]["paddy"] == "w":
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
            print("Predict Broken : {}, {}".format(pred, type(pred)))

            # if((pred[0] == 1) & (round(pred_prob[0][0]*100, 2) > 80.00)):
            if pred == [1] and pred_prob[0][0] >= 0.90:
                print("Kernel {} Broken Grain".format(key))
                dict_img[key]["broken"] = "broke"
                cv2.rectangle(
                    img_broken,
                    dict_img[key]["loc_start"],
                    dict_img[key]["loc_stop"],
                    (204, 0, 0),
                    4,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][0]*100, 2)), dict_img[key]['loc_text'], font, 2, (2, 106, 253), 5, cv2.LINE_AA)

            elif pred == [1] and pred_prob[0][0] <= 0.90:
                print("Kernel {} Broken Grain".format(key))
                dict_img[key]["broken"] = "head"
                cv2.rectangle(
                    img_broken,
                    dict_img[key]["loc_start"],
                    dict_img[key]["loc_stop"],
                    (0, 204, 0),
                    4,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][0]*100, 2)), dict_img[key]['loc_text'], font, 2, (2, 106, 253), 5, cv2.LINE_AA)

            elif pred == [2] and pred_prob[0][1] >= 0.90:
                print("Kernel {} Head Grain".format(key))
                dict_img[key]["broken"] = "head"
                cv2.rectangle(
                    img_broken,
                    dict_img[key]["loc_start"],
                    dict_img[key]["loc_stop"],
                    (0, 204, 0),
                    4,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][0]*100, 2)), dict_img[key]['loc_text'], font, 2, (2, 106, 253), 5, cv2.LINE_AA)

            elif pred == [2] and pred_prob[0][1] < 0.90:
                print("Kernel {} Head Grain".format(key))
                dict_img[key]["broken"] = "whole"
                cv2.rectangle(
                    img_broken,
                    dict_img[key]["loc_start"],
                    dict_img[key]["loc_stop"],
                    (0, 0, 204),
                    4,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][0]*100, 2)), dict_img[key]['loc_text'], font, 2, (2, 106, 253), 5, cv2.LINE_AA)

            else:
                print("Kernel {} Whole Grain".format(key))
                dict_img[key]["broken"] = "whole"
                cv2.rectangle(
                    img_broken,
                    dict_img[key]["loc_start"],
                    dict_img[key]["loc_stop"],
                    (0, 0, 204),
                    4,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_broken, '{}'.format(round(pred_prob[0][1]*100, 2)), dict_img[key]['loc_text'], font, 2, (34, 139, 34), 5, cv2.LINE_AA)

        else:
            continue

    return dict_img
