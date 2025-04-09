import cv2
import numpy as np


def sizing(size, i, img_size, dict_img):
    if size > 4.6:
        cv2.rectangle(
            img_size, dict_img[i]["loc_start"], dict_img[i]["loc_stop"], (0, 255, 0), 6
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_size,
            "{0:.2f}".format(size),
            dict_img[i]["loc_text"],
            font,
            2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    else:
        cv2.rectangle(
            img_size, dict_img[i]["loc_start"], dict_img[i]["loc_stop"], (0, 0, 255), 6
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_size,
            "{0:.2f}".format(size),
            dict_img[i]["loc_text"],
            font,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
