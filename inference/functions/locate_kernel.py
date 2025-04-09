import cv2
import numpy as np


def location(loc, i, img_loc, dict_img):
    cv2.rectangle(
        img_loc,
        (loc["loc_start"][0], loc["loc_start"][1]),
        (loc["loc_stop"][0], loc["loc_stop"][1]),
        (0, 0, 255),
        6,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img_loc,
        str(i),
        (loc["loc_text"][0], loc["loc_text"][1]),
        font,
        1,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return img_loc
