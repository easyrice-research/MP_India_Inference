import cv2
import numpy as np


def red(image, i, img_red, dict_img):
    try:
        img = image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        _, thres_red = cv2.threshold(img_lab[:, :, 1], 139, 255, cv2.THRESH_BINARY)
        _, thres_gray = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thres = np.logical_or(thres_red, thres_gray)

        thres = thres.astype("uint8")
        thres[thres == 1] = 255

        thres_red_sum = np.sum(thres_red == 255)
        thres_sum = np.sum(thres == 255)
        thres_gray_sum = np.sum(thres_gray == 255)

        print(thres_gray_sum)
        print(thres_sum)
        if thres_red_sum > 90:
            print("this is red")
            dict_img[i]["red"] = "yes"

            cv2.rectangle(
                img_red,
                dict_img[i]["loc_start"],
                dict_img[i]["loc_stop"],
                (255, 0, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_red,
                "YES",
                dict_img[i]["loc_text"],
                font,
                2,
                (255, 0, 255),
                5,
                cv2.LINE_AA,
            )

        else:
            print("this is not red")
            dict_img[i]["red"] = "no"

    except:
        print("Error : picture no :", i)
