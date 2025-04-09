import cv2
import numpy as np


def glutinous(image, i, img_glu, dict_img):
    try:
        img = image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, thres1 = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
        _, thres2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        _, thres3 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

        thres_mol1 = cv2.morphologyEx(
            thres1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        thres_mol2 = cv2.morphologyEx(
            thres2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        thres_mol3 = cv2.morphologyEx(
            thres3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        thres_mol1_c = np.sum(thres_mol1 == 255)
        thres_mol2_c = np.sum(thres_mol2 == 255)
        thres_mol3_c = np.sum(thres_mol3 == 255)

        compare_12 = (thres_mol2_c / thres_mol1_c) * 100
        compare_23 = (thres_mol3_c / thres_mol2_c) * 100

        if (compare_12 > 70) and (compare_23 > 30):
            print("this is kao niew")
            dict_img[i]["glutinous"] = "yes"

            cv2.rectangle(
                img_glu,
                dict_img[i]["loc_start"],
                dict_img[i]["loc_stop"],
                (255, 0, 255),
                6,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_glu,
                "YES",
                dict_img[i]["loc_text"],
                font,
                2,
                (255, 0, 255),
                5,
                cv2.LINE_AA,
            )

        else:
            print("this is not kao niew")
            dict_img[i]["glutinous"] = "no"
    except:
        print("Error : picture no :", i)
