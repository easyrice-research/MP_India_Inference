# %%time
# Intial_Values
list_contour = []
pathout = "./outtest/"
img_dim = 224
dpi = 300
inch = pred_inst.shape[1] / dpi
mm = inch * 25.4
result = mm / pred_inst.shape[1]

img_copy = img.copy()
dict_img = dict()

lbl = pred_inst
thres = overall.astype("uint8")
all_loc = ndimage.find_objects(lbl)
degree = np.zeros(len(list(all_loc)))

for idx, i in enumerate(range(len(list(all_loc)))):
    try:

        #     print(str(i) + '/' + str(nlbl))
        #     listb = arrange_label(loc)
        # Arrange Label
        loc = all_loc[i]
        print(loc)
        mask_img = np.zeros_like(img)
        mask_thres = np.zeros_like(lbl)
        mask_img[np.where(lbl == (i + 1))] = img[np.where(lbl == (i + 1))]
        mask_thres[np.where(lbl == (i + 1))] = 255
        mask_thres = cv2.morphologyEx(
            mask_thres.astype("uint8"), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        # Crop Image from origin picture and Rotate
        crop_img = mask_img[loc[0].start : loc[0].stop, loc[1].start : loc[1].stop]
        crop_thres = mask_thres[loc[0].start : loc[0].stop, loc[1].start : loc[1].stop]
        crop_thres = crop_thres.astype("uint8")

        # Find Most contour area
        contours = contour_area(crop_thres)
        list_contour.append(contours)

        # Find Degree to rotate
        degree = rotated_degree(degree, contours, i)

        rotated = ndimage.rotate(crop_img, degree[i] - 90)
        rotated_thres = ndimage.rotate(crop_thres, degree[i] - 90)
        contours1 = contour_area(rotated_thres)

        # Mask only Grain
        topx, bottomx, topy, bottomy, out = mark_only_grain(rotated, contours1)
        out2 = out[topx:bottomx, topy:bottomy]

        # Calculate Sizing
        length = (bottomx - topx) * result
        width = (bottomy - topy) * result

        # Image Resizing
        img_resize = image_resizing(topx, bottomx, bottomy, topy, out, i, img_dim)

        # Write Text
        cv2.rectangle(
            img_copy,
            (loc[1].start - 5, loc[0].start - 5),
            (loc[1].stop + 5, loc[0].stop + 7),
            (0, 0, 255),
            6,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_copy,
            str(i),
            (loc[1].start, loc[0].start - 20),
            font,
            2,
            (0, 0, 255),
            5,
            cv2.LINE_AA,
        )

        dict_img[i] = {"image": img_resize}
        dict_img[i]["image_crop"] = out2
        dict_img[i]["loc_start"] = (loc[1].start - 5, loc[0].start - 5)
        dict_img[i]["loc_stop"] = (loc[1].stop + 5, loc[0].stop + 7)
        dict_img[i]["loc_text"] = (loc[1].start, loc[0].start - 20)
        dict_img[i]["size"] = length

    except Exception as e:
        print("i is :", i)
        print(e)
