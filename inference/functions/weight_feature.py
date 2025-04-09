import numpy as np
from inference.functions.purity_utils import *
from inference.misc.utils import get_bounding_box

def weight_feature(
    model_type, l_channel, pred_inst, kernel_pos, kernel, kernel_type
):
    inst_id, _, _, pixel_count, _ = kernel
    y1, y2, x1, x2 = kernel_pos
    l_channel = l_channel[y1:y2, x1:x2]
    lab = l_channel[np.where(pred_inst[y1:y2, x1:x2] == inst_id)]
    b = [i for i in range(0, 280, 30)]
    hist, _ = np.histogram(lab, bins=b)
    feature = hist / pixel_count

    w_bases = [
        9.888136660434709e-06,
        9.218318742716537e-06,
        8.151976981269914e-06,
        9.943307462734612e-06,
        9.435210959200047e-06,
    ]

    # ข้าวเต้ม ขาวหัก ข้าวเปลือก แดง glu
    # 1.White','2.Red','3.Paddy','4.Yellow','5.Glutinous','6.Chalky','7.Damaged','8.Undermilled'
    if model_type == "QA":
        try:
            w_base = w_bases[2]
        except:
            print((kernel_type,))

    elif model_type == "QC":
        if kernel_type in [4]:
            w_base = 0
        else:
            w_base = w_bases[kernel_type - 1]
    feature = np.concatenate([feature, [pixel_count * w_base]])

    return feature

def preprocess(pred_inst, pred_type, img, ans,globle_max_x, args):
    inst_id, pixel_count, pred_inst_centroid = args
    inst_map = np.array(pred_inst == inst_id, np.uint8)
    y1, y2, x1, x2 = get_bounding_box(inst_map)
    
    if globle_max_x >= 0 and ((pred_inst.shape[1]-1)-40+globle_max_x < x2+4): 
        kernel = np.ones((3,3),np.uint8)
        tempk_inst_map = cv2.morphologyEx(inst_map[y1:y2+1, x1:x2+1].astype('uint8'), cv2.MORPH_OPEN, kernel)
        _,k_max_x = np.where( tempk_inst_map>0)
        count_max_x = list(k_max_x).count(max(k_max_x))
        print((globle_max_x,(pred_inst.shape[1]-1)-40+globle_max_x ,x2,count_max_x))

        if count_max_x >= 5 :
            return
        
    y1 = y1 - 2 if y1 - 2 >= 0 else -1
    x1 = x1 - 2 if x1 - 2 >= 0 else -1
    x2 = x2 + 2 if x2 + 2 <= pred_inst.shape[1] - 1 else -1
    y2 = y2 + 2 if y2 + 2 <= pred_inst.shape[0] - 1 else -1
    
    if -1 in [x1,x2,y1,y2]:
        return 

    # edit
    inst_pred_type = pred_type[y1:y2, x1:x2]
    inst_pred_type = inst_pred_type[inst_pred_type > 0]
    counts_type = np.bincount(inst_pred_type)
    kernel_type = 0
    try:
        kernel_type = np.argmax(counts_type)
    except:
        # kernel pic too big
        return
    try:
        kernel_pic, length, width = kernel_length_width(
            img=img, inst_map=inst_map, pos=[y1, y2, x1, x2], ans=ans
        )
    except:
        # kernel pic too big
        return

    return (
        kernel_pic,
        kernel_type,
        inst_id,
        length,
        width,
        pixel_count,
        pred_inst_centroid,
        [y1, y2, x1, x2],
    )
