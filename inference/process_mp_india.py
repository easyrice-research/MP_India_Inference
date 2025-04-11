from functools import partial
from inference.functions.crop_utils import *
from inference.functions.purity_utils import *
from inference.functions.weight_feature import *
import torch 
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
# import logging
from tqdm import tqdm
import requests
import os
from multiprocessing import get_context
import cv2

from inference.return_models import get_dino_finetuned_downloaded, get_classfier
from inference.process_dataset import CustomDataset

ERROR_ENDPOINT = os.environ["ERROR_ENDPOINT"]
CLASS_MAP = {
    'overall': [0, 1, 2],
    'Pusa_1121_Basmati': [0, 1],
    'Pusa_1509_Basmati': [0, 2]
}
CLASS_DICT = {0: 'Other', 1: 'Pusa 1121 Basmati', 2: 'Pusa 1509 Basmati'}

# logger = logging.getLogger("inference").setLevel(logging.INFO)

transformer = None
clf_model = None

def draw_img(input_img,draw_size):
    blur_img = cv2.blur(input_img,(5,5)) 
    
    gray_img = cv2.cvtColor(blur_img,cv2.COLOR_RGB2GRAY)
            
    img = gray_img.copy()
    ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)
    img_black = np.zeros([224,224,3],dtype=np.uint8)
    img_black_draw = cv2.drawContours(img_black, c, -1, (255,255,255),draw_size)
    return img_black_draw

def top_bottom(img_to_rotate):
    image = img_to_rotate.copy()
    image = cv2.cvtColor(img_to_rotate,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(image,50,255,cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return img_to_rotate,top,bottom

def predict_defect_shape(img,top,bottom):
    if top[1] <= 10:
        img_selected_top = img[top[1]:top[1]+10,:,:]
    else:
        img_selected_top = img[top[1]-10:top[1]+10,:,:]
        
    if bottom[1] >= 214:
        img_selected_bottom = img[bottom[1]-10:bottom[1],:,:]
    else:
        img_selected_bottom = img[bottom[1]-10:bottom[1]+10,:,:]

    img_selected_top_gray = cv2.cvtColor(img_selected_top,cv2.COLOR_BGR2GRAY)
    img_selected_bottom_gray = cv2.cvtColor(img_selected_bottom,cv2.COLOR_BGR2GRAY)

    ret_top,thresh_top = cv2.threshold(img_selected_top_gray,30,255,cv2.THRESH_BINARY)
    ret_bottom,thresh_bottom = cv2.threshold(img_selected_bottom_gray,30,255,cv2.THRESH_BINARY)

    cnts_top,hierarchy_top = cv2.findContours(thresh_top,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_bottom,hierarchy_bottom = cv2.findContours(thresh_bottom,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    c_top = sorted(cnts_top, key = cv2.contourArea, reverse = True)
    c_bottom = sorted(cnts_bottom, key = cv2.contourArea, reverse = True)

    if len(c_top)<=1 and len(c_bottom)<=1:
        return 0 #normal

    elif len(c_top)>1 or len(c_bottom)>1:
        return 1 #defect

def map_class_names(model: str, class_map: dict, class_dict: dict):
    return {i: class_dict[i] if i in class_map[model] else 'None' for i in class_dict}

def preprocess_image(image):
    """
    Preprocess the input image for the DINO model.
    Args:
        image: The input image."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])

    return transform(image)
    
def get_embeddings(images, transformer):
    """
    Get the embeddings for the input data using the DINO model.
    Args:
        imgs : List of images.
        transformer (torch.nn.Module): The DINO model."""
    
    embeddings = []
    for image in tqdm(images, desc="Processing images", unit="image"):
        transformer.eval()
        with torch.no_grad():
            image = Image.fromarray(image) # Load image
            image = preprocess_image(image)  # Apply normalization
            image = image.unsqueeze(0) # Add batch dimension
            image = transformer(image.to("cuda"))
            embeddings.append(image.cpu().numpy())

    return np.array(embeddings)

def predict_and_transform(model_weight, model, infer_loader):
    """
    Predict the classes for the input data using the classifier model.
    Args:
        model (torch.nn.Module): The classifier model.
        val_loader (DataLoader): DataLoader"""
    
    class_names = map_class_names(model_weight, CLASS_MAP, CLASS_DICT)
    
    result = []
    # Disable gradient computation for inference
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        for batch in infer_loader:
            x_batch = batch[0].to("cuda")
            x_batch = x_batch.squeeze(1)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)  # Get the predicted class index for each sample
            result.append([{"class": class_names[preds.cpu().numpy()], "weight": 0}])  # Move back to CPU and store the predictions

    print(f"predict result: {result}")
    return {'result': result}

def infer(img=None, pred_inst=None, pred_type=None, pred_inst_centroid=None, model_weight="overall", request_id=None):
    """
    Perform inference using the DINO model and classifier.
    Args:
        img (torch.Tensor): The input image tensor.
        model_weight (string): The input model name.
        request_id (string): The request id from user
    Returns:
        List: result: Array<{class:string, weight: number}>
    """
    
    pred_inst_centroids = pred_inst_centroid
    # For weight_func
    kernel_attrs = []
    batch_size = 1024
    list_inst = []
    im = []
    _, counts_pixel = np.unique(pred_inst, return_counts=True)
    counts_pixel = np.delete(counts_pixel, 0)
    insts_list = list(np.unique(pred_inst))
    insts_list.remove(0)

    if len(insts_list) > 0:
        # load model
        dpi = 312

        inch = pred_inst.shape[1] / dpi
        mm = inch * 25.4
        ans = mm / pred_inst.shape[1]
        
        _ , globle_max_x = np.where(pred_inst[:,-40:]>0)
#         print(globle_max_x)
        try :
            globle_max_x = max(globle_max_x) -1
        except :
            globle_max_x = -1

        func = partial(preprocess, pred_inst, pred_type, img, ans , globle_max_x)

#         func = partial(preprocess, pred_inst, pred_type, img, ans)

        cpu_count = os.cpu_count() // 2 or 1
        with get_context("spawn").Pool(cpu_count) as p:
            for preprocessed in p.map(
                func, zip(insts_list, counts_pixel, pred_inst_centroids)
            ):
                if preprocessed is None:
                    continue
                (
                    kernel_pic,
                    _,
                    inst_id,
                    length,
                    width,
                    pixel_count,
                    pred_inst_centroid,
                    kernel_pos,
                ) = preprocessed
                list_inst.append(inst_id)
                im.append(kernel_pic)
                kernel_attrs.append(
                    (inst_id, length, width, pixel_count, pred_inst_centroid)
                )
    if (len(im) == 0):
        try:
            myobj = {'request_id': request_id, 'error_code': 'image_p1',
                     'reason': 'cannot process image has problem'}
            requests.post(ERROR_ENDPOINT, json=myobj)
        except:
            pass
        raise Exception("Cannot retreive target image")

    x_test = np.asarray(im)

    list_im = []
    list_defect_2 = []
    # if model_weight in ['overall_PT1', 'nfavor_ml_1', 'dy_gk79_1']:
    for each_im in x_test:
        uint8_img = each_im.astype('uint8')
        try:
           img_black_draw = draw_img(uint8_img,2)
           top_bottom_img,top,bottom  = top_bottom(img_black_draw)
           y_defect = predict_defect_shape(top_bottom_img,top,bottom)
        except:
           y_defect = 1
        
        list_defect_2.append(y_defect)
        list_im.append(rotate_image(uint8_img))

    x_test = np.asarray(list_im)
    
    embeddings = get_embeddings(x_test, transformer)
    embeddings = torch.from_numpy(embeddings).float().to("cuda")
    infer_embeddings = CustomDataset(embeddings, None)
    infer_loader = DataLoader(infer_embeddings, batch_size=64, shuffle=False)
    preds = predict_and_transform(model_weight, clf_model, infer_loader)

    return preds

def model_initilize():
    global transformer, clf_model
    transformer_path = "/models/mp-india-models/teacher_checkpoint.pth"
    transformer = get_dino_finetuned_downloaded(model_path=transformer_path, modelname="dinov2_vitb14_reg")
    clf_path = "/models/mp-india-models/best_model_04-04_14-41-epoch=191-val_acc=0.7654-trinary-non-lanta-tuned.ckpt"
    clf_model = get_classfier(checkpoint_path=clf_path)

model_initilize()