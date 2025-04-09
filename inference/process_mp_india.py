import torch 
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import requests
import os

from return_models import get_dino_finetuned_downloaded, get_classfier
from dataset import CustomDataset

ERROR_ENDPOINT = os.environ["ERROR_ENDPOINT"]
CLASS_MAP = {
    'overall': [0, 1, 2],
    '1121': [0, 1],
    '1509': [0, 2]
}
CLASS_DICT = {0: 'Other', 1: 'Pusa 1121 Basmati', 2: 'Pusa 1509 Basmati'}

logger = logging.getLogger("inference").setLevel(logging.INFO)

transformer = None
clf_model = None

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
    logger.info("Getting embeddings for the input data")
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

    return {'result': result}

def infer(img, model_weight, request_id):
    """
    Perform inference using the DINO model and classifier.
    Args:
        img (torch.Tensor): The input image tensor.
        model_weight (string): The input model name.
        request_id (string): The request id from user
    Returns:
        List: result: Array<{class:string, weight: number}>
    """
    if (len(img) == 0):
        try:
            myobj = {'request_id': request_id, 'error_code': 'image_p1',
                     'reason': 'cannot process image has problem'}
            requests.post(ERROR_ENDPOINT, json=myobj)
        except:
            pass
        raise Exception("Cannot retreive target image")


    embeddings = get_embeddings(img, transformer)
    embeddings = torch.from_numpy(embeddings).float().to("cuda")
    infer_embeddings = CustomDataset(embeddings, None)
    infer_loader = DataLoader(infer_embeddings, batch_size=64, shuffle=False)
    preds = predict_and_transform(model_weight, clf_model, infer_loader)

    return preds

def model_initilize():
    global transformer, clf_model
    transformer_path = "/media/nas/MP/MP_India-weights/simdinov2/training_199999/teacher_checkpoint.pth"
    transformer = get_dino_finetuned_downloaded(model_path=transformer_path, modelname="dinov2_vitb14_reg")
    clf_path = "/media/nas/MP/MP_India-weights/clf/best_model_04-04_14-41-epoch=191-val_acc=0.7654-trinary-non-lanta-tuned.ckpt"
    clf_model = get_classfier(checkpoint_path=clf_path)

model_initilize()