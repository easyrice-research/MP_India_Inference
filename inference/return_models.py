import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import pytorch_lightning as pl
from inference.models_transformer import TransformerClassifier
logger = logging.getLogger("inference")

# for 224
def get_dino_finetuned_downloaded(model_path, modelname):
    """
    Load a pretrained DINO model from the specified path.
    Args:
        model_path (str): Path to the pretrained model weights.
        modelname (str): Name of the DINO model architecture.
    Returns:
        model (torch.nn.Module): The DINO model with loaded weights.
    """
    model = torch.hub.load('dinov2', 'dinov2_vitb14_reg', source='local', pretrained=False) # load from local file
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vits14_reg": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitb14_reg": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitl14_reg": 1024,
            "dinov2_vitg14": 1536,
            "dinov2_vitg14_reg": 153
        }
        # change shape of pos_embed
        pos_embed = nn.Parameter(torch.zeros(1, 257, input_dims[modelname])) # calculate as ((image_height/patch size) x (image_width/patch size ) + 1)
        model.pos_embed = pos_embed
            
        # load state dict
        msg = model.load_state_dict(new_state_dict, strict=True)
        logger.info('Pretrained weights found at {} and loaded with msg: {}'.format(model_path, msg))
        print("Pretrained weights found at {} and loaded with msg: {}".format(model_path, msg))
    model.to("cuda")
    return model

def get_classfier(checkpoint_path=None):
    """
    Get the classifier model for the DINO model.
    Args:
        checkpoint_path (str): Path to the pretrained classifier weights.
    Returns:
        model (torch.nn.Module): The classifier model.
    """
    classifier_model = TransformerClassifier.load_from_checkpoint(checkpoint_path, strict=False)

    return classifier_model
    