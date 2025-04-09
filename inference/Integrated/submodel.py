from joblib import dump, load
from tensorflow.keras.models import load_model
import os

# initial
# path = 'Integrated/Models/'

path = os.environ["INTEGRATED_MODEL_PATH"]

# paddy_model_path = 'Integrated/Models/paddy_model.joblib'
# yellow_model_path = 'Integrated/Models/yellow_model.joblib'
# glutinous_model_path = 'Integrated/Models/glutinous_model.joblib'
# chalky_model_path = 'Integrated/Models/chalky_model.joblib'

# damaged_model_path = 'Integrated/Models/damaged_model.joblib'
# undermilled_model_path = 'Integrated/Models/undermilled_model.joblib'
# deep_model_path = 'Integrated/Models/efficientnet_deep_model'

# weight_model_path = 'Integrated/Models/weight_model.joblib'


def paddy_model(model_size=None):  ## XGboost ##
    paddy_model = load(path + model_size + "paddy_model.joblib")
    return paddy_model


def yellow_model(model_size=None):  ## XGboost ##
    yellow_model = load(path + model_size + "yellow_model.joblib")
    return yellow_model


def glutinous_model(model_size=None):  ## XGboost ##
    glutinous_model = load(path + model_size + "glutinous_model.joblib")
    return glutinous_model


def chalky_model(model_size=None):  ## XGboost ##
    chalky_model = load(path + model_size + "chalky_model.joblib")
    return chalky_model


def damaged_model(model_size=None):  ## Logistc ##
    damaged_model = load(path + model_size + "damaged_model.joblib")
    return damaged_model


def undermilled_model(model_size=None):  ## Logistc ##
    undermilled_model = load(path + model_size + "undermilled_model.joblib")
    return undermilled_model


def undermilled_damaged_model(model_size=None):  ## Logistc ##
    undermilled_damaged_model = load(
        path + model_size + "UndermilledDamaged_model.joblib"
    )
    return undermilled_damaged_model


def head_model(model_size=None):  ## Logistc ##
    head_model = load(path + model_size + "head_model.joblib")
    return head_model


def allclass_model(model_size=None):  ## Logistc ##
    allclass_model = load(path + model_size + "allclass_model.joblib")
    return allclass_model


def shape_model(model_size=None):  ## Logistc ##
    shape_model = load(path + model_size + "shape_model.joblib")
    return shape_model


def efficient_deep_model():  ## EfficientNetB7 ##
    efficient_deep_model = load_model(path + "efficientnet_deep_model")
    return efficient_deep_model


def weight_model(model_size=None):  ## weight ##
    weight_model = load(path + model_size + "weight_model.joblib")
    return weight_model


def defect_shape_model(model_size=None):  ## deep ##
    defect_model = load_model(path + model_size + "model_new_fulldata_jpg.h5")
    # defect_model = load_model(path+model_size+'model_new_fulldata.h5')
    return defect_model
