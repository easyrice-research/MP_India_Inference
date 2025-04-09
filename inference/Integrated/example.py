import submodel
import predict_submodel
import predict_deep_submodel
import numpy as np

np_image = np.zeros(224, 224, 3)

"""
    predict_submodel
        • yellow_model_path -> XGBoost
        • glutinous_model_path -> XGBoost
        • chalky_model_path -> XGBoost
"""

yellow_model = submodel.yellow_model()
pred = predict_submodel.predict(np_image, yellow_model)

"""
    predict_deep_submodel
        • deep_model_path -> EfficientNet
        • undermilled_model -> Logistic Reg
        • damaged_model -> Logistic Reg
"""

deep_model = submodel.efficient_deep_model()
undermilled_model = submodel.undermilled_model()
pred = predict_deep_submodel.predict(np_image, deep_model, undermilled_model)
