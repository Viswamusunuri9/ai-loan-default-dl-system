import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "models/loan_model.keras"  # or .h5 if you didn’t change


def load_trained_model():
    model = load_model(MODEL_PATH)
    return model


def predict(model, input_data):
    """
    input_data: numpy array (1, n_features)
    """

    prob = model.predict(input_data)[0][0]

    if prob > 0.5:
        label = "High Risk (Default)"
    else:
        label = "Low Risk (Safe)"

    return prob, label