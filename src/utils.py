from tensorflow.keras.models import load_model
import config
import os


def save(model):
    if not os.path.exists("../saved_model/"):
        os.makedirs('../saved_model/')
    model.save(config.FINAL_MODEL_PATH)


def load():
    model = load_model(config.FINAL_MODEL_PATH)
    return model
