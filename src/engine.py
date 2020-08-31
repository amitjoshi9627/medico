import config
import numpy as np
import utils
from tensorflow.keras.preprocessing import image


def train_fn(model):
    model.train()


def get_results(img):
    img = image.load_img(img, target_size=config.TARGET_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    model = utils.load()
    result = model.predict_classes(img)
    return result[0]
