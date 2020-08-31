from tensorflow.keras.losses import categorical_crossentropy

LOSS = categorical_crossentropy
OPTIMIZER = "adam"
INPUT_SHAPE = (256, 256, 3)

TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 32

STEPS_PER_EPOCH = 8
VALIDATION_STEPS = 2
NUM_CLASSES = 4

TRAIN_PATH = "../data/train"
VAL_PATH = "../data/val"
TEST_PATH = "../data/test"
FINAL_MODEL_PATH = "../saved_model/PlantDiseaseClassifierModel.h5"
MODEL_PATH = "../saved_model/Model.h5"