import config
import dataset
import utils
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


class PlantDiseaseClassifierModel:

    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation="relu", input_shape=config.INPUT_SHAPE))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(config.NUM_CLASSES, activation="softmax"))

        self.model.compile(
            loss=config.LOSS, optimizer=config.OPTIMIZER, metrics=["accuracy"])

        self.model.summary()

    def train(self):
        mc = ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy',
                             mode='max', save_best_only=True, verbose=1)
        self.model.fit_generator(dataset.train_generator, steps_per_epoch=config.STEPS_PER_EPOCH,
                                 epochs=config.EPOCHS, validation_data=dataset.val_generator, validation_steps=config.VALIDATION_STEPS, callbacks=[mc])
        utils.save(self.model)
