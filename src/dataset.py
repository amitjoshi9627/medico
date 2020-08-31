import config
from tensorflow.keras.preprocessing import image

train_datagen = image.ImageDataGenerator(
    rescale=1/255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True,vertical_flip=True)
val_datagen = image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_PATH, target_size=config.TARGET_SIZE, class_mode="categorical", batch_size=config.BATCH_SIZE)
val_generator = val_datagen.flow_from_directory(
    config.VAL_PATH, target_size=config.TARGET_SIZE, class_mode="categorical", batch_size=config.BATCH_SIZE)
