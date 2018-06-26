from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K

# Variables defining our inputs and training length
img_width, img_height = 224, 224
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 23000
nb_validation_samples = 2000
epochs = 20
batch_size = 8

# Read from the tensorflow prefs json file
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# The base model name
model = 'raw_resnet'

callbacks = [
        CSVLogger(
            f"../weights/{model}.csv", 
            append=True),
        ModelCheckpoint(
            f"../weights/{model}.h5",
            monitor='val_acc',
            save_best_only=True,
            period=1)
        ]

model = load_model(f"weights/{model}.h5")

print(model.summary())

# Training data generator that applies random transformations
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Test/Validation generator that rescales only and does not apply transformations
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create the training data pipeline
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Create the validation data pipeline
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Fit the model using the training and validation data pipelines
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks)
