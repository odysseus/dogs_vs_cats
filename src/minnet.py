from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras import backend as K

# Variables defining our inputs and training length
img_width, img_height = 224, 224
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 2048
nb_validation_samples = 2000
epochs = 100
batch_size = 32

# Read from the tensorflow prefs json file
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Create a logger
csv_logger = CSVLogger('../weights/minnet.csv', append=False)

# Create a checkpoint object
model_checkpoint = ModelCheckpoint(
        '../weights/minnet.h5', 
        monitor='val_acc', 
        save_best_only=True,
        period=1)

# Create the model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# The optimizer
adam = Adam(lr=1e-4)

# Compile using binary cross entropy as loss function
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

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
    callbacks=[csv_logger, model_checkpoint])
