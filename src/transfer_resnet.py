from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

# Variables defining our inputs and training length
img_width, img_height = 224, 224
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 2048
nb_validation_samples = 2000
epochs = 100
batch_size = 16

# Read from the tensorflow prefs json file
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Create a logger
csv_logger = CSVLogger('../weights/transfer_resnet.csv', append=False)

# Create a checkpoint object
model_checkpoint = ModelCheckpoint(
        '../weights/transfer_resnet.h5',
        monitor='val_acc',
        save_best_only=True,
        period=1)

# Loads the pre-trained ResNet50 model without the top layer as our base
base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

# Save a reference to the output layer of the base model
x = base_model.output

# Add an average pooling layer after the output of the base model
# remember that the base model has been stripped of the top layer
x = GlobalAveragePooling2D()(x)

# Adding a dense layer after the average pooling
x = Dense(1024, activation='relu')(x)

# Finally adding a classification layer after the dense layer
predictions = Dense(1, activation='sigmoid')(x)

# Define our model by passing the input and output layer of the network
model = Model(inputs=base_model.input, outputs=predictions)

# Set the entire base model to be not trainable while the new layers learn
for layer in base_model.layers:
    layer.trainable = False

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
