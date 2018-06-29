import os
import csv
import numpy as np

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

# Typical setup to feed in data
img_width, img_height = 224, 224
batch_size = 20
validation_dir = '../data/validation'
model_name = 'transfer_resnet'

valid_datagen = ImageDataGenerator(rescale=1.0/ 255)

valid_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False)

# Load the model and predict
model = load_model(f"../weights/{model_name}.h5")

pred = model.predict_generator(
        valid_generator, 
        steps = 2000 // batch_size,
        verbose=1
        ).flatten()

# Threshold into classes
tpred = np.copy(pred)
tpred[tpred > 0.50] = 1
tpred[tpred < 0.499999999999999] = 0
tpred

# Find where the predictions and the actual classes are not equal
incorrect = []
for i, v in enumerate(tpred):
    if v != valid_generator.classes[i]:
        incorrect.append(1)
    else:
        incorrect.append(0)
        
print(f"Incorrect: {sum(incorrect)}")

# Find the filenames of the incorrect answers
badfiles = []
for i, v in enumerate(incorrect):
    if v == 1:
        badfiles.append(valid_generator.filenames[i])

# Write these to a file
with open(f"../vis/{model_name}_incorrect.txt", 'w') as f:
    for fn in badfiles:
        f.write(f"{fn}\n")
