import os
import csv
import numpy as np

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

# Attributes for our test data
img_width, img_height = 224, 224
batch_size = 20
test_data_dir = '../data/test'
model_name = 'minnet'

test_datagen = ImageDataGenerator(rescale=1.0/ 255)

# We use no class mode and no shuffling because we want the predictions in order
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

# Default ordering is not the same as the directory so manually feed it filenames
ordered = []
for i in range(1, 12501):
    ordered.append(f"testdata/{i}.jpg")

test_generator.filenames = ordered

# Load model and predict
model = load_model(f"../weights/{model_name}.h5")

pred = model.predict_generator(
        test_generator, 
        steps = 12500 // batch_size,
        verbose=1
        ).flatten()

# Write final predictions to file
with open(f"../predictions/{model_name}_predictions.csv", 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["id", "label"])
    for i, v in enumerate(pred):
        writer.writerow([i+1, v])
