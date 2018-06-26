import os
import csv
import numpy as np

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
batch_size = 20
test_data_dir = '../data/test'
model_name = 'transfer_vgg16'

test_datagen = ImageDataGenerator(rescale=1.0/ 255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

ordered = []
for i in range(1, 12501):
    ordered.append(f"testdata/{i}.jpg")

test_generator.filenames = ordered

model = load_model(f"../weights/{model_name}.h5")

pred = model.predict_generator(
        test_generator, 
        steps = 12500 // batch_size
        ).flatten()

with open(f"../predictions/{model_name}_predictions.csv", 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["id", "label"])
    for i, v in enumerate(pred):
        writer.writerow([i+1, v])