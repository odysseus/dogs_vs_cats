# Dataset and Software Requirements

## Dataset

The dataset can be downloaded in its entirety from the following Kaggle competition:

[Dogs vs. Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Software

For simply evaluating the model video card computation is probably not required. The libraries are highly hardware dependent so refer to the instructions given on the site. The following stack is in use on my machine to train the models:

1. [CUDA Libary](https://developer.nvidia.com/gpu-accelerated-libraries)
2. [TensorFlow Backend](https://www.tensorflow.org/install/)
3. [Keras Frontend](https://keras.io/#installation)
4. [Anaconda Install](https://www.anaconda.com/download)

Additional Python libraries may be required based on your setup. While I haven't tried another setup personally, in theory any installation that makes Keras work should be sufficient.

## Project Download

1. The project git can be found here: [Github](https://github.com/odysseus/dogs_vs_cats)
  - `data` is not present in the project git but would contains three subfolders: `test`, `train` and `validation`, where `test` contains the submission test images.
  - `img` contains images for the writeup and does not have any bearing on the code.
  - `predictions` contains predictions for Kaggle submission.
  - Code files are in the `src` directory.
  - `vis` is simply a Jupyter notebook for creating graphs and can be safely ignored.
  - The `weights` directory contains csv files for the model runs.
2. Github cannot handle the model files, so the individual models can be loaded via [Google Drive](https://drive.google.com/open?id=11ayvW8_AEXkY42S7sNZIn5aeWtnylx--)
