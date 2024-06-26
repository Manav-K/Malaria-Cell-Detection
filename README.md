Malaria Cell Detection Using Deep Learning
Overview
This project aims to detect malaria in cell images using deep learning models. The model is trained to classify images of cells as either infected with malaria or healthy. The project leverages a publicly available dataset from Kaggle, along with PyTorch and the torchvision library.


The dataset used in this project is sourced from Kaggle and contains thousands of cell images categorized into infected and healthy classes.

Dataset URL: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Installation
To get started, clone the repository and install the required dependencies:


git clone https://github.com/manavkakulamarri/deep-learning-project-live.git
cd deep-learning-project-live
pip install -r requirements.txt

Usage
1. Download the dataset:

The dataset can be downloaded directly using the following script, which requires Kaggle API credentials:

import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria'
od.download(dataset_url)

2.
Prepare the dataset:

After downloading, the dataset needs to be organized and prepared for training.

import os
import shutil

data_dir = './cell-images-for-detecting-malaria/cell_images'
os.listdir(data_dir)

3.
Training the Model:

Use the provided scripts to train the model.

python train.py
Evaluating the Model:

4.

Evaluate the trained model's performance on the validation set.

python evaluate.py
Making Predictions:

5.
Use the trained model to make predictions on new cell images.

python predict.py --image_path path_to_image

Training and Evaluation;

Training Procedure

1.
Define the device:

Use GPU if available, otherwise fall back to CPU.

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

2.
Load and preprocess the dataset:

Use torchvision to apply transformations and create data loaders.

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

dataset = ImageFolder(data_dir, transform=...)

3.
Train the model:

Train the model using the training data loader and validate with the validation data loader.

history = fit(epochs, lr, model, train_dl, valid_dl, torch.optim.Adam)

Evaluation
Evaluate the model's performance using accuracy and loss metrics.
Plot the training and validation losses to ensure there is no overfitting.


Results
The model achieved an accuracy of approximately 97% on the validation set after training for 10 epochs.

Example Predictions

def show_image_prediction(img, label):
    plt.imshow(img.permute((1, 2, 0)))
    pred = predict_image(img, model, dataset.classes)
    print('Target:', dataset.classes[label])
    print('Prediction:', pred)
Test the model on new images to verify its predictions.

Contributing
If you would like to contribute to this project, please fork the repository and create a pull request with your changes.


Acknowledgments
The dataset was provided by Kaggle and can be accessed here.
Special thanks to the PyTorch and torchvision communities for their excellent libraries and documentation.
