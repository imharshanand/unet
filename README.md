# U-Net Segmentation Model for Rock Detection
This repository contains the implementation of a U-Net model for semantic segmentation, specifically designed for detecting rocks in images. The project includes data loading, model training, evaluation, and inference steps.

## Table of Contents
Introduction
Dataset
Model Architecture
Dependencies
Training the Model
Running Inference
Results
Usage
References
Introduction
This project implements a U-Net model for the task of semantic segmentation. The primary goal is to detect rocks in images, classifying each pixel as either part of a rock or the background. The U-Net architecture is well-suited for this task due to its encoder-decoder structure, which captures both local and global features.

## Dataset
The dataset consists of images and their corresponding masks. Each mask is a binary image where the pixel value is 1 if the pixel belongs to a rock and 0 otherwise.

Images Directory: Contains the RGB images.
Masks Directory: Contains the binary masks.
The dataset is split into training and validation sets to evaluate the model's performance.

## Model Architecture
The U-Net model used in this project consists of an encoder-decoder structure:

* Encoder: Four convolutional blocks with max-pooling layers to reduce the spatial dimensions.
* Bottleneck: A convolutional block that connects the encoder and decoder.
* Decoder: Four up-convolutional blocks with skip connections from the corresponding encoder layers.
* Final Layer: A convolutional layer that outputs the final segmentation map

## Dependencies
Ensure you have the following libraries installed:  
(Check unet_env_HowToCreate for more details)

* torch
* torchvision
* Pillow
* numpy
* matplotlib
* tqdm
