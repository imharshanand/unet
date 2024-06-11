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



### U-Net Architecture

| **Layer Type** | **Details**               |
|----------------|----------------------------|
| **Input Layer**| RGB Image (3 channels)     |
|                |                            |
| **Encoder**    |                            |
| Block 1        | Conv2D (64 filters, 3x3)   |
|                | ReLU                       |
|                | Conv2D (64 filters, 3x3)   |
|                | ReLU                       |
|                | MaxPooling (2x2)           |
| Block 2        | Conv2D (128 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (128 filters, 3x3)  |
|                | ReLU                       |
|                | MaxPooling (2x2)           |
| Block 3        | Conv2D (256 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (256 filters, 3x3)  |
|                | ReLU                       |
|                | MaxPooling (2x2)           |
| Block 4        | Conv2D (512 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (512 filters, 3x3)  |
|                | ReLU                       |
|                | MaxPooling (2x2)           |
|                |                            |
| **Bottleneck** |                            |
|                | Conv2D (1024 filters, 3x3) |
|                | ReLU                       |
|                | Conv2D (1024 filters, 3x3) |
|                | ReLU                       |
|                |                            |
| **Decoder**    |                            |
| UpConv 1       | ConvTranspose2D (512 filters, 2x2) |
|                | Concatenate with Encoder Block 4 |
| Block 1        | Conv2D (512 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (512 filters, 3x3)  |
|                | ReLU                       |
| UpConv 2       | ConvTranspose2D (256 filters, 2x2) |
|                | Concatenate with Encoder Block 3 |
| Block 2        | Conv2D (256 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (256 filters, 3x3)  |
|                | ReLU                       |
| UpConv 3       | ConvTranspose2D (128 filters, 2x2) |
|                | Concatenate with Encoder Block 2 |
| Block 3        | Conv2D (128 filters, 3x3)  |
|                | ReLU                       |
|                | Conv2D (128 filters, 3x3)  |
|                | ReLU                       |
| UpConv 4       | ConvTranspose2D (64 filters, 2x2)  |
|                | Concatenate with Encoder Block 1 |
| Block 4        | Conv2D (64 filters, 3x3)   |
|                | ReLU                       |
|                | Conv2D (64 filters, 3x3)   |
|                | ReLU                       |
|                |                            |
| **Final Layer**| Conv2D (num_classes, 1x1)  |
|                |                            |

This table provides a clear overview of the U-Net model's structure, layer by layer, and shows the relationships between the Encoder, Bottleneck, Decoder, and Final Layer.


# Detailed Explanation of U-Net Architecture Components

#### Encoder

The encoder is responsible for capturing the context in the input image through a series of convolutional and max-pooling layers. It progressively reduces the spatial dimensions while increasing the depth (number of channels) to learn hierarchical features.

1. **Block 1:**
   - **Conv2D (64 filters, 3x3):** Applies 64 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (64 filters, 3x3):** Another convolutional layer with 64 filters of size 3x3.
   - **ReLU:** Activation function.
   - **MaxPooling (2x2):** Reduces the spatial dimensions by a factor of 2.

2. **Block 2:**
   - **Conv2D (128 filters, 3x3):** Applies 128 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (128 filters, 3x3):** Another convolutional layer with 128 filters of size 3x3.
   - **ReLU:** Activation function.
   - **MaxPooling (2x2):** Reduces the spatial dimensions by a factor of 2.

3. **Block 3:**
   - **Conv2D (256 filters, 3x3):** Applies 256 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (256 filters, 3x3):** Another convolutional layer with 256 filters of size 3x3.
   - **ReLU:** Activation function.
   - **MaxPooling (2x2):** Reduces the spatial dimensions by a factor of 2.

4. **Block 4:**
   - **Conv2D (512 filters, 3x3):** Applies 512 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (512 filters, 3x3):** Another convolutional layer with 512 filters of size 3x3.
   - **ReLU:** Activation function.
   - **MaxPooling (2x2):** Reduces the spatial dimensions by a factor of 2.

#### Bottleneck

The bottleneck serves as a bridge between the encoder and decoder, capturing the most compressed representation of the input.

- **Conv2D (1024 filters, 3x3):** Applies 1024 convolutional filters of size 3x3.
- **ReLU:** Activation function.
- **Conv2D (1024 filters, 3x3):** Another convolutional layer with 1024 filters of size 3x3.
- **ReLU:** Activation function.

#### Decoder

The decoder reconstructs the spatial dimensions of the input image while using the hierarchical features captured by the encoder through skip connections.

1. **UpConv 1:**
   - **ConvTranspose2D (512 filters, 2x2):** Upsamples the feature map by a factor of 2.
   - **Concatenate with Encoder Block 4 output:** Combines the upsampled features with corresponding encoder features.

   - **Conv2D (512 filters, 3x3):** Applies 512 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (512 filters, 3x3):** Another convolutional layer with 512 filters of size 3x3.
   - **ReLU:** Activation function.

2. **UpConv 2:**
   - **ConvTranspose2D (256 filters, 2x2):** Upsamples the feature map by a factor of 2.
   - **Concatenate with Encoder Block 3 output:** Combines the upsampled features with corresponding encoder features.

   - **Conv2D (256 filters, 3x3):** Applies 256 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (256 filters, 3x3):** Another convolutional layer with 256 filters of size 3x3.
   - **ReLU:** Activation function.

3. **UpConv 3:**
   - **ConvTranspose2D (128 filters, 2x2):** Upsamples the feature map by a factor of 2.
   - **Concatenate with Encoder Block 2 output:** Combines the upsampled features with corresponding encoder features.

   - **Conv2D (128 filters, 3x3):** Applies 128 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (128 filters, 3x3):** Another convolutional layer with 128 filters of size 3x3.
   - **ReLU:** Activation function.

4. **UpConv 4:**
   - **ConvTranspose2D (64 filters, 2x2):** Upsamples the feature map by a factor of 2.
   - **Concatenate with Encoder Block 1 output:** Combines the upsampled features with corresponding encoder features.

   - **Conv2D (64 filters, 3x3):** Applies 64 convolutional filters of size 3x3.
   - **ReLU:** Activation function.
   - **Conv2D (64 filters, 3x3):** Another convolutional layer with 64 filters of size 3x3.
   - **ReLU:** Activation function.

#### Final Layer

- **Conv2D (num_classes, 1x1):** Produces the final output with the same spatial dimensions as the input image, but with a depth equal to the number of classes (for segmentation).
