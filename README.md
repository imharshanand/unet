# UNet for Rock Segmentation

This repository contains the implementation of a UNet model for rock segmentation. The model is trained and evaluated using a custom dataset of rock images and their corresponding segmentation masks. This README provides detailed instructions on how to set up, train, and evaluate the model, as well as how to run inference on new images and videos.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Inference](#inference)
  - [Inference on Images](#inference-on-images)
  - [Inference on Videos](#inference-on-videos)
- [System Information](#system-information)

## Installation

To get started, clone this repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
git clone https://github.com/yourusername/unet-rock-segmentation.git
cd unet-rock-segmentation
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt


data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
