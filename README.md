# Project Title

Image Colorization using UNet

## Overview

This project implements a deep learning pipeline using UNet, aimed at colorizing images. It includes custom data loading, data transformations, and a training loop designed to work efficiently on both CPU and GPU.

## Features

- **Custom Dataset Handling**: Load and split datasets for training and testing.
- **Image Transformations**: Apply normalization and resizing to images.
- **Model Training**: Train a model using the provided dataset and visualize training progress.
- **GPU/CPU Compatibility**: Automatically detects and uses GPU if available.

## Requirements

To run this project, you need the following:

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Other dependencies listed in `requirements.txt` (if available)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
2. Navigate to the project directory:
   ```bash
    cd your-repository
3. Install the required packages:
    ```bash
   pip install -r requirements.txt

## Usage
1. To run the project, use the following command:
    ```bash
   python main.py <dataset_path> <mode>

## Example
    python main.py ./data/train train



