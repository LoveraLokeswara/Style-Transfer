# Neural Style Transfer with Classification

This project implements neural style transfer using Adaptive Instance Normalization (AdaIN) and includes a classification task component. The implementation is designed to run on Google Colab.

## Overview

The project consists of two main parts:

### Part 1: Style Transfer
- Implements neural style transfer using the AdaIN (Adaptive Instance Normalization) technique
- Uses a pre-trained VGG19 encoder to extract content and style features
- Includes a custom decoder network to generate stylized images
- Supports controllable style transfer intensity via an alpha parameter
- Trained on WikiArt (style images) and COCO (content images) datasets

### Part 2: Classification Task
- Image classification component using the PACS dataset
- Classifies images across different styles and categories
- Includes data augmentation capabilities

## Project Structure

```
Style-Transfer/
├── Style_Transfer.ipynb    # Main notebook with implementation
├── classify.h5            # Trained classification model
├── classify-augmentation.h5  # Classification model with augmentation
├── saved.h5               # Saved style transfer model
└── report.pdf             # Project report
```

## Key Features

- **Custom Dataset Classes**: 
  - `ImageDataset`: Loads images for style transfer training
  - `ClassificationDataset`: Loads images with labels for classification tasks

- **AdaIN Layer**: Custom Keras layer implementing Adaptive Instance Normalization for style transfer

- **VGG19 Encoder**: Pre-trained VGG19 network used for feature extraction

- **Custom Decoder**: Upsampling network to reconstruct stylized images

## Requirements

The project requires the following Python libraries:
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV (cv2)
- tqdm

## Usage

### Running on Google Colab

1. Open `Style_Transfer.ipynb` in Google Colab
2. Mount your Google Drive containing the datasets
3. Run the cells sequentially to:
   - Set up the environment
   - Load datasets
   - Train the style transfer model
   - Train the classification model
   - Generate stylized images

### Datasets

The project uses three datasets:
- **WikiArt**: Style images for training the style transfer model
- **COCO**: Content images for training the style transfer model
- **PACS**: Classification dataset with multiple styles (sketch, cartoon, photo, art painting)

## Model Components

- **Encoder**: VGG19 up to conv4_1 (frozen, pre-trained weights)
- **AdaIN Layer**: Normalizes content features using style statistics
- **Decoder**: Mirror architecture of encoder with upsampling layers

## Training

The training process includes:
- Content loss: Measures similarity between generated and content features
- Style loss: Measures statistical similarity across multiple VGG19 layers
- Adaptive learning rate and weight decay
- Model checkpointing at regular intervals

## Results

The trained models can generate images that combine the content of one image with the artistic style of another, with adjustable style intensity.
