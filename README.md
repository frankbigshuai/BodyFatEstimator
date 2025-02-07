# Body Fat Percentage Prediction

## Overview

This project focuses on predicting body fat percentage using deep learning models trained on a dataset of 11000 shirtless human body muscle photos. The goal is to develop an AI-powered system that can estimate body fat percentage based on user-uploaded images.

## Features

- **Image-Based Prediction**: The model takes an input image and estimates the user's body fat percentage.
- **ResNet-50 Pretrained Model**: Utilizes a pretrained ResNet-50 model, fine-tuned to extract meaningful features from images.
- **Two-Step Prediction Strategy**:
  - **Classification Head**: Predicts the body fat percentage range (class label) to which the image belongs.
  - **Regression Head**: Further refines the prediction within the estimated range to provide a precise body fat percentage.
- **Pretrained Model Parameters Provided**: The trained model parameters are directly available, so no additional data processing or training is required.
- **Separate Models for Males and Females**: Trains distinct models for male and female datasets to improve prediction accuracy.
- **Image Segmentation Using U2Net**: Extracts the person from the image, removes the background, and retains only salient regions to prevent background interference.
- **Performance Evaluation**: Uses accuracy, recall, and AUC for classification, and mean absolute error (MAE) for regression assessment.

## Technologies Used

- **Python**
- **PyTorch**
- **ResNet-50 (Pretrained & Fine-tuning Strategy)**
- **U2Net for Image Segmentation**
- **Scikit-learn**
- **NumPy & Pandas**
- **Matplotlib for visualization**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/frankbigshuai/BodyFatEstimator.git
   cd BodyFatEstimator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pretrained model parameters.

## Dataset

- The dataset consists of 11000 images of shirtless individuals, split into male and female subsets.
- U2Net is applied to extract the main subject and remove background noise.

## Model Architecture

- Uses **ResNet-50** pretrained on ImageNet for feature extraction, then fine-tuned on body fat percentage estimation.
- Implements a **classification head** to predict the body fat percentage range.
- Implements a **regression head** to refine predictions within the classified range.
- Fine-tuned separately on male and female body composition data.
- **U2Net** for image segmentation, ensuring only relevant regions are processed.

## Performance Metrics

- **Classification Accuracy**: Measures the correctness of predicted body fat percentage ranges.
- **Recall**: Evaluates model sensitivity.
- **AUC**: Assesses classification quality.
- **Mean Absolute Error (MAE)**: Measures the difference between the predicted and actual body fat percentage in regression.

## Future Improvements

- Implement real-time inference via web and mobile applications.
- Explore multimodal approaches incorporating text-based input (e.g., age, weight, height).

## Contributors

- **Yuntian** - Lead Developer

## License

This project is licensed under the MIT License.

