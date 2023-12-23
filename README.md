# Time Series Prediction with LSTM

This repository contains the code for training and testing a deep learning model for time series prediction using Long Short-Term Memory (LSTM) networks.

## Table of Contents

- [Time Series Prediction with LSTM](#time-series-prediction-with-lstm)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Testing](#testing)
  - [Results](#results)
    - [Files Generated](#files-generated)
    - [Interpretation](#interpretation)
    - [Visualizations](#visualizations)

## Introduction

This project focuses on predicting time series data, specifically temperature, using LSTM neural networks. The code includes data preprocessing, model definition, training, and testing steps.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- tqdm

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Data

The dataset used in this project contains time series data. The main feature for prediction is temperature, and additional features can be easily configured in the code. The dataset is loaded and processed using the `data_factory` module.

## Model Architecture

The implemented model architecture consists of an encoder-decoder LSTM-based Seq2Seq model. The `Encoder`, `Decoder`, `Seq2Seq`, and `lstm` components are defined in the `LSTM.py` file. You can customize the architecture by adjusting the hyperparameters in the model classes.

```python
# Example of customizing the model architecture
encoder = Encoder(input_dim=len(columns), hidden_dim=48, num_layers=4)
decoder = Decoder(output_dim=len(columns), hidden_dim=128, num_layers=4)
model = Seq2Seq(encoder, decoder, device=device)
net = lstm(input_dim=len(columns), hidden_dim=24, num_layers=2, output_dim=len(columns))
```

## Training

To train the model, execute the `train` function in the `exp_main.py` script. Hyperparameters such as batch size, learning rate, and optimizer type are configured in the `args` dictionary.

```bash
python exp_main.py
```

## Testing

To assess the model's performance, utilize the `test` function in the `exp_main.py` script. This function employs a pre-trained model to make predictions on the test dataset, providing evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

```bash
python exp_main.py
```

## Results

The model evaluation results, including key performance metrics and visualizations, are stored in the `./results/` directory after running the testing process. This section outlines the files generated during testing and how to interpret them.

### Files Generated

1. **metrics.npy**: This file contains key performance metrics calculated during the testing phase. Metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Mean Squared Percentage Error (MSPE).

2. **pred.npy**: The predictions made by the model on the test dataset are stored in this file.

3. **true.npy**: This file includes the actual values from the test dataset.

### Interpretation

- **MAE (Mean Absolute Error)**: The average absolute difference between the predicted and actual values. Lower values indicate better performance.

- **MSE (Mean Squared Error)**: The average of the squared differences between predicted and actual values. Lower values signify better accuracy.

- **RMSE (Root Mean Squared Error)**: The square root of MSE, providing a measure of the model's prediction error.

- **MAPE (Mean Absolute Percentage Error)**: The average percentage difference between predicted and actual values.

- **MSPE (Mean Squared Percentage Error)**: Similar to MAPE but uses squared differences.

### Visualizations

In the `./results/` directory, visualizations such as prediction vs. ground truth plots are saved. These visualizations help to qualitatively assess the model's performance.

Feel free to explore these results to understand how well the model is capturing patterns and making predictions on the test dataset.

