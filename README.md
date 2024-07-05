# LSTM Deep Neural Network for Travel Time Prediction

This repository contains a Python project demonstrating how to build and train a deep learning model for time series prediction using LSTM and deep neural networks in PyTorch. The model is designed to predict the `currenttraveltime` based on several traffic-related features.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Saving](#model-saving)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HappytheCoder/LSTM-Deep-Neural-Network-for-Time-Series-Prediction.git
   ```

2. **Install the required libraries:**
   ```bash
   pip install numpy pandas torch scikit-learn matplotlib
   ```

## Usage

1. **Prepare your dataset:**
   Place your dataset (`data_extended2.csv`) in the project directory.

2. **Run the script:**
   ```bash
   python main.py
   ```

## Data Preparation

The data is read from a CSV file and then processed to create sequences of features and corresponding targets for training and testing.

- **Selected Features**:
  - `currentspeed`
  - `currentdensity`
  - `currenttraveltime`
  - `currentflow`

- **Sequence Parameters**:
  - `sequence_length`: 20
  - `target_size`: 10
  - `step_size`: 1

The `create_sequences` function creates sequences and targets for the model.

## Model Architecture

The model consists of an LSTM network followed by a series of dense layers.

- **LSTM Configuration**:
  - `input_size`: Number of features (4 in this case)
  - `hidden_size`: 64
  - `num_layers`: 2
  - `dropout_prob`: 0.2

- **Deep Neural Network Configuration**:
  - Hidden layers: [128, 64, 32]
  - Output size: 1

The `LSTMNetwork` class defines the architecture, combining LSTM and a deep neural network.

## Training

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. A learning rate scheduler is used to adjust the learning rate based on validation loss. Gradient clipping is implemented to improve training stability.

- **Training Parameters**:
  - `num_epochs`: 1000
  - `learning_rate`: 0.0001
  - `weight_decay`: 1e-4
  - `patience`: 10 (for early stopping)

## Evaluation

After training, the model's performance is evaluated using the Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R² score.

## Results

The script plots the actual vs. predicted values and the training and validation losses over epochs.

## Model Saving

The trained model, along with its configuration and weights, is saved in various formats for later use.

- **Metadata**: `model_metadata.json`
- **Traced Model**: `traced_model_lstm_dnn.pt`
- **Model Weights**: `model_weights.pth`
- **ONNX Model**: `lstm_travel_time_dnn2.onnx`

## License

This project is licensed under the MIT License.
