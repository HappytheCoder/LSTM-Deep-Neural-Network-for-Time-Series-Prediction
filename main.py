# Importing necessary libraries

# JSON library to handle JSON data
import json

# Numpy for numerical operations
import numpy as np

# Pandas for data manipulation and analysis
import pandas as pd

# PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn for preprocessing, model selection, and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import platform
import seaborn as sns

# Matplotlib for plotting and visualization
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)



# Set device for training
if platform.system() == 'Darwin':  # Check if the system is MacOS
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
else:  # For other platforms like Windows or Linux
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sequences(data, sequence_length, target_size, step_size, features, target_feature):
    """
    Function to create sequences and corresponding targets for each group (ID, Day).

    Parameters:
    - data: pandas DataFrame containing the data.
    - sequence_length: Number of timesteps in each input sequence.
    - target_size: Number of timesteps ahead to predict.
    - step_size: Number of timesteps to move forward when creating sequences.
    - features: List of features to include in the sequences.
    - target_feature: Feature to predict.

    Returns:
    - sequences: Numpy array of input sequences.
    - targets: Numpy array of target values.
    """
    sequences = []
    targets = []
    grouped = data.groupby(['ID', 'Day'])

    for _, group in grouped:
        group = group.sort_values(by='timestep')
        values = group[features].values
        for i in range(0, len(values) - sequence_length - target_size + 1, step_size):
            target_index = i + sequence_length + target_size - 1
            if target_index < len(values):
                sequences.append(values[i:i + sequence_length])
                targets.append(group.iloc[target_index][target_feature])

    return np.array(sequences), np.array(targets)


# Load data
data = pd.read_csv("data_extended_new_demand_network.csv")

# Define features
selected_features = ['currentspeed', 'currentdensity', 'currenttraveltime', 'currentflow']

# Prepare data using selected features
sequence_length = 30
target_size = 10
step_size = 1
X, y = create_sequences(data, sequence_length, target_size, step_size, selected_features, 'currenttraveltime')

# plotting y values frequency
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=10, kde=True)
plt.title('Distribution of Current Travel Time')
plt.xlabel('Current Travel Time')
plt.ylabel('Frequency')
plt.show()

# Standardize sequences
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data into training and testing sets using sklearn
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Define Deep Neural Network class
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the deep neural network.

        Parameters:
        - input_size: Number of input features.
        - hidden_sizes: List of sizes for hidden layers.
        - output_size: Number of output features.
        """
        super(DeepNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor.

        Returns:
        - x: Output tensor.
        """
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


# Define LSTM model
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, deep_nn_hidden_sizes, output_size):
        """
        Initialize the LSTM network with a deep neural network on top.

        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of features in the hidden state.
        - num_layers: Number of recurrent layers.
        - deep_nn_hidden_sizes: List of sizes for hidden layers in the deep neural network.
        - output_size: Number of output features.
        """
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.deep_nn = DeepNN(hidden_size, deep_nn_hidden_sizes, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor.
        """
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        output, _ = self.lstm(x, (h_0, c_0))
        output = self.deep_nn(output[:, -1, :])
        return output


# Model configuration
input_size = len(selected_features)
hidden_size = 32
num_layers = 2
deep_nn_hidden_sizes = [64,32]  # Example: Three hidden layers with sizes 128, 64, and 32
output_size = 1

# Initialize the model
model = LSTMNetwork(input_size, hidden_size, num_layers, deep_nn_hidden_sizes, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
weight_decay = 1e-4 # Adjusted weight decay parameter
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=weight_decay)  # Increased learning rate slightly


# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop parameters
num_epochs = 2000
train_losses = []
val_losses = []
patience = 10
best_val_loss = float('inf')
patience_counter = 0

# Move data to the appropriate device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()

    # Implement gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    optimizer.step()
    train_losses.append(loss.item())

    # Validation phase
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        val_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {test_loss.item():.4f}')

    # # Step the scheduler based on the validation loss
    scheduler.step(test_loss)
    #
    # Early stopping
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# Calculate RMSE, MSE, and R^2
rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy(), test_outputs.cpu().numpy()))
mse = mean_squared_error(y_test.cpu().numpy(), test_outputs.cpu().numpy())
r2 = r2_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())
print(f'RMSE: {rmse:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R^2: {r2:.2f}')

# Plot actual vs. predicted
plt.figure(figsize=(14, 7))
plt.plot(y_test.cpu().numpy(), label='Actual')
plt.plot(test_outputs.cpu().numpy(), label='Predicted')
plt.title('Actual vs. Predicted')
plt.xlabel('Time')
plt.ylabel('Current Travel Time')
plt.legend()
plt.show()

# Plot training and validation losses
plt.figure(figsize=(14, 7))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model metadata as a JSON file
metadata = {
    'sequence_length': sequence_length,
    'num_features': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'deep_nn_hidden_sizes': deep_nn_hidden_sizes,
    'output_size': output_size
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)

# Trace the model and save it
example_input = X_test[20].unsqueeze(0)  # Example input tensor with fixed sequence length
print("Example Input:", example_input)
print("Example Prediction:", model(example_input).item())
print("Actual Value:", y_test[20].item())
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save('traced_model_lstm_dnn.pt')
torch.save(model.state_dict(), 'model_weights.pth')

# Export the model to ONNX
torch.onnx.export(model, example_input, "lstm_travel_time_dnn11.onnx")
