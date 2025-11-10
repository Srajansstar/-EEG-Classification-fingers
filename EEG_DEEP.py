#!/usr/bin/env python

"""
EEG classification script using ML and Deep Learning models.

This script loads EEG data from a .mat file, preprocesses it using MNE,
extracts features, and then applies:
1. A suite of classical ML models (RF, SVM, XGBoost, etc.).
2. RNN, GRU, and LSTM models on extracted PSD features.
3. A hybrid CNN-LSTM model on the raw EEG data, comparing various optimizers.
"""

import mne
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from mne.decoding import Scaler
from mne.time_frequency import psd_multitaper

# --- Constants ---
RANDOM_STATE = 77
FILE_PATH = r"C:\Studies\Assignment\EEG Assignment 1\5F-SubjectA-160408-5St-SGLHand-HFREQ.mat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)

# =============================================================================
# Data Preprocessing and Setup
# =============================================================================

def load_and_preprocess_data(file_path):
    """Loads and preprocesses EEG data from the .mat file into an MNE Epochs object."""
    print("Loading and preprocessing data...")
    data = scipy.io.loadmat(file_path)
    nested_data = data['o']

    eeg_data = nested_data['data'][0, 0]
    samples, channels = eeg_data.shape
    nS = 1000  # Samples per trial

    n_trials = samples // nS
    
    #  Corrected slicing to trim samples, not channels
    eeg_data_trimmed = eeg_data[:n_trials * nS, :]

    #  Corrected reshape and transpose for (n_epochs, n_channels, n_times)
    X_raw = eeg_data_trimmed.reshape(n_trials, nS, channels).transpose(0, 2, 1)

    Y_marker = nested_data['marker'][0, 0]
    Y_trimmed = Y_marker[:n_trials * nS]
    Y_trials = Y_trimmed.reshape(n_trials, nS)

    # Get the most frequent label for each trial
    y_raw = np.array([np.bincount(y_row.astype(int)).argmax() for y_row in Y_trials])

    ch_names = [str(item[0]) for sublist in nested_data['chnames'][0][0] for item in sublist]
    sfreq = nested_data['sampFreq'][0, 0][0, 0]
    n_channels = len(ch_names)

    info = mne.create_info(ch_names=ch_names,
                           ch_types=['eeg'] * n_channels,
                           sfreq=sfreq)
    info.set_montage('standard_1020', on_missing='ignore')

    event_id = {
        'No raise': 0,
        'thumb': 1,
        'Index Finger': 2,
        'Middle Finger': 3,
        'Ring Finger': 4,
        'pinkie finger': 5,
        'Session break': 90,
        'Experiment end': 91,
        'Initial relaxation': 99,
    }

    # Create MNE events array
    events = np.column_stack((np.arange(len(y_raw)) * nS,
                              np.zeros(len(y_raw), dtype=int),
                              y_raw))
    
    #  Use X_raw directly (it's already n_epochs, n_channels, n_times)
    epochs = mne.EpochsArray(X_raw, info, events, tmin=0, event_id=event_id, verbose=False)

    print(epochs.info)

    # --- MNE Plotting ---
    epochs.plot_sensors(show_names=True)
    epochs.average().plot_image(titles="Average EEG Power")
    epochs.average().plot(titles="Average EEG Signal")
    epochs.plot_psd(fmin=2., fmax=40., average=True, spatial_colors=True)
    if 'Index Finger' in epochs.event_id:
        epochs['Index Finger'].average().plot(titles="Index Finger Average")

    return epochs, X_raw, y_raw

# =============================================================================
# Feature Extraction 
# =============================================================================

def extract_features(X, sfreq):
    """Extracts PSD features from EEG epochs."""
    n_epochs, n_channels, _ = X.shape
    features = []
    print(f"Extracting PSD features from {n_epochs} epochs...")
    for epoch in X:
        psd, freqs = psd_multitaper(epoch, sfreq=sfreq, fmin=0.1, fmax=50, verbose=False)
        features.append(psd.flatten())
    return np.array(features)

# =============================================================================
# Part 1: Classical Machine Learning Models
# =============================================================================

def run_ml_models(epochs, y_raw):
    """Trains and evaluates a suite of classical ML models on extracted features."""
    print("\n--- Running Classical ML Models ---")
    X = epochs.get_data(copy=False)
    sfreq = epochs.info['sfreq']

    scaler = Scaler(epochs.info)
    X_scaled = scaler.fit_transform(X)
    X_features = extract_features(X_scaled, sfreq)

    # Filter out non-task labels
    valid_labels = np.array([0, 1, 2, 3, 4, 5])
    mask = np.isin(y_raw, valid_labels)
    X_features_filtered = X_features[mask]
    y_raw_filtered = y_raw[mask]

    # Resample using BorderlineSMOTE
    smote = BorderlineSMOTE(k_neighbors=max(1, min(pd.Series(y_raw_filtered).value_counts()) - 1), 
                            random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_features_filtered, y_raw_filtered)
    print(f"Data shape after SMOTE: {X_resampled.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5), 
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)
    }

    results = {}
    plt.figure(figsize=(18, 16))
    
    # Handle XGBoost label encoding
    le = LabelEncoder()
    y_train_xgb = le.fit_transform(y_train)

    print("Training and evaluating models...")
    for i, (name, model) in enumerate(models.items(), 1):
        
        # --- Train ---
        if name == "XGBoost":
            model.fit(X_train, y_train_xgb)
        else:
            model.fit(X_train, y_train)

        # --- Predict ---
        if name == "XGBoost":
            y_pred_transformed = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_transformed)
        else:
            y_pred = model.predict(X_test)
            
        # --- Evaluate ---
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # --- Plot Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(3, 3, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=valid_labels, yticklabels=valid_labels)
        plt.title(f"{name} Confusion Matrix", pad=20)
        plt.xlabel("Predicted", labelpad=15)
        plt.ylabel("Actual", labelpad=15)

    plt.suptitle("ML Model Confusion Matrices", fontsize=24, y=1.02)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    return X_resampled, y_resampled

# =============================================================================
# Part 2: Deep Learning - RNN/GRU/LSTM on Features
# =============================================================================

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        out, _ = self.rnn(x)
        out = self.bn(out[:, -1, :]) 
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.bn(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out)

def train_rnn_model(model, X_train, y_train, X_test, y_test, epochs=30, learning_rate=0.05):
    """Training loop for RNN-based models on feature data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    accuracy_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = 100 * (predicted == y_test).sum().item() / y_test.size(0)

        accuracy_list.append(accuracy)
        if (epoch + 1) % 100 == 0:
            print(f"Model: {model.__class__.__name__}, Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%")

    return accuracy_list

def run_rnn_models(X_resampled, y_resampled, device):
    """Initializes, trains, and plots RNN, GRU, and LSTM models."""
    print("\n--- Running RNN/GRU/LSTM Models on Features ---")
    
    #  Use the resampled feature data
    X_tensor_features = torch.tensor(X_resampled, dtype=torch.float32)
    y_tensor_labels = torch.tensor(y_resampled, dtype=torch.long)

    #  Split the correct tensors
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor_features, y_tensor_labels, test_size=0.2, random_state=RANDOM_STATE
    )

    # Move data to device
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    input_size = X_resampled.shape[1]
    hidden_size = 128
    output_size = len(torch.unique(y_tensor_labels))
    num_epochs = 3000 

    # --- RNN ---
    rnn_model = RNNModel(input_size, hidden_size, output_size).to(device)
    rnn_accuracy = train_rnn_model(rnn_model, X_train, y_train, X_test, y_test, num_epochs)
    print("Trained RNN")

    # --- GRU ---
    gru_model = GRUModel(input_size, hidden_size, output_size).to(device)
    gru_accuracy = train_rnn_model(gru_model, X_train, y_train, X_test, y_test, num_epochs)
    print("Trained GRU")

    # --- LSTM ---
    lstm_model = LSTMModel(input_size, hidden_size, output_size).to(device)
    lstm_accuracy = train_rnn_model(lstm_model, X_train, y_train, X_test, y_test, num_epochs)
    print("Trained LSTM")

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), rnn_accuracy, label="RNN")
    plt.plot(range(1, num_epochs + 1), gru_accuracy, label="GRU")
    plt.plot(range(1, num_epochs + 1), lstm_accuracy, label="LSTM")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Over Epochs (on PSD Features)")
    plt.legend()


# =============================================================================
# Part 3: Deep Learning - CNN-LSTM on Raw Data
# =============================================================================

class EEG(nn.Module):
    """Hybrid CNN-LSTM model for raw EEG classification."""
    def __init__(self, num_classes=6, input_channels=22, seq_length=1000):
        super(EEG, self).__init__()
        
        # 1D Conv layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        # Calculate resulting sequence length after 3 pools
        lstm_input_size = 128
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True) #Using Bi diectional LSTM

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        #for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1) 
        
        x, _ = self.lstm(x)
        
        x = x[:, -1, :]
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_cnn_model(model, dataloader, optimizer_name, optimizer_fn, device, num_epochs=50):
    """Training loop for the CNN-LSTM model to compare optimizers."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())
    losses = []
    
    print(f"Training with {optimizer_name}...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        #  Log loss for plots
        if (epoch + 1) % 5 == 0:
            losses.append(avg_loss)
            print(f"{optimizer_name} - Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
    return losses

def run_cnn_lstm_comparison(X_raw, y_raw, device):
    """Compares different optimizers on the CNN-LSTM model."""
    print("\n--- Running CNN-LSTM Optimizer Comparison on Raw Data ---")
    
    #  Filter data to only include valid task labels
    valid_labels = np.array([0, 1, 2, 3, 4, 5])
    mask = np.isin(y_raw, valid_labels)
    X_raw_filtered = X_raw[mask]
    y_raw_filtered = y_raw[mask]
    
    print(f"Using {len(y_raw_filtered)} epochs for CNN-LSTM training.")
    
    X_tensor = torch.tensor(X_raw_filtered, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw_filtered, dtype=torch.long)

    dataset = EEGDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model parameters
    n_classes = len(valid_labels)
    n_channels = X_raw_filtered.shape[1]

    optimizers = {
        "SGD+Momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        "NAG": lambda params: optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True),
        "Adam": lambda params: optim.Adam(params, lr=0.01), 
        "RMSprop": lambda params: optim.RMSprop(params, lr=0.001),
        "Adagrad": lambda params: optim.Adagrad(params, lr=0.01),
        "Adadelta": lambda params: optim.Adadelta(params, lr=1.0)
    }

    loss_results = {}
    for name, opt_fn in optimizers.items():
        # Re-initialize model for each optimizer
        model = EEG(num_classes=n_classes, input_channels=n_channels).to(device)
        loss_results[name] = train_cnn_model(model, dataloader, name, opt_fn, device)

    # --- Plot Loss Curves ---
    plt.figure(figsize=(12, 7))
    epoch_ticks = np.linspace(5, 50, 10, dtype=int)
    
    for name, losses in loss_results.items():
        plt.plot(epoch_ticks, losses, label=name, marker="o", linestyle="-", linewidth=2)
    
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Optimizer Comparison on EEG-CNN-LSTM Model", fontsize=16, fontweight="bold")
    plt.xticks(epoch_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    # --- Plot Loss Table ---
    df = pd.DataFrame(loss_results).round(3)
    df.index = [f"Epoch {i}" for i in epoch_ticks]

    fig, ax = plt.subplots(figsize=(12, 5)) # Adjusted size
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc="center", loc="center", colWidths=[0.15] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) # Scale table
    
    plt.title("Optimizer Loss Comparison Table", fontsize=16, y=0.8)
    plt.tight_layout()


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Main function to run the entire EEG analysis pipeline."""

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.backends.cudnn.version()}")
    
    #  Load and plot MNE data
    epochs, X_raw, y_raw = load_and_preprocess_data(FILE_PATH)
    
    #  Run classical ML models on PSD features
    X_resampled, y_resampled = run_ml_models(epochs, y_raw)
    
    #  Run RNNs on PSD features
    run_rnn_models(X_resampled, y_resampled, DEVICE)
    
    #  Run CNN-LSTM on raw data
    run_cnn_lstm_comparison(X_raw, y_raw, DEVICE)
    
    print("\nAnalysis complete")
    plt.show()

if __name__ == "__main__":
    main()