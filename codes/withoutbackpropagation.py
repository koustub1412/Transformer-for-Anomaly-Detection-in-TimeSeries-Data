
import sys
import numpy as np
import pandas as pd
import os
import json

# Ensure the file path is passed from the backend (as an argument)
if len(sys.argv) < 2:
    print("Error: No file path provided.")
    sys.exit(1)

# Get the file path from the command-line argument
input_file = sys.argv[1]

# Validate file existence
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    sys.exit(1)

# Load and process the data
# data = pd.read_csv(input_file)
data = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')


data = data.dropna()

selected_columns = ['Open', 'High', 'Low', 'Close']

class TransformerAnomalyDetection:
    def __init__(self, input_dim, embed_dim, latent_dim, num_epochs=10):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.encoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.encoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_Q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_K = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.decoder_W_V = np.random.randn(embed_dim, embed_dim) * np.sqrt(2 / embed_dim)
        self.reconstruction_W = np.random.randn(latent_dim, input_dim) * np.sqrt(2 / input_dim)

    def preprocess_data(self, data, columns):
        self.mean = data[columns].mean()
        self.std = data[columns].std()
        data[columns] = (data[columns] - self.mean) / self.std
        return data

    def positional_encoding(self, sequence_length, embed_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pos_encoding = np.zeros((sequence_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def encode(self, X):
        Q = X @ self.encoder_W_Q
        K = X @ self.encoder_W_K
        V = X @ self.encoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ V, attention_scores

    def decode(self, encoded_X):
        Q = encoded_X @ self.decoder_W_Q
        K = encoded_X @ self.decoder_W_K
        V = encoded_X @ self.decoder_W_V
        attention_scores = Q @ K.T / np.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)
        decoded_X = attention_weights @ V
        projected_decoded_X = decoded_X @ np.random.randn(self.embed_dim, self.latent_dim)
        return projected_decoded_X

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def reconstruct(self, decoded_X):
        reconstructed_X = decoded_X @ self.reconstruction_W.T
        return reconstructed_X

    def calculate_loss(self, X, reconstructed_X):
        return np.mean((X[:self.latent_dim] - reconstructed_X) ** 2)

    def min_max_scale(self, losses):
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        return (losses - min_loss) / (max_loss - min_loss)

    def association_discrepancy(self, attention_scores):
        avg_attention = np.mean(attention_scores, axis=0)
        discrepancy = np.abs(attention_scores - avg_attention)
        return np.mean(discrepancy, axis=-1)

    def detect_anomalies(self, X, threshold_multiplier=2):
        losses = []
        attention_scores_all = []
        association_discrepancies = []

        for i in range(X.shape[0]):
            encoded_X, attention_scores = self.encode(X[i:i+1])
            decoded_X = self.decode(encoded_X)
            reconstructed_X = self.reconstruct(decoded_X)
            loss = self.calculate_loss(X[i], reconstructed_X)
            losses.append(loss)

            association_discrepancy_value = self.association_discrepancy(attention_scores)
            association_discrepancies.append(np.mean(association_discrepancy_value))

            attention_scores_all.append(attention_scores)

        scaled_losses = self.min_max_scale(np.array(losses))
        mean_loss = np.mean(scaled_losses)
        std_loss = np.std(scaled_losses)
        threshold = mean_loss + threshold_multiplier * std_loss

        combined_scores = scaled_losses + np.array(association_discrepancies)
        anomalies = np.where(combined_scores > threshold)[0]

        return anomalies, scaled_losses.tolist(), threshold, combined_scores.tolist()

# Load and preprocess data
model = TransformerAnomalyDetection(input_dim=4, embed_dim=8, latent_dim=4, num_epochs=10)
data = model.preprocess_data(data, selected_columns)

# Dynamically calculate sequence length
X = data[selected_columns].values
sequence_length = X.shape[0]

# Project X to match embed_dim
projection_matrix = np.random.randn(X.shape[1], model.embed_dim)
X_projected = X @ projection_matrix

# Add positional encoding
positional_enc = model.positional_encoding(sequence_length, embed_dim=model.embed_dim)
X_embedded = X_projected + positional_enc

# Detect anomalies
anomalies, losses, threshold, combined_scores = model.detect_anomalies(X_embedded)

# Return anomalies and other relevant data as JSON
output = {
    'anomalies': anomalies.tolist(),
    'losses': losses,
    'threshold': threshold,
    'combined_scores': combined_scores
}

print(json.dumps(output))  # Output anomalies and data as JSON
