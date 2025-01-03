import json
import sys
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess data
if len(sys.argv) < 2:
    print("Error: No file path provided.")
    sys.exit(1)

input_file = sys.argv[1]

# Validate file existence
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    sys.exit(1)

# Load and process the data
df = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

X = df.values

# Normalize data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-8
X_train = (X - X_mean) / X_std
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

class AnomalyAttention:
    def __init__(self, N, d_model):
        self.N = N
        self.d_model = d_model
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.Ws = np.random.randn(d_model, 1)

    def forward(self, x):
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)
        sigma = np.clip(np.dot(x, self.Ws), 1e-3, 1.0)
        P = self.prior_association(sigma)
        S = self.softmax(np.dot(Q, K.T) / np.sqrt(self.d_model))
        Z = np.dot(S, V)
        return Z, P, S

    def prior_association(self, sigma):
        N = sigma.shape[0]
        p = np.arange(N).reshape(1, -1)
        diff = np.abs(p - p.T)
        gaussian = np.exp(-0.5 * (diff / sigma).T ** 2) / np.sqrt(2 * np.pi * sigma)
        return gaussian / (np.sum(gaussian, axis=1, keepdims=True) + 1e-8)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

class AnomalyTransformer:
    def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
        self.N = N
        self.d_model = d_model
        self.lambda_ = lambda_
        self.hidden_dim = hidden_dim
        self.attention_layer = AnomalyAttention(N, d_model)

        # Correct weight initialization for hidden weights and output weights
        self.hidden_weights = np.random.randn(d_model, hidden_dim)  # (d_model, hidden_dim)
        self.output_weights = np.random.randn(hidden_dim, d_model)  # (hidden_dim, d_model)

    def forward(self, data):
        # Compute hidden layer activations
        hidden_activations = np.dot(data, self.hidden_weights)  # (N, hidden_dim)
        hidden_activations = np.maximum(0, hidden_activations)  # ReLU activation

        # Output layer
        output_activations = np.dot(hidden_activations, self.output_weights)  # (N, d_model)

        # Get attention-related outputs
        Z, P, S = self.attention_layer.forward(data)

        return hidden_activations, output_activations, P, S

    def loss_function(self, x_hat, x, P, S):
        frob_norm = np.linalg.norm(x_hat - x)
        
        # Association discrepancy: difference between prior association P and series association S
        assoc_discrepancy = np.sum(np.abs(P - S), axis=1)  # Summing over the difference between P and S
        
        # KL divergence term
        kl_div = np.sum(P * (np.log(P + 1e-8) - np.log(S + 1e-8)))
        
        return frob_norm + self.lambda_ * kl_div + np.sum(assoc_discrepancy)

    def anomaly_score(self, data):
        _, x_hat, P, S = self.forward(data)
        reconstruction_error = np.linalg.norm(data - x_hat, axis=1)
        
        # Association discrepancy is now the difference between P and S
        assoc_discrepancy = np.sum(np.abs(P - S), axis=1)  # Absolute difference
        
        return reconstruction_error + assoc_discrepancy

# Minimax strategy: alternating between minimization and maximization phases
def train(model, data, lr, epochs):
    losses = []
    for epoch in range(epochs):
        phase = "minimize" if epoch % 2 == 0 else "maximize"  # Alternate phases
        # Forward pass
        hidden_activations, x_hat, P, S = model.forward(data)

        # Calculate loss based on the current phase
        loss = model.loss_function(x_hat, data, P, S)
        if not np.isfinite(loss):
            print(f"Epoch {epoch + 1}/{epochs}, Loss encountered NaN or Inf. Skipping...")
            continue

        # Compute gradients
        grad_output = (x_hat - data) / data.shape[0]  # Gradient w.r.t. output

        # Calculate gradient w.r.t hidden layer activations
        grad_hidden = np.dot(grad_output, model.output_weights.T) * (hidden_activations > 0)  # ReLU derivative

        # Update weights
        model.output_weights -= lr * np.dot(hidden_activations.T, grad_output) / data.shape[0]
        model.hidden_weights -= lr * np.dot(data.T, grad_hidden) / data.shape[0]

        losses.append(loss)
        print(f"Epoch {epoch + 1}/{epochs}, Phase: {phase.capitalize()}, Loss: {loss}")

    return np.mean(losses)

# Initialize the model
N, d_model = X_train.shape
hidden_dim = 64
model = AnomalyTransformer(N, d_model, hidden_dim)
learning_rate = 0.01
epochs = 10

# Train the model
average_loss = train(model, X_train, learning_rate, epochs)

# Calculate anomaly scores
anomaly_scores = model.anomaly_score(X_train)

# Apply minimax strategy to refine the anomaly detection threshold
minimax_scores = np.maximum(anomaly_scores, np.percentile(anomaly_scores, 10))

# Simulate labels
np.random.seed(42)
true_labels = np.zeros(N, dtype=int)
anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
true_labels[anomaly_indices] = 1

# Threshold and normalize scores
threshold = np.percentile(minimax_scores, 90)
predictions = (minimax_scores > threshold).astype(int)

# Get the indices of anomalies (where prediction is 1)


# Evaluate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)
roc_auc = roc_auc_score(true_labels, minimax_scores)

# Create the result JSON
'''result = {
    "anomalies": anomaly_indices_detected,
    "losses": [float(average_loss)] * epochs,
    "threshold": float(threshold),
    "combined_scores": [float(score) for score in minimax_scores]
}'''

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")



# Output the result as JSON
#output_json = json.dumps(result, indent=4)
#print(output_json)
# Get the indices of anomalies (where prediction is 1)
anomaly_indices_detected = np.where(predictions == 1)[0].tolist()

# Print the indices
print(f"Indices of detected anomaly points: {anomaly_indices_detected}")