
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
X_train = torch.FloatTensor(X)

# Normalize data
X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

# Define the AnomalyAttention module
class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.N = N
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        sigma = torch.clamp(self.Ws(x), min=1e-3, max=1.0)
        P = self.prior_association(sigma)
        S = (Q @ K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        S = torch.softmax(S, dim=-1)
        Z = S @ V
        return Z, P, S

    @staticmethod
    def prior_association(sigma):
        N = sigma.shape[0]
        p = torch.arange(N).unsqueeze(0).repeat(N, 1)
        diff = torch.abs(p - p.T).float()
        gaussian = torch.exp(-0.5 * (diff / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
        return gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)

# Define the AnomalyTransformer module
class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
        super(AnomalyTransformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.lambda_ = lambda_
        self.hidden_dim = hidden_dim
        self.attention_layers = AnomalyAttention(N, d_model)
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        Z, P, S = self.attention_layers(x)
        hidden = torch.relu(self.hidden_layer(Z))
        x_hat = self.output_layer(hidden)
        return x_hat, P, S

    def anomaly_score(self, x):
        """
        Calculate the anomaly score for each input sample.
        """
        x_hat, P, S = self(x)  # Forward pass
        reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
        assoc_discrepancy = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='none'), dim=1)
        return reconstruction_error + assoc_discrepancy

# Define the loss functions for minimization and maximization phases
def minimize_phase_loss(x, x_hat, P, S, lambda_):
    reconstruction_loss = torch.linalg.norm(x - x_hat, ord='fro')
    kl_divergence = torch.sum(F.kl_div(torch.log(P + 1e-8), S.detach(), reduction='batchmean'))
    return reconstruction_loss + lambda_ * kl_divergence

def maximize_phase_loss(x, x_hat, P, S, lambda_):
    reconstruction_loss = torch.linalg.norm(x - x_hat, ord='fro')
    kl_divergence = torch.sum(F.kl_div(torch.log(P.detach() + 1e-8), S, reduction='batchmean'))
    return reconstruction_loss + lambda_ * kl_divergence

# Training function with minimization and maximization phases
def train_minimax(model, data, optimizer, lambda_, epochs):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        x_hat, P, S = model(data)

        # Minimization phase
        minimize_loss = minimize_phase_loss(data, x_hat, P, S, lambda_)
        minimize_loss.backward(retain_graph=True)

        # Maximization phase
        maximize_loss = maximize_phase_loss(data, x_hat, P, S, lambda_)
        maximize_loss.backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += (minimize_loss.item() + maximize_loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Minimize Loss: {minimize_loss.item():.4f}, Maximize Loss: {maximize_loss.item():.4f}",file=sys.stderr)

    return total_loss / epochs  # Return the average loss over epochs

# Initialize the model and optimizer
N, d_model = X_train.shape
hidden_dim = 64
lambda_ = 0.1
model = AnomalyTransformer(N, d_model, hidden_dim, lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10

# Train the model
average_loss = train_minimax(model, X_train, optimizer, lambda_, epochs)

# Calculate anomaly scores
anomaly_scores = model.anomaly_score(X_train).detach().numpy()
anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)

# Apply minimax strategy to refine the anomaly detection threshold
minimax_scores = []
for i, score in enumerate(anomaly_scores):
    minimax_scores.append(max(score, min(anomaly_scores)))

# Simulate labels
np.random.seed(42)
true_labels = np.zeros(N, dtype=int)
anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
true_labels[anomaly_indices] = 1

# Threshold and normalize scores
threshold = np.percentile(minimax_scores, 90)
predictions = (minimax_scores > threshold).astype(int)

# Get the indices of anomalies (where prediction is 1)
anomaly_indices_detected = np.where(predictions == 1)[0].tolist()

# Evaluate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)
roc_auc = roc_auc_score(true_labels, minimax_scores)

# Create the result JSON
result = {
    "anomalies": anomaly_indices_detected,
    "losses": [float(average_loss)] * epochs,
    "threshold": float(threshold),
    "combined_scores": [float(score) for score in minimax_scores]
}
# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")
# output_json = json.dumps(result, indent=4)
# print(output_json)
