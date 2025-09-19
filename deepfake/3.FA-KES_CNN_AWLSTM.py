import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tqdm import tqdm

# -------------------------------
# Auto GPU Detection
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------------
# AWLSTM Cell
# -------------------------------
class AWLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AWLSTMCell, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.project = nn.Linear(input_size, hidden_size)

    def forward(self, x, hx=None):
        lstm_out, (hn, cn) = self.lstm(x, hx)  # (B, T, H)
        x_trimmed = x[:, -lstm_out.size(1):, :]  # Align
        x_proj = self.project(x_trimmed.reshape(-1, x_trimmed.shape[2]))
        x_proj = x_proj.view(x.size(0), -1, lstm_out.size(2))
        output = self.alpha * lstm_out + (1 - self.alpha) * x_proj
        return output, (hn, cn)

# -------------------------------
# CNN + AWLSTM Model
# -------------------------------
class DeepfakeCNNRNN(nn.Module):
    def __init__(self, input_dim=384, cnn_channels=64, rnn_hidden_size=128):
        super(DeepfakeCNNRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.input_proj = nn.Linear(cnn_channels, rnn_hidden_size)
        self.rnn = AWLSTMCell(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)             # (B, 1, 384)
        x = self.cnn(x)                # (B, 64, 96)
        x = x.permute(0, 2, 1)         # (B, 96, 64)
        x = self.input_proj(x)         # (B, 96, 128)
        out, _ = self.rnn(x)           # (B, 96, 128)
        return self.classifier(out[:, -1])  # Last timestep (B, 1)

# -------------------------------
# Load Dataset
# -------------------------------
file_path = "fake_news_embeddings.csv"
assert os.path.exists(file_path), f"{file_path} not found!"
df = pd.read_csv(file_path)

X = df.drop(columns=["label"]).values.astype(np.float32)
y = 1 - df["label"].values.astype(np.float32)  # flip labels

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Tensor Conversion
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

# Dataset Split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, pin_memory=True)

# -------------------------------
# Model, Optimizer, Loss
# -------------------------------
model = DeepfakeCNNRNN(input_dim=384).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_probs, val_preds, val_targets = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_probs.extend(preds.view(-1).cpu().numpy())
            val_preds.extend(preds.view(-1).round().cpu().numpy())
            val_targets.extend(yb.view(-1).cpu().numpy())

    acc = accuracy_score(val_targets, val_preds)
    rmse = np.sqrt(mean_squared_error(val_targets, val_probs))
    r2 = r2_score(val_targets, val_probs)

    print(f"[E{epoch:02}] Loss: {total_loss:.4f} | Acc: {acc:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

# -------------------------------
# Save Final Model
# -------------------------------
torch.save(model.state_dict(), "fakes_cnn_awlstmm.pth")
print("[INFO] Model saved to fakes_cnn_awlstmm.pth")
