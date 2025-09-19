import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt

# -------------------------------
# Model Definition
# -------------------------------
class DeepfakeCNNRNN(nn.Module):
    def __init__(self, input_dim=512, cnn_channels=64, rnn_hidden_size=128, rnn_layers=1):
        super(DeepfakeCNNRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.rnn = nn.LSTM(input_size=input_dim // 4, hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)            # (batch, 1, features)
        x = self.cnn(x)               # (batch, channels, features//4)
        x = x.permute(0, 2, 1)        # (batch, time, channels)
        out, _ = self.rnn(x)          # (batch, time, hidden)
        return self.classifier(out[:, -1, :])

# -------------------------------
# Data Preparation
# -------------------------------
df = pd.read_csv("deepfaketimit_with_videoid.csv")
X = df.drop(columns=["video_id", "512"]).values.astype(np.float32)
y = df["512"].values.astype(np.float32)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

# Train/Val split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# -------------------------------
# Model Training
# -------------------------------
model = DeepfakeCNNRNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

val_probs_final = []
val_targets_final = []

for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_probs, val_preds, val_targets = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_probs.extend(preds.squeeze().numpy())
            val_preds.extend(preds.squeeze().round().numpy())
            val_targets.extend(yb.squeeze().numpy())

    acc = accuracy_score(val_targets, val_preds)
    rmse = np.sqrt(mean_squared_error(val_targets, val_probs))
    r2 = r2_score(val_targets, val_probs)

    print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f} | Val Accuracy: {acc:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Save last val set for ROC
    val_probs_final = val_probs
    val_targets_final = val_targets

# -------------------------------
# Save the Trained Model
# -------------------------------
torch.save(model.state_dict(), "deepfake_cnn_rnn_final.pth")

# -------------------------------
# ROC Curve Plot
# -------------------------------
fpr, tpr, _ = roc_curve(val_targets_final, val_probs_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Proposed CNN–LSTM Deepfake Detection')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
