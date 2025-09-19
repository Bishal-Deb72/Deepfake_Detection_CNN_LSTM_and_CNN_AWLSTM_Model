import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# -------------------------------
# Custom AWLSTM Cell
# -------------------------------
class AWLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AWLSTMCell, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.project = nn.Linear(input_size, hidden_size)  # Ensure this matches input from CNN

    def forward(self, x, hx=None):
        lstm_out, (hn, cn) = self.lstm(x, hx)  # [batch, time, hidden]
        x_trimmed = x[:, -lstm_out.size(1):, :]  # Align sequence length

        x_proj = self.project(x_trimmed.reshape(-1, x_trimmed.shape[2]))  # [batch*time, hidden]
        x_proj = x_proj.view(x.size(0), -1, lstm_out.size(2))              # [batch, time, hidden]

        output = self.alpha * lstm_out + (1 - self.alpha) * x_proj
        return output, (hn, cn)




# -------------------------------
# CNN + AWLSTM Model
# -------------------------------
class DeepfakeCNNRNN(nn.Module):
    def __init__(self, input_dim=512, cnn_channels=64, rnn_hidden_size=128):
        super(DeepfakeCNNRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # -- The correct input_size is cnn_channels (64), not input_dim // 4 (128)
        self.rnn = AWLSTMCell(input_size=cnn_channels, hidden_size=rnn_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])


# -------------------------------
# Load and Preprocess Dataset
# -------------------------------
df = pd.read_csv("deepfaketimit_with_videoid.csv")
X = df.drop(columns=["video_id", "512"]).values.astype(np.float32)
y = df["512"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# -------------------------------
# Train Model
# -------------------------------
model = DeepfakeCNNRNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
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

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {acc:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

# -------------------------------
# Save Trained Model
# -------------------------------
torch.save(model.state_dict(), "deepfake_cnn_awlstm.pth")

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get all predictions and targets from the validation set
model.eval()
all_val_probs = []
all_val_targets = []
with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        all_val_probs.extend(preds.view(-1).cpu().numpy())
        all_val_targets.extend(yb.view(-1).cpu().numpy())

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_val_targets, all_val_probs)
roc_auc = roc_auc_score(all_val_targets, all_val_probs)

# Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='navy', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
