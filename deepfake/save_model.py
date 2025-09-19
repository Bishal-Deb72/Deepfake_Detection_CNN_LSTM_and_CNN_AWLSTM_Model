import torch
from deepfake_cnn_rnn_train import DeepfakeCNNRNN  #  Use your actual file name

# Initialize model architecture
model = DeepfakeCNNRNN()

# Optionally load trained weights if you already saved them earlier
# model.load_state_dict(torch.load("deepfake_cnn_rnn.pth"))

# Save model weights
torch.save(model.state_dict(), "deepfake_cnn_rnn.pth")
print(" Model saved as deepfake_cnn_rnn.pth")
