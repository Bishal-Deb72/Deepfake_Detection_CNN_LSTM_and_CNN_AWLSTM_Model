import torch
from deepfake_cnn_rnn_train import DeepfakeCNNRNN  # or wherever your model class is

# Recreate the model architecture
model = DeepfakeCNNRNN()

# Load weights
model.load_state_dict(torch.load("deepfake_cnn_rnn.pth"))
model.eval()

print(" Model loaded successfully and ready for inference.")
