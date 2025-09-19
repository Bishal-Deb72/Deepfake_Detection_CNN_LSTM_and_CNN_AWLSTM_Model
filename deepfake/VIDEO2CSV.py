import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# Load face embedding model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Face crop & transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def extract_embeddings(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    data = []
    count = 0
    face_found = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 3 == 0:  # every 3rd frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = resnet(face_tensor).squeeze().cpu().numpy()
                    data.append(list(emb) + [label])
                    face_found += 1
                break  # use only the first detected face

        count += 1

    cap.release()
    print(f"[INFO] {os.path.basename(video_path)} — Faces detected: {face_found}")
    return data

def collect_real_from_root(root_dir):
    real_data = []
    for file in os.listdir(root_dir):
        if file.endswith('.mov') and 'original' in file:
            path = os.path.join(root_dir, file)
            print(f"[REAL] {file}")
            real_data += extract_embeddings(path, label=0)
    return real_data

def collect_from_directory(base_dir, label):
    all_data = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('-video-fram1.avi'):
                path = os.path.join(root, file)
                print(f"[FAKE] {file}")
                all_data += extract_embeddings(path, label)
    return all_data

# CHANGE THIS to your dataset path
BASE_DIR = "DeepfakeTIMIT/DeepfakeTIMIT"

REAL_DIR = BASE_DIR
LOW_DIR = os.path.join(BASE_DIR, "lower_quality")
HIGH_DIR = os.path.join(BASE_DIR, "higher_quality")

# Run pipeline
print("[INFO] Starting real video processing...")
real_data = collect_real_from_root(REAL_DIR)

print("[INFO] Starting fake video processing (lower_quality)...")
fake_data1 = collect_from_directory(LOW_DIR, label=1)

print("[INFO] Starting fake video processing (higher_quality)...")
fake_data2 = collect_from_directory(HIGH_DIR, label=1)

# Save CSV
final_data = real_data + fake_data1 + fake_data2
df = pd.DataFrame(final_data)
df.to_csv("deepfaketimit.csv", index=False)
print(f"[SUCCESS] CSV saved as deepfaketimit.csv — Total samples: {len(df)}")
