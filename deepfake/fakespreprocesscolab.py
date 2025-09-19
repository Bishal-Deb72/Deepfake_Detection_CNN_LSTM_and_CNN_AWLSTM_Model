#  Install Sentence-Transformers (only needed once)
#pip install -q sentence-transformers

#  File upload
from google.colab import files
uploaded = files.upload()

import pandas as pd
from sentence_transformers import SentenceTransformer

#  Load uploaded CSV
#  Load uploaded CSV with fallback encoding
csv_file = list(uploaded.keys())[0]
try:
    df = pd.read_csv(csv_file)
except UnicodeDecodeError:
    print("[WARNING] UTF-8 decoding failed. Retrying with ISO-8859-1...")
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')  # fallback


#  Preview
print("Sample rows from the uploaded file:")
print(df.head())

#  Ensure required columns exist
if 'article_content' not in df.columns or 'labels' not in df.columns:
    raise ValueError("Your CSV must contain 'article_content' and 'labels' columns!")

#  Load BERT model for text embeddings
print("\n[INFO] Generating 512-dimensional embeddings using Sentence-BERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, ~384 dim

#  Compute embeddings
embeddings = model.encode(df['article_content'].tolist(), show_progress_bar=True)

#  Convert to DataFrame and attach labels
emb_df = pd.DataFrame(embeddings)
emb_df['label'] = df['labels']

#  Save as new CSV
output_filename = "fake_news_embeddings.csv"
emb_df.to_csv(output_filename, index=False)

#  Offer download
print(f" File '{output_filename}' is ready for download.")
files.download(output_filename)