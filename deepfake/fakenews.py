# main.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, GlobalMaxPooling1D, RepeatVector, LSTM, Dense,
    Dropout, Multiply, Permute, Lambda, Flatten, Activation
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(int)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # for Conv1D/LSTM
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_awlstm_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN Layer
    x = Conv1D(128, kernel_size=3, activation='relu')(inputs)
    x = GlobalMaxPooling1D()(x)
    x = RepeatVector(10)(x)  # Expand for LSTM input

    # LSTM Layer
    lstm_out = LSTM(64, return_sequences=True)(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)

    # Weighted sum of LSTM outputs with attention
    weighted_sum = Multiply()([lstm_out, attention])
    sentence_representation = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_sum)

    # Output layer
    x = Dense(64, activation='relu')(sentence_representation)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=384, epochs=20):
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val)
    )
    return history


def evaluate_model(model, X_val, y_val):
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))


def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_file = 'fake_news_embeddings.csv'  # Make sure this file exists in the same directory
    X_train, X_val, y_train, y_val = load_data(csv_file)
    model = build_awlstm_model((384, 1))  # 384 embedding dimensions
    history = train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_val, y_val)
    plot_history(history)
