import os
import numpy as np
from src.preprocessing import load_data, preprocess
from src.model import build_model
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "data/raw/Home_loan_data.csv"
MODEL_PATH = "models/loan_model.keras"

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("Building model...")
    model = build_model(X_train.shape[1])

    print("Training model...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Accuracy: {acc:.4f}")

    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print("Training complete!")


if __name__ == "__main__":
    train()