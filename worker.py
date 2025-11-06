# worker.py (Micro-Worker, File-based)

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from typing import List

# --- Helper Functions ---
def json_to_weights(json_weights: List[list]) -> List[np.ndarray]:
    return [np.array(w) for w in json_weights]

def weights_to_json(weights: List[np.ndarray]) -> List[List[float]]:
    return [w.tolist() for w in weights]

# --- Main Worker Logic ---
def train_single_client(client_id, round_id, epochs, global_weights_json, client_data_x, client_data_y):
    print(f"      -> Worker process for {client_id} started...")
    
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"), keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"), keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(), keras.layers.Dropout(0.5), keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    global_weights = json_to_weights(global_weights_json)
    model.set_weights(global_weights)
    history = model.fit(client_data_x, client_data_y, epochs=epochs, batch_size=32, verbose=0)
    
    new_weights = model.get_weights()
    weight_updates = [new_weights[i] - global_weights[i] for i in range(len(new_weights))]
    
    update_payload = {
        "client_id": client_id, "round_id": round_id,
        "weight_updates": weights_to_json(weight_updates), "training_loss": float(history.history['loss'][-1])
    }
    requests.post("http://localhost:8000/submit-update", json=update_payload)
    print(f"      -> Worker for {client_id} finished and submitted.")

if __name__ == "__main__":
    # --- Load all data once ---
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    
    # --- NEW: Read task details from a temporary file ---
    task_filepath = sys.argv[1]
    with open(task_filepath, 'r') as f:
        task_details = json.load(f)
    
    client_id = task_details['client_id']
    client_idx = int(client_id.split('_')[1])
    
    data_shards_x = np.array_split(x_train, 100)
    client_x = data_shards_x[client_idx]
    
    data_shards_y = np.array_split(y_train, 100)
    client_y = data_shards_y[client_idx]
    
    train_single_client(
        client_id, task_details['round_id'], task_details['epochs'],
        task_details['model_weights'], client_x, client_y
    )