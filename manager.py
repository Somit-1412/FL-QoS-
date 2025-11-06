# manager.py (Final Version)

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import subprocess
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from typing import List
import tempfile

# --- THE DEFINITIVE FIX ---
# Find all available GPUs and enable memory growth on them.
# This prevents the manager process from greedily allocating all VRAM.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Enabled memory growth on {len(gpus)} GPU(s).")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# --- END OF FIX ---


# --- Configuration ---
SERVER_URL = "http://localhost:8000"
TOTAL_CLIENTS = 100
CLIENTS_PER_ROUND = 25
TOTAL_ROUNDS = 50

# --- Helper Functions ---
def json_to_weights(json_weights: List[list]) -> List[np.ndarray]:
    return [np.array(w) for w in json_weights]

# --- 1. Initial Setup and Registration ---
print("🚀 MANAGER: Starting simulation.")
print("   - Registering all clients with the server...")

for i in range(TOTAL_CLIENTS):
    client_id = f"client_{i:04d}"
    qos = np.random.uniform(0.1, 1.0)
    try:
        requests.post(f"{SERVER_URL}/register", json={"client_id": client_id, "qos_score": qos})
    except requests.exceptions.ConnectionError:
        print(f"\n❌ FATAL: Could not connect to the server at {SERVER_URL}. Please ensure the server is running.")
        exit()
print(f"   - All {TOTAL_CLIENTS} clients registered.")

# --- 2. The Main Orchestration Loop ---
global_accuracy_history = []
_, (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

evaluation_model = keras.Sequential([keras.Input(shape=(32, 32, 3)), keras.layers.Conv2D(32, (3, 3), activation="relu"), keras.layers.MaxPooling2D((2, 2)), keras.layers.Conv2D(64, (3, 3), activation="relu"), keras.layers.MaxPooling2D((2, 2)), keras.layers.Flatten(), keras.layers.Dropout(0.5), keras.layers.Dense(10, activation="softmax")])
evaluation_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\n📊 MANAGER: Evaluating initial model (Round 0)...")
response = requests.get(f"{SERVER_URL}/get-global-model")
global_weights = json_to_weights(response.json()['weights'])
evaluation_model.set_weights(global_weights)
_, accuracy = evaluation_model.evaluate(x_test, y_test, verbose=0)
global_accuracy_history.append(accuracy)
print(f"   - ✅ Baseline Accuracy (Round 0): {accuracy*100:.2f}%")

for r in range(1, TOTAL_ROUNDS + 1):
    print(f"\n--- MANAGER: Starting Round {r}/{TOTAL_ROUNDS} ---")

    response = requests.post(f"{SERVER_URL}/start-round?num_clients={CLIENTS_PER_ROUND}")
    tasks = response.json()['tasks']
    print(f"   - Server selected {len(tasks)} clients. Spawning micro-workers...")

    for client_id, task_data in tasks.items():
        task_data['client_id'] = client_id
        
        with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.json') as temp_f:
            json.dump(task_data, temp_f)
            temp_f.flush()
            
            subprocess.run([sys.executable, "worker.py", temp_f.name], check=True)

    print(f"--- MANAGER: All workers for Round {r} finished. Evaluating global model... ---")
    
    response = requests.get(f"{SERVER_URL}/get-global-model")
    global_weights = json_to_weights(response.json()['weights'])
    evaluation_model.set_weights(global_weights)
    _, accuracy = evaluation_model.evaluate(x_test, y_test, verbose=0)
    global_accuracy_history.append(accuracy)
    print(f"   - ✅ Round {r} complete. Global Model Accuracy: {accuracy*100:.2f}%")

# --- 3. Reporting and Visualization ---
print("\n🏁 MANAGER: Simulation finished. Generating results...")
plt.figure(figsize=(10, 6))
plt.plot(range(0, TOTAL_ROUNDS + 1), global_accuracy_history, marker='o', linestyle='-')
plt.title('FL-QoS: Global Model Accuracy Over Communication Rounds')
plt.xlabel('Communication Round')
plt.ylabel('Global Accuracy')
plt.grid(True)
plt.xticks(range(0, TOTAL_ROUNDS + 1, 5))
plt.ylim(bottom=0, top=max(global_accuracy_history) * 1.1 if global_accuracy_history else 1.0)
plt.savefig("fl_qos_accuracy.png")

print("   - Accuracy plot saved to 'fl_qos_accuracy.png'")
print("\n--- ✅ Evaluation Complete ---")
for r, acc in enumerate(global_accuracy_history):
    print(f"Round {r}: Accuracy = {acc*100:.2f}%")