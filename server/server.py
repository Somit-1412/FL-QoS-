# server.py

# --- Import necessary libraries ---
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import threading # To run aggregation logic without blocking the API

# --- 1. Initialize the FastAPI application ---
app = FastAPI(
    title="FL-QoS Server",
    description="Manages clients and orchestrates federated learning rounds with QoS-based selection and proper aggregation."
)

# --- 2. Centralized Server State Management ---
class ServerState:
    def __init__(self):
        self.client_database: Dict[str, Dict] = {}
        self.global_model = self.create_cifar10_cnn_model()
        self.current_round = 0
        self.current_round_updates: List[Dict] = []
        self.selected_clients_for_round: set = set()
        self.lock = threading.Lock()
        print("✅ Server State initialized. Global CIFAR-10 model created.")

    def create_cifar10_cnn_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

server_state = ServerState()

# --- 3. Define the structure of the data our API expects ---
class ClientStatus(BaseModel):
    client_id: str
    qos_score: float

class ModelUpdate(BaseModel):
    client_id: str
    round_id: int
    weight_updates: List[list]
    training_loss: float

# --- 4. Helper functions for model serialization ---
def weights_to_json(weights: List[np.ndarray]) -> List[List[float]]:
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: List[List[float]]) -> List[np.ndarray]:
    return [np.array(w) for w in json_weights]

# --- 5. Core FL-QoS Logic: Selection, Allocation, and Aggregation ---
def select_clients_and_create_tasks(num_clients_to_select: int, epsilon: float = 0.2) -> Dict:
    with server_state.lock:
        if len(server_state.client_database) < num_clients_to_select:
            return {}

        available_clients = list(server_state.client_database.items())
        
        if random.random() < epsilon:
            print(f"🔬 EXPLORATION: Selecting {num_clients_to_select} clients randomly.")
            selected_client_tuples = random.sample(available_clients, num_clients_to_select)
            selected_client_ids = [client_id for client_id, data in selected_client_tuples]
        else:
            print(f"🎯 EXPLOITATION: Selecting top {num_clients_to_select} clients by utility score.")
            client_scores = []
            for client_id, data in available_clients:
                system_utility = data.get('qos_score', 0.1)
                statistical_utility = data.get('training_loss', 10.0)
                combined_score = system_utility * statistical_utility
                client_scores.append((client_id, combined_score))
            
            sorted_clients = sorted(client_scores, key=lambda item: item[1], reverse=True)
            selected_client_ids = [client_id for client_id, score in sorted_clients[:num_clients_to_select]]

        print(f"Selected clients for round {server_state.current_round + 1}: {selected_client_ids}")
        server_state.selected_clients_for_round = set(selected_client_ids)
        server_state.current_round_updates = []
        
        tasks = {}
        for client_id in selected_client_ids:
            qos_score = server_state.client_database[client_id]['qos_score']
            
            if qos_score > 0.8: epochs = 5
            elif qos_score > 0.5: epochs = 3
            else: epochs = 1
            
            tasks[client_id] = {
                "round_id": server_state.current_round + 1,
                "model_weights": weights_to_json(server_state.global_model.get_weights()),
                "epochs": epochs
            }
            print(f"  > Assigned task to {client_id}: {epochs} epochs")
        
        return tasks

def aggregate_updates_federated_averaging():
    with server_state.lock:
        if not server_state.current_round_updates:
            print("No updates to aggregate.")
            return

        print(f"\n--- Aggregating updates for round {server_state.current_round} ---")
        
        avg_update = [np.zeros_like(w) for w in server_state.global_model.get_weights()]
        num_updates = len(server_state.current_round_updates)

        for update_data in server_state.current_round_updates:
            client_updates = json_to_weights(update_data['weight_updates'])
            for i, layer_update in enumerate(client_updates):
                avg_update[i] += layer_update

        avg_update = [layer_sum / num_updates for layer_sum in avg_update]

        current_weights = server_state.global_model.get_weights()
        new_weights = [current_weights[i] + avg_update[i] for i in range(len(current_weights))]
        server_state.global_model.set_weights(new_weights)

        print(f"✅ Global model updated with averaged weights from {num_updates} clients.")
        print("--- Round Complete ---\n")

# --- 6. Define the API Endpoints ---
@app.post("/register")
def register_client(status: ClientStatus):
    with server_state.lock:
        server_state.client_database[status.client_id] = {
            "qos_score": status.qos_score,
            "training_loss": 10.0
        }
    return {"message": f"Client {status.client_id} registered successfully."}

@app.post("/start-round")
def start_training_round(num_clients: int, epsilon: Optional[float] = 0.2):
    print(f"\n--- Received request to start a new round for {num_clients} clients ---")
    tasks = select_clients_and_create_tasks(num_clients, epsilon)
    if not tasks:
        raise HTTPException(status_code=400, detail="Not enough registered clients to start a round.")
    
    with server_state.lock:
        server_state.current_round += 1
    
    print(f"Round {server_state.current_round} started. Tasks assigned.")
    return {
        "message": f"Round {server_state.current_round} started.",
        "round_id": server_state.current_round,
        "tasks": tasks
    }

@app.post("/submit-update")
def submit_update(update: ModelUpdate):
    with server_state.lock:
        if update.round_id != server_state.current_round:
            raise HTTPException(status_code=400, detail=f"Update from wrong round. Server is on {server_state.current_round}.")
        if update.client_id not in server_state.selected_clients_for_round:
            raise HTTPException(status_code=403, detail="Client was not selected for the current round.")
        
        server_state.current_round_updates.append(update.dict())
        
        if update.client_id in server_state.client_database:
            server_state.client_database[update.client_id]['training_loss'] = update.training_loss
        
        received_count = len(server_state.current_round_updates)
        expected_count = len(server_state.selected_clients_for_round)
        
        print(f"Received update from {update.client_id} for round {server_state.current_round}. Loss: {update.training_loss:.4f}. ({received_count}/{expected_count})")

        if received_count == expected_count:
            # --- THIS IS THE FIX ---
            # Instead of calling the function directly, run it in a background thread.
            # This frees up the server to respond immediately.
            aggregation_thread = threading.Thread(target=aggregate_updates_federated_averaging)
            aggregation_thread.start()
            return {"status": "Update received and aggregation triggered."}
        else:
            return {"status": f"Update received and stored. Waiting for {expected_count - received_count} more clients."}

@app.get("/get-global-model")
def get_global_model():
    with server_state.lock:
        return {"weights": weights_to_json(server_state.global_model.get_weights())}

# --- 7. Main Execution ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)