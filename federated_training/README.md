# 📘 Federated YOLOv8 Training

This module implements a **Federated Learning (FL)** training pipeline for YOLOv8 using **Flower (FLWR)**.  
It supports multiple strategies (FedAvg, FedAvgM, FedAdam, FedYogi), Jetson-safe operations, per-round timing, model aggregation, and final global evaluation.

This system is designed for benchmarking **cloud–edge** and **multi-client distributed** training scenarios.

---

## 📂 Project Structure

```
.
├── server.py
├── server.env
├── client.py
├── client.env
├── requirements.txt
└── README_federated.md
```

---

## ⚙️ 1. Environment Setup

### 1.1 Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

_On Windows:_

```bash
python -m venv venv
venv\Scripts\activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **GPU users:**  
> Install the correct PyTorch wheel manually according to your CUDA version:  
> https://pytorch.org/

---

## 📑 2. Configuration

Two separate `.env` files are used:

- `server.env` → Flower server + global YOLO model
- `client.env` → Federated client configuration

---

### 2.1 Server Configuration — `server.env`

Example:

```ini
# ===== Federated Server Config =====
NUM_ROUNDS=5
MIN_CLIENTS=2
SERVER_ADDRESS=0.0.0.0:9090

# YOLO model initialization
MODEL_INIT=model.pt

# Output directory and filenames
OUTPUT_DIR=server_output
SAVE_FILENAME=server_final_model.pt

# Dataset for global validation
DATA_SERVER=dataset_server/dataset.yaml

# Inference / validation params
IMG_SIZE=320
BATCH=4
DEVICE=0

# Federated strategy: FedAvg | FedAvgM | FedAdam | FedYogi
STRATEGY=FedAvg

# FedAdam / FedYogi hyperparameters
ETA=0.0005
BETA1=0.9
BETA2=0.99
TAU=1e-9

# FedAvgM hyperparameters
SERVER_LR=0.0
SERVER_MOM=0.9
```

---

### 2.2 Client Configuration — `client.env`

Example:

```ini
# ===== Federated Client Config =====

# Local dataset (YOLO format)
DATA=dataset_client/dataset.yaml

# Initial YOLO model (same architecture as MODEL_INIT on server)
MODEL=model.pt

# Local training hyperparameters
EPOCHS=1
BATCH=4
IMG_SIZE=320

# Device: cpu | gpu
DEVICE=cpu

# Client identity
CLIENT_ID=client1
SEED=42

# FedBN toggle (1 = ON, 0 = OFF)
FEDBN=1

# Initial learning rate
LR0=0.0015

# Flower server address
SERVER_ADDRESS=0.0.0.0:9090
```

Each client instance can have its own `client.env` file (different dataset path, ID, device, etc.).

---

## ▶️ 3. Running the Federated Learning System

### 3.1 Start the Flower Server

From the `federated/` folder:

```bash
python server.py
```

The server will:

1. Load settings from `server.env`
2. Initialize the YOLOv8 model from `MODEL_INIT`
3. Start the Flower server on `SERVER_ADDRESS`
4. Run `NUM_ROUNDS` of federated training
5. Track per-round training time
6. Aggregate client updates according to the selected strategy
7. Save the final global model to `OUTPUT_DIR/SAVE_FILENAME`
8. Run centralized validation on the final global model
9. Append timing and evaluation metrics to `federated_learning.csv`

---

### 3.2 Start one or more Clients

From the `federated/` folder (or on edge devices):

```bash
python client.py
```

Each client will:

1. Load its configuration from `client.env`
2. Connect to the Flower server at `SERVER_ADDRESS`
3. Receive the current global YOLO model
4. Train locally for `EPOCHS` epochs on its own dataset
5. Send updated parameters + number of local images back to the server
6. Save the final local model as `CLIENT_ID_final.pt` during the evaluation phase

To launch multiple clients on the same machine, you can:

- Duplicate `client.env` as `client1.env`, `client2.env`, etc.
- Launch multiple terminals, each one with different `CLIENT_ID`, `DATA`, and possibly `DEVICE`

---

## 🔧 4. Federated Strategies Supported

The system supports the following Flower strategies:

| Strategy   | Description                            |
|-----------|----------------------------------------|
| FedAvg    | Standard federated averaging           |
| FedAvgM   | FedAvg with server-side momentum       |
| FedAdam   | Adaptive server optimizer (Adam-like)  |
| FedYogi   | Adaptive server optimizer (Yogi)       |

Choose the strategy in `server.env`:

```ini
STRATEGY=FedAvg
```

### 4.1 Strategy Hyperparameters

- **FedAdam / FedYogi**
  - `ETA` – server learning rate  
  - `BETA1`, `BETA2` – momentum/decay coefficients  
  - `TAU` – small constant for numerical stability  

- **FedAvgM**
  - `SERVER_LR` – server learning rate  
  - `SERVER_MOM` – server momentum  

All these hyperparameters are read automatically from the environment.

---

## 📊 5. Outputs

### 5.1 Global CSV Log

The server writes a log file:

```text
server_output/federated_learning.csv
```

The file contains:

- A header row with strategy, number of rounds, and configuration
- One row per round with:
  - `round`
  - `round_time_s`
- Final section with:
  - Global validation metrics (precision, recall, mAP50, mAP50–95) for:
    - class = `all`
    - each individual class

This CSV can be used for:

- Comparing FL strategies  
- Measuring per-round overhead  
- Plotting training time and performance curves  

---

### 5.2 Final Global Model

After the FL rounds, the server saves:

```text
server_output/server_final_model.pt
```

This file contains the aggregated YOLOv8 model parameters and can be used for:

- Centralized inference
- Further fine-tuning
- Deployment to edge devices

---

### 5.3 Local Client Models

Each client exports its final local model as:

```text
CLIENT_ID_final.pt
```

These models represent each client's last state at the end of its participation in the FL process.

---

## 🧠 6. Jetson / Edge-Safe Behaviour

The client code includes a few patches to improve robustness on Jetson and heterogeneous edge devices:

- **Safe NMS**: wraps `torchvision.ops.nms` to fallback to CPU if CUDA NMS is not available
- **Disabled YOLO validation** during local training:
  - avoids issues with NMS on GPU-less environments
  - reduces overhead on lightweight devices
- **AMP checks disabled**:
  - `check_amp` is overridden to return `False`

Together, these modifications make the federated client more robust on:

- NVIDIA Jetson platforms
- CPU-only edge devices
- Mixed environments where CUDA is partially available

---