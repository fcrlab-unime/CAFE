# CAFE — Centralized and Federated Experiments

**CAFE** (Centralized And Federated Experiments) is a unified framework designed to run, compare, and analyze **centralized** and **federated** training experiments using **YOLOv8** for object detection tasks.

The repository provides reproducible pipelines, environment-driven configuration, and evaluation workflows for benchmarking **edge–cloud**, **distributed**, and **federated learning** scenarios.

---

## 📁 Repository Structure

```
CAFE/
├── centralized_training/
│   ├── centralized_training.py
│   ├── example.env
│   ├── requirements.txt
│   └── README.md
│
├── federated_training/
│   ├── server.py
│   ├── client.py
│   ├── server.env
│   ├── client.env
│   ├── requirements.txt
│   └── README.md
│
└── README.md   <-- (this file)
```

---

## 🎯 Project Goals

CAFE provides a controlled environment for comparing:

- **Centralized vs Federated** training performance  
- **CPU, GPU, and Jetson-edge devices**  
- **Different federated learning strategies** (FedAvg, FedAvgM, FedAdam, FedYogi)  
- **Training time, communication cost, accuracy, and scalability**  

The project enables reproducible experiments through:

- `.env`-based configuration  
- deterministic training options  
- unified logging and CSV output  
- consistent dataset management for both approaches  

---

## ⚙️ 1. Centralized Training

The folder **`centralized_training/`** contains:

- `centralized_training.py` — YOLOv8 centralized training script  
- `.env` — configuration file  
- automatic CSV metrics output  
- training time profiling  

### ▶️ Run centralized training

```bash
cd centralized_training
python centralized_training.py
```

---

## ⚙️ 2. Federated Training

The folder **`federated_training/`** contains:

- `server.py` — Flower federated learning server  
- `client.py` — YOLOv8 federated client  
- `server.env` — server configuration  
- `client.env` — client configuration  
- Jetson-safe operations  
- final aggregated model + CSV logs  

### ▶️ Start server

```bash
cd federated_training
python server.py
```

### ▶️ Start a client

```bash
python client.py
```

To launch multiple clients, duplicate `client.env` and modify:

```
CLIENT_ID=client2
DATA=dataset_client2/fasdd.yaml
```

---

## 📊 Results & Outputs

Both centralized and federated modules generate:

- YOLOv8 model checkpoints (`.pt`)  
- per-class and global metrics  
- CSV logs  
- timing information  
- (Federated) final aggregated model  
- (Federated) round-by-round time measurements  

---


