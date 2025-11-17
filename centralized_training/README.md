# 📘 Centralized YOLOv8 Training

This module implements a **centralized training pipeline** for YOLOv8 models, fully configurable through a `.env` file and capable of producing a final **CSV** containing global and per-class metrics.  
It is ideal as a baseline for comparisons with **cloud**, **edge**, or **federated learning** scenarios.

---

## 📂 Project Structure

```
.
├── centralized_training.py
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ 1. Environment Setup

### 1.1 Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If using GPU CUDA, install the correct PyTorch version manually from:  
> https://pytorch.org/

---

## 📑 2. Configuration via `.env`

The `.env` file lets you modify all training parameters **without editing the code**.

Example:

```ini
# Dataset YOLO (yaml)
DATA=dataset.yaml

# Base model YOLO (pretrained)
MODEL=yolov8n.pt

# Training hyperparameters
EPOCHS=20
IMG_SIZE=320
BATCH=4

# Device: cpu | gpu
DEVICE=cpu

# Reproducibility
SEED=42
DETERMINISTIC=1   # 1=true, 0=false

# Evaluation mode:
#   always = validate every epoch
#   last   = validate only after final epoch
EVAL_MODE=always
```

---

## ▶️ 3. Running the Training

After configuring `.env`, run:

```bash
python centralized_training.py
```

The script will:
1. Load the YOLOv8 model  
2. Load dataset + parameters from `.env`  
3. Run centralized training  
4. Perform evaluation (per epoch or only final)  
5. Measure total training time  
6. Produce a CSV file with metrics  

---

## 📊 4. Output

The script automatically generates:

```
centralized_{MODEL}_ep{EPOCHS}_b{BATCH}_{DEVICE}.csv
```

### 4.1 CSV Contents

The CSV contains:

- **Global metrics** (class = all)  
- **Per-class metrics**

Columns include:

- model  
- epochs  
- image size  
- batch size  
- device  
- seed / deterministic mode  
- training time (sec/min)  
- precision  
- recall  
- mAP50  
- mAP50-95  

Example header:

```
model,epochs,imgsz,batch,device,seed,deterministic,eval_mode,time_sec,time_min,class,precision,recall,mAP50,mAP50-95
```

---

## 🧠 5. Why This Script?

This centralized baseline is useful for evaluating:

- edge vs cloud training  
- federated learning vs centralized learning  
- CPU vs GPU performance  
- different datasets  
- different YOLO model sizes  

It ensures reproducibility through:

- fixed seed  
- deterministic settings  
- zero workers  
- `.env` configuration  

---