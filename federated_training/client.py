# =====================================================
# YOLOv8 Federated Client (with .env support)
# =====================================================

import os
import yaml
import random
import numpy as np
import torch
import flwr as fl
import torchvision.ops
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import ultralytics.utils.checks as checks
from dotenv import load_dotenv


# Load client.env
load_dotenv("client.env")


# =====================================================
# SAFE NMS FOR JETSON (unchanged)
# =====================================================
old_nms = torchvision.ops.nms
def safe_nms(boxes, scores, iou_thres):
    try:
        return old_nms(boxes, scores, iou_thres)
    except NotImplementedError:
        print("[WARN] torchvision::nms CUDA not available, fallback CPU.")
        return old_nms(boxes.cpu(), scores.cpu(), iou_thres).to(boxes.device)
torchvision.ops.nms = safe_nms

def dummy_validate(self):
    print("[INFO] Validation disabled (Jetson-safe).")
    return {}, 0.0

def dummy_final_eval(self):
    print("[INFO] Final evaluation disabled.")
    self.metrics = {}
    self.fitness = 0.0
    return

def dummy_check_amp(model):
    print("[INFO] AMP disabled.")
    return False

BaseTrainer.validate = dummy_validate
BaseTrainer.final_eval = dummy_final_eval
checks.check_amp = dummy_check_amp


# =====================================================
# ENV VARS
# =====================================================
DATA = os.environ.get("DATA", "dataset_client/dataset.yaml")
MODEL = os.environ.get("MODEL", "model.pt")

EPOCHS = int(os.environ.get("EPOCHS", "1"))
BATCH = int(os.environ.get("BATCH", "4"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "320"))

DEVICE_RAW = os.environ.get("DEVICE", "cpu").lower()
DEVICE = "cuda:0" if DEVICE_RAW in ("0", "gpu", "cuda") and torch.cuda.is_available() else "cpu"

CLIENT_ID = os.environ.get("CLIENT_ID", "clientX")
SEED = int(os.environ.get("SEED", "42"))
FEDBN = os.environ.get("FEDBN", "1").lower() in ("1", "true", "yes")
LR0 = float(os.environ.get("LR0", "0.0015"))

FINAL_MODEL_PATH = f"{CLIENT_ID}_final.pt"


# =====================================================
# SEEDING
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================================
# FEDERATED CLIENT LOGIC
# =====================================================
class YOLOClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = YOLO(MODEL)

        for name, p in self.model.model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

        with open(DATA, "r") as f:
            self.cfg = yaml.safe_load(f)

        print(f"[{CLIENT_ID}] FedBN={FEDBN} | Device={DEVICE} | LR0={LR0}")

    def get_parameters(self, config=None):
        params = [p.cpu().numpy() for _, p in self.model.model.named_parameters()]
        if not FEDBN:
            params += [b.cpu().numpy() for _, b in self.model.model.named_buffers()]
        return params

    def set_parameters(self, parameters):
        i = 0
        with torch.inference_mode(False):
            for _, p in self.model.model.named_parameters():
                if i < len(parameters):
                    p.data = torch.tensor(parameters[i], dtype=p.dtype, device=p.device).clone()
                    i += 1
            if not FEDBN:
                for _, b in self.model.model.named_buffers():
                    if i < len(parameters):
                        b.data = torch.tensor(parameters[i], dtype=b.dtype, device=b.device).clone()
                        i += 1

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.validator = None

        print(f"[{CLIENT_ID}] Training start: {EPOCHS} epoch(s)")

        self.model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=0,
            verbose=False,
            val=False,
            patience=0,
            save=False,
            lr0=LR0,
            plots=False,
        )

        return self.get_parameters(), 1, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.save(FINAL_MODEL_PATH)
        print(f"[{CLIENT_ID}] Saved final local model")
        return 0.0, 1, {}



if __name__ == "__main__":
    server_addr = os.environ.get("SERVER_ADDRESS", "0.0.0.0:9090")
    print(f"[{CLIENT_ID}] Connecting to server at {server_addr}")
    fl.client.start_numpy_client(server_address=server_addr, client=YOLOClient())
