# =====================================================
# YOLOv8 Federated Server (with .env support)
# =====================================================

import os
import time
import csv
import torch
import flwr as fl
from ultralytics import YOLO
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from dotenv import load_dotenv

# Load server.env
load_dotenv("server.env")

# =====================================================
# ENV VARS
# =====================================================
NUM_ROUNDS     = int(os.environ.get("NUM_ROUNDS", "2"))
MIN_CLIENTS    = int(os.environ.get("MIN_CLIENTS", "2"))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "0.0.0.0:9090")

MODEL_INIT     = os.environ.get("MODEL_INIT", "model.pt")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", "server_output")
SAVE_FILENAME  = os.environ.get("SAVE_FILENAME", "server_final_model.pt")

DATA_SERVER    = os.environ.get("DATA_SERVER", "dataset_server/dataset.yaml")
IMG_SIZE       = int(os.environ.get("IMG_SIZE", "320"))
BATCH          = int(os.environ.get("BATCH", "4"))
DEVICE         = os.environ.get("DEVICE", "0")

STRATEGY_NAME  = os.environ.get("STRATEGY", "FedAvg")

# FedAdam / FedYogi
ETA   = float(os.environ.get("ETA", "5e-4"))
BETA1 = float(os.environ.get("BETA1", "0.9"))
BETA2 = float(os.environ.get("BETA2", "0.99"))
TAU   = float(os.environ.get("TAU", "1e-9"))

# FedAvgM
SERVER_LR  = float(os.environ.get("SERVER_LR", "0.0"))
SERVER_MOM = float(os.environ.get("SERVER_MOM", "0.9"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_DIR, SAVE_FILENAME)
CSV_PATH  = os.path.join(OUTPUT_DIR, "federated_learning.csv")

print(f"[SERVER] Starting on {SERVER_ADDRESS}")
print(f"[SERVER] Config: rounds={NUM_ROUNDS}, min_clients={MIN_CLIENTS}, strategy={STRATEGY_NAME}")

_training_start = None
_round_start_time = None
round_times = {}
final_parameters = None

# =====================================================
# CSV UTILS
# =====================================================
def _init_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"strategy={STRATEGY_NAME}, num_rounds={NUM_ROUNDS}, "
                         f"min_clients={MIN_CLIENTS}, img_size={IMG_SIZE}, batch={BATCH}, device={DEVICE}"])
        writer.writerow(["round", "round_time_s"])

def _write_round_time(round_id, round_time):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_id, round(round_time, 2)])

# =====================================================
# STRATEGY WRAPPER
# =====================================================
class TimedStrategy:
    def __init__(self, base_strategy):
        self.base = base_strategy

    def __getattr__(self, name):
        return getattr(self.base, name)

    def configure_fit(self, server_round, parameters, client_manager):
        global _round_start_time, _training_start
        res = self.base.configure_fit(server_round, parameters, client_manager)
        _round_start_time = time.time()
        if _training_start is None:
            _training_start = _round_start_time
        print(f"[SERVER] Round {server_round}: strategy sampled {len(res)} clients")
        return res

    def aggregate_fit(self, server_round, results, failures):
        global final_parameters
        aggregated = self.base.aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            final_parameters = aggregated[0]
        if _round_start_time:
            round_time = time.time() - _round_start_time
            round_times[server_round] = round_time
            print(f"[SERVER] Round {server_round} duration: {round_time:.2f}s")
            _write_round_time(server_round, round_time)
        return aggregated

# =====================================================
# STRATEGY BUILDER
# =====================================================
def get_strategy():
    def init_params_from_model():
        model = YOLO(MODEL_INIT)
        nds = []
        for _, p in model.model.named_parameters():
            nds.append(p.detach().cpu().numpy())
        for _, b in model.model.named_buffers():
            nds.append(b.detach().cpu().numpy())
        return ndarrays_to_parameters(nds)

    if STRATEGY_NAME == "FedAvg":
        base = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=MIN_CLIENTS,
            min_available_clients=MIN_CLIENTS,
        )

    elif STRATEGY_NAME == "FedAvgM":
        base = fl.server.strategy.FedAvgM(
            fraction_fit=1.0,
            min_fit_clients=MIN_CLIENTS,
            min_available_clients=MIN_CLIENTS,
            server_learning_rate=SERVER_LR,
            server_momentum=SERVER_MOM,
            initial_parameters=init_params_from_model(),
        )

    elif STRATEGY_NAME in ("FedAdam", "FedYogi"):
        init_params = init_params_from_model()

        if STRATEGY_NAME == "FedAdam":
            base = fl.server.strategy.FedAdam(
                fraction_fit=1.0,
                min_fit_clients=MIN_CLIENTS,
                min_available_clients=MIN_CLIENTS,
                eta=ETA, beta_1=BETA1, beta_2=BETA2, tau=TAU,
                initial_parameters=init_params,
            )
        else:
            base = fl.server.strategy.FedYogi(
                fraction_fit=1.0,
                min_fit_clients=MIN_CLIENTS,
                min_available_clients=MIN_CLIENTS,
                eta=ETA, beta_1=BETA1, beta_2=BETA2, tau=TAU,
                initial_parameters=init_params,
            )
    else:
        raise ValueError(f"Unknown strategy {STRATEGY_NAME}")

    return TimedStrategy(base)

# =====================================================
# SERVER START
# =====================================================
_init_csv()
strategy = get_strategy()

fl.server.start_server(
    server_address=SERVER_ADDRESS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

# =====================================================
# SAVE FINAL MODEL
# =====================================================
if final_parameters is not None:
    print(f"[SERVER] Saving final global model to {SAVE_PATH}")
    ndarrays = parameters_to_ndarrays(final_parameters)

    model = YOLO(MODEL_INIT)
    i = 0
    for _, param in model.model.named_parameters():
        param.data = torch.tensor(ndarrays[i], dtype=param.dtype)
        i += 1
    for _, buffer in model.model.named_buffers():
        if i < len(ndarrays):
            buffer.data = torch.tensor(ndarrays[i], dtype=buffer.dtype)
            i += 1

    model.save(SAVE_PATH)

# =====================================================
# VALIDATION
# =====================================================
print("[SERVER] Running validation...")
model = YOLO(SAVE_PATH)
results = model.val(
    data=DATA_SERVER,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    verbose=True,
)

mp, mr, map50, map95 = results.box.mean_results()

with open(CSV_PATH, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# Final validation metrics"])
    w.writerow(["class", "precision", "recall", "mAP50", "mAP50-95"])
    w.writerow(["all", float(mp), float(mr), float(map50), float(map95)])

print(f"[SERVER] Final metrics appended to {CSV_PATH}")
