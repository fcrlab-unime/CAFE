import os
import time
import csv
from ultralytics import YOLO
from dotenv import load_dotenv

# ================== LOAD .ENV ==================
load_dotenv()

# ================== CONFIG (from environment) ==================
DATA = os.environ.get("DATA", "dataset.yaml")
MODEL = os.environ.get("MODEL", "model.pt")
EPOCHS = int(os.environ.get("EPOCHS", "20"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "320"))
BATCH = int(os.environ.get("BATCH", "4"))

DEVICE_RAW = os.environ.get("DEVICE", "cpu").lower()
DEVICE = "0" if DEVICE_RAW == "gpu" else DEVICE_RAW

SEED = int(os.environ.get("SEED", "42"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1").lower() in ("1", "true", "yes")
EVAL_MODE = os.environ.get("EVAL_MODE", "always").lower()  # "always" | "last"
# ===============================================================

OUTPUT_CSV = f"centralized_{MODEL.replace('.pt','')}_ep{EPOCHS}_b{BATCH}_{DEVICE}.csv"

model = YOLO(MODEL)

# Start timer
start = time.time()

results = model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    seed=SEED,
    deterministic=DETERMINISTIC,
    workers=0,
    val=(EVAL_MODE == "always"),
)

if EVAL_MODE == "last":
    print("\n[INFO] Running final evaluation only...")
    results = model.val(data=DATA, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE)

# End timer
elapsed = time.time() - start
minutes = elapsed / 60
print(f"\nTraining completed in {elapsed:.2f} seconds ({minutes:.2f} minutes)")

# Prepare CSV rows
rows = []
metrics = results.results_dict

rows.append({
    "model": MODEL,
    "epochs": EPOCHS,
    "imgsz": IMG_SIZE,
    "batch": BATCH,
    "device": DEVICE,
    "seed": SEED,
    "deterministic": DETERMINISTIC,
    "eval_mode": EVAL_MODE,
    "time_sec": round(elapsed, 2),
    "time_min": round(minutes, 2),
    "class": "all",
    "precision": round(metrics.get("metrics/precision(B)", 0), 4),
    "recall": round(metrics.get("metrics/recall(B)", 0), 4),
    "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 4),
    "mAP50-95": round(metrics.get("metrics/mAP50-95(B)", 0), 4),
})

for class_id, class_name in results.names.items():
    p, r, map50, map5095 = results.class_result(class_id)
    rows.append({
        "model": MODEL,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "eval_mode": EVAL_MODE,
        "time_sec": round(elapsed, 2),
        "time_min": round(minutes, 2),
        "class": class_name,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "mAP50": round(map50, 4),
        "mAP50-95": round(map5095, 4),
    })

# Save CSV
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Results saved in {OUTPUT_CSV}")
