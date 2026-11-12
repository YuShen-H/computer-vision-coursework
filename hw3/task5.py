
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    import cv2
except ImportError:
    raise SystemExit("Install OpenCV in your current environment: python -m pip install opencv-python==4.10.0.84")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLIT_FILE = BASE_DIR / "outputs/task2/split_indices.npz"
BASELINE_CHECKPOINT = BASE_DIR / "outputs/task3_cnn/main/cnn_best.pt"
OUTPUT_DIR = BASE_DIR / "outputs/task5"
DEVICE = "auto" 

EPOCHS = 60
BATCH_SIZE = None
LEARNING_RATE = None
MIN_LEARNING_RATE = None
WEIGHT_DECAY = None
DROPOUT_RATE = None
CONV_DROPOUT = None
PATIENCE = None
MIN_DELTA = None
RANDOM_SEED = None


AUGMENT_PROBABILITY = 0.8
ROTATION_MAX_DEGREES = 30.0       
NOISE_SIGMA_MAX = 40.0         
BLUR_SIGMA_MIN = 0.3
BLUR_SIGMA_MAX = 2.0
BRIGHTNESS_MIN = 0.4           
EVALUATION_BATCH_SIZE = 256
EVALUATION_NOISE_SEED = 42        
ROTATIONS = [-30, -20, -10, 10, 20, 30]
NOISE_SIGMAS = [10, 25, 40]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
BRIGHTNESS_FACTORS = [0.8, 0.6, 0.4]
# =======================================================

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class CNN(nn.Module):
    def __init__(self, dropout_rate=0.0, conv_dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(conv_dropout),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(conv_dropout),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 2 * 2, 64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        # Call model.eval() and use torch.no_grad() for inference.
        return torch.softmax(self(x), dim=1)


def read_idx(path, images):
    requested = path
    if not path.is_file():
        path = path.with_suffix("")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {requested} and extracted equivalent.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    offset = 16 if images else 8
    if len(raw) < offset:
        raise ValueError(f"Incomplete IDX file: {path}")
    header = struct.unpack(">IIII" if images else ">II", raw[:offset])
    if header[0] != (2051 if images else 2049):
        raise ValueError(f"Invalid IDX magic number: {path}")
    shape = header[1:]
    if images and shape[1:] != (28, 28):
        raise ValueError(f"Expected 28x28 grayscale images: {path}")
    values = np.frombuffer(raw, dtype=np.uint8, offset=offset)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Invalid IDX payload length: {path}")
    return values.reshape(shape)


def load_test(data_dir):
    images = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", True)
    labels = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", False)
    if images.shape != (10000, 28, 28) or labels.shape != (10000,):
        raise ValueError("Expected all 10,000 official test images and labels.")
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 1000)):
        raise ValueError("Expected 1,000 official test examples per class.")
    return images, labels.astype(np.int64)


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(model):
    # Includes BatchNorm running means/variances and counters, not just weights.
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def conditions():
    cases = [{"id": "clean", "family": "clean", "value": 0.0, "label": "Clean"}]
    for angle in ROTATIONS:
        cases.append({"id": f"rotation_{angle:+d}", "family": "rotation", "value": angle,
                      "label": f"Rotation {angle:+d} deg"})
    for sigma in NOISE_SIGMAS:
        cases.append({"id": f"noise_{sigma}", "family": "noise", "value": sigma,
                      "label": f"Noise sigma={sigma}"})
    for sigma in BLUR_SIGMAS:
        cases.append({"id": f"blur_{sigma:g}", "family": "blur", "value": sigma,
                      "label": f"Blur sigma={sigma:g}"})
    for factor in BRIGHTNESS_FACTORS:
        cases.append({"id": f"brightness_{factor:g}", "family": "brightness", "value": factor,
                      "label": f"Brightness x{factor:g}"})
    return cases


def corrupt(images, case, seed):
    # Always work in float32 on the 0–255 scale. Normalize only for inference.
    source = images.astype(np.float32, copy=True)
    family, value = case["family"], case["value"]
    if family == "clean":
        return source
    if family == "brightness":
        return np.clip(source * value, 0, 255).astype(np.float32)
    if family == "noise":
        # Same underlying standard-normal draws across sigma levels and models.
        noise = np.random.default_rng(seed).standard_normal(source.shape, dtype=np.float32)
        return np.clip(source + noise * value, 0, 255).astype(np.float32)
    result = np.empty_like(source)
    if family == "rotation":
        matrix = cv2.getRotationMatrix2D((13.5, 13.5), value, 1.0)
        for i, image in enumerate(source):
            result[i] = cv2.warpAffine(image, matrix, (28, 28), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif family == "blur":
        for i, image in enumerate(source):
            result[i] = cv2.GaussianBlur(image, (0, 0), sigmaX=value, sigmaY=value,
                                        borderType=cv2.BORDER_REFLECT_101)
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return np.clip(result, 0, 255)


def evaluate(model, pixels, labels, batch_size, device, tag):
    model.eval()
    predictions = np.empty(len(labels), dtype=np.int64)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    batches = range(0, len(labels), batch_size)
    with torch.inference_mode(), tqdm(batches, total=(len(labels) + batch_size - 1) // batch_size,
            desc=tag, unit="batch", dynamic_ncols=True, ascii=True, leave=False,
            mininterval=0.5, file=sys.stdout) as progress:
        for start in progress:
            end = min(start + batch_size, len(labels))
            x = torch.from_numpy(np.ascontiguousarray(pixels[start:end, None] / np.float32(255.0))).to(device)
            y = torch.from_numpy(labels[start:end]).to(device)
            logits = model(x)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Non-finite predictions during {tag}")
            total_loss += criterion(logits, y).item()
            predictions[start:end] = logits.argmax(1).cpu().numpy()
    return total_loss / len(labels), predictions


def metrics(labels, predictions):
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    tp = np.diag(cm).astype(np.float64)
    supports, totals = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, totals, out=np.zeros(10), where=totals != 0)
    recall = np.divide(tp, supports, out=np.zeros(10), where=supports != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10),
                   where=(precision + recall) != 0)
    overall = {"accuracy": float(tp.sum() / len(labels)), "macro_precision": float(precision.mean()),
               "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean())}
    classes = [{"label": i, "class_name": CLASSES[i], "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]), "support": int(supports[i])}
               for i in range(10)]
    return overall, classes, cm


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def data_digest(images, labels):
    return hashlib.sha256((images.astype(np.float32) / np.float32(255)).tobytes()
                          + labels.astype("<i8").tobytes()).hexdigest()


def read_checkpoint(path, augmentation=None):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(f"Expected a Task 3/5 checkpoint dictionary: {path}")
    cfg = ck.get("config", {})
    if ck.get("model_name") != "CNN" or ck.get("architecture_version") != "cnn_v2":
        raise ValueError("Use the selected cnn_v2 baseline from outputs/task3_cnn/main/cnn_best.pt")
    if ck.get("checkpoint_type") != "best":
        raise ValueError("This comparison requires a best checkpoint, not a final checkpoint.")
    if ck.get("classes", CLASSES) != CLASSES:
        raise ValueError("Unexpected class-label order.")
    if augmentation is not None and bool(cfg.get("augmentation", False)) != augmentation:
        raise ValueError(f"Unexpected augmentation flag in {path}")
    return ck


def frozen_model(ck, device):
    cfg = ck.get("config", {})
    model = CNN(float(ck.get("dropout_rate", cfg.get("dropout_rate", 0))),
                float(ck.get("conv_dropout", cfg.get("conv_dropout", 0))))
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def load_training(data_dir, split_path, expected_hash):
    # Intentionally does not open t10k files during training/validation.
    images = read_idx(data_dir / "train-images-idx3-ubyte.gz", True)
    labels = read_idx(data_dir / "train-labels-idx1-ubyte.gz", False)
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise ValueError("Expected the 60,000 original training images and labels.")
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 6000)):
        raise ValueError("Expected 6,000 original training examples per class.")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing Task 2 split: {split_path}")
    with np.load(split_path, allow_pickle=False) as f:
        train_idx, val_idx = f["train_indices"].copy(), f["val_indices"].copy()
    for idx, n, per_class in [(train_idx, 48000, 4800), (val_idx, 12000, 1200)]:
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) != n:
            raise ValueError("Invalid training/validation index shape or dtype.")
        if idx.min() < 0 or idx.max() >= 60000:
            raise ValueError("Split index is out of bounds.")
        if not np.array_equal(np.bincount(labels[idx], minlength=10), np.full(10, per_class)):
            raise ValueError("Expected the same stratified 48,000/12,000 split as Task 3.")
    if len(np.unique(np.concatenate([train_idx, val_idx]))) != 60000:
        raise ValueError("Training and validation indices overlap or contain duplicates.")
    split_hash = hashlib.sha256(train_idx.astype("<i8").tobytes() + val_idx.astype("<i8").tobytes()).hexdigest()
    if expected_hash and split_hash != expected_hash:
        raise ValueError("The Task 2 split differs from the baseline CNN split.")
    return (images[train_idx], labels[train_idx].astype(np.int64),
            images[val_idx], labels[val_idx].astype(np.int64), train_idx, val_idx, split_hash)


def augment_one(image, rng, profile, forced_family=None):
    x = image.astype(np.float32, copy=True)
    if forced_family is None and rng.random() >= profile["probability"]:
        return x, "Clean"
    family = forced_family or ["rotation", "noise", "blur", "brightness"][int(rng.integers(4))]
    if family == "rotation":
        angle = float(rng.uniform(-profile["rotation_max"], profile["rotation_max"]))
        matrix = cv2.getRotationMatrix2D((13.5, 13.5), angle, 1.0)
        x = cv2.warpAffine(x, matrix, (28, 28), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        text = f"Rotation {angle:+.1f} deg"
    elif family == "noise":
        sigma = float(rng.uniform(0, profile["noise_max"]))
        x += rng.standard_normal(x.shape, dtype=np.float32) * sigma
        text = f"Noise sigma={sigma:.1f}"
    elif family == "blur":
        sigma = float(rng.uniform(profile["blur_min"], profile["blur_max"]))
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma,
                             borderType=cv2.BORDER_REFLECT_101)
        text = f"Blur sigma={sigma:.2f}"
    elif family == "brightness":
        factor = float(rng.uniform(profile["brightness_min"], 1.0))
        x *= factor
        text = f"Brightness x{factor:.2f}"
    else:
        raise ValueError(f"Unknown augmentation: {family}")
    return np.clip(x, 0, 255).astype(np.float32), text


class AugmentedDataset(Dataset):
    def __init__(self, images, labels, profile, seed):
        self.images, self.labels, self.profile = images, labels, profile
        # Separate from model/dropout/shuffle RNGs so initialization matches baseline policy.
        self.rng = np.random.default_rng(seed + 1000)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image, _ = augment_one(self.images[i], self.rng, self.profile)
        return torch.from_numpy(image[None] / np.float32(255)), int(self.labels[i])


def training_epoch(model, loader, optimizer, device, tag):
    model.train()
    total, loss_sum, correct = 0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with tqdm(loader, desc=tag, unit="batch", leave=False, ascii=True, dynamic_ncols=True,
              mininterval=0.5, file=sys.stdout) as progress:
        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
            total += len(y)
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            progress.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.2%}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
    return loss_sum / total, correct / total


def training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [r["epoch"] for r in history]
    for ax, metric in zip(axes[:2], ["loss", "accuracy"]):
        ax.plot(epochs, [r[f"train_{metric}"] for r in history], "--", alpha=0.6, label="Train (augmented, online)")
        if "train_eval_loss" in history[0]:
            ax.plot(epochs, [r[f"train_eval_{metric}"] for r in history], label="Train (clean, eval)")
        ax.plot(epochs, [r[f"val_{metric}"] for r in history], label="Validation (clean)")
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"Augmented CNN {metric}")
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[2].plot(epochs, [r["learning_rate"] for r in history])
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Cosine Annealing LR")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def augmentation_preview(images, labels, profile, seed, path):
    rng = np.random.default_rng(seed + 2000)
    samples = [(images[0], "Original")]
    samples += [augment_one(images[0], rng, profile, family) for family in ["rotation", "noise", "blur", "brightness"]]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    for ax, (image, text) in zip(axes, samples):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(text, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Training augmentation examples | {CLASSES[int(labels[0])]} | one transformation at a time")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def resolve_controls(args, baseline):
    original = baseline.get("config", {})
    mapping = {"epochs": "epochs", "batch_size": "batch_size", "lr": "learning_rate",
               "min_lr": "min_learning_rate", "weight_decay": "weight_decay",
               "dropout": "dropout_rate", "conv_dropout": "conv_dropout",
               "patience": "patience", "min_delta": "min_delta", "seed": "seed"}
    cfg, changes = {}, {}
    for attr, key in mapping.items():
        if key not in original:
            raise ValueError(f"Baseline checkpoint lacks saved setting {key}. Use the selected CNN main checkpoint.")
        val = getattr(args, attr)
        cfg[key] = original[key] if val is None else val
        if cfg[key] != original[key]:
            changes[key] = {"baseline": original[key], "augmented": cfg[key]}
    for key in ["optimizer", "scheduler", "early_stopping", "train_eval"]:
        if key not in original:
            raise ValueError(f"Baseline checkpoint lacks saved setting {key}.")
        cfg[key] = original[key]
    if cfg["optimizer"] != "AdamW" or cfg["scheduler"] != "CosineAnnealingLR":
        raise ValueError("This script expects the selected AdamW + CosineAnnealingLR baseline.")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["patience"] < 1 or cfg["seed"] < 0:
        raise ValueError("epochs, batch size, patience must be positive; seed must be non-negative.")
    if not np.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("learning rate must be finite and positive.")
    if not 0 <= cfg["min_learning_rate"] <= cfg["learning_rate"]:
        raise ValueError("min learning rate must be between 0 and initial learning rate.")
    for k in ["weight_decay", "min_delta"]:
        if not np.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError(f"{k} must be finite and non-negative.")
    for k in ["dropout_rate", "conv_dropout"]:
        if not 0 <= cfg[k] < 1:
            raise ValueError(f"{k} must be in [0,1).")
    return cfg, changes


def profile_from_args(args):
    profile = {k: getattr(args, k) for k in ["probability", "rotation_max", "noise_max", "blur_min", "blur_max", "brightness_min"]}
    if not all(np.isfinite(v) for v in profile.values()):
        raise ValueError("Augmentation parameters must be finite.")
    if not 0 <= profile["probability"] <= 1 or not 0 < profile["brightness_min"] <= 1:
        raise ValueError("probability must be in [0,1]; brightness-min in (0,1].")
    if profile["rotation_max"] < 0 or profile["noise_max"] < 0 or not 0 < profile["blur_min"] <= profile["blur_max"]:
        raise ValueError("Invalid rotation/noise/blur ranges.")
    return profile


def fit_augmented(data, config, out, device):
    train_x, train_y, val_x, val_y = data[:4]
    set_seed(config["seed"])
    dataset = AugmentedDataset(train_x, train_y, config["augmentation_profile"], config["seed"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config["seed"]))
    model = CNN(config["dropout_rate"], config["conv_dropout"]).to(device)
    # Intentionally no baseline load_state_dict: this model starts with fresh weights.
    config["initial_model_state_sha256"] = state_digest(model)
    write_json(out / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config["epochs"]-1), eta_min=config["min_learning_rate"])
    history, best_loss, best_epoch, reference, bad = [], float("inf"), 0, float("inf"), 0
    stopped = False

    def save(path, epoch, loss, accuracy, kind):
        torch.save({"model_name": "CNN", "architecture_version": "cnn_v2", "checkpoint_type": kind,
                    "epoch": epoch, "val_loss": loss, "val_accuracy": accuracy,
                    "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "config": deepcopy(config),
                    "dropout_rate": config["dropout_rate"], "conv_dropout": config["conv_dropout"],
                    "classes": CLASSES, "seed": config["seed"], "optimizer": "AdamW",
                    "epochs": config["epochs"], "batch_size": config["batch_size"],
                    "learning_rate": optimizer.param_groups[0]["lr"], "weight_decay": config["weight_decay"],
                    "normalization": "float32(pixel) / 255.0", "input_shape": [1,28,28],
                    "early_stopping_state": {"bad_epochs": bad, "reference_loss": reference, "triggered": stopped}}, path)

    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        t = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = training_epoch(model, loader, optimizer, device, f"Aug CNN Train {epoch:02d}/{config['epochs']}")
        va_loss, predicted = evaluate(model, val_x, val_y, config["batch_size"], device, f"Clean Val {epoch:02d}")
        va_acc = float(np.mean(predicted == val_y))
        if not np.isfinite(va_loss):
            raise RuntimeError("Validation loss is not finite.")
        if va_loss < reference - config["min_delta"]:
            reference, bad = va_loss, 0
        else:
            bad += 1
        stopped = bool(config["early_stopping"] and bad >= config["patience"])
        row = {"epoch": epoch, "train_loss": tr_loss, "train_accuracy": tr_acc,
               "val_loss": va_loss, "val_accuracy": va_acc, "learning_rate": lr,
               "early_stopping_bad_epochs": bad}
        if config["train_eval"]:
            clean_loss, clean_pred = evaluate(model, train_x, train_y, config["batch_size"], device, f"Clean Train-eval {epoch:02d}")
            row.update(train_eval_loss=clean_loss, train_eval_accuracy=float(np.mean(clean_pred==train_y)))
        row["epoch_seconds"] = time.perf_counter()-t
        history.append(row)
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            save(out/"cnn_best.pt", epoch, va_loss, va_acc, "best")
        write_csv(out/"cnn_history.csv", history)
        print(f"Epoch {epoch}/{config['epochs']} | train loss={tr_loss:.4f} acc={tr_acc:.2%} | "
              f"clean val loss={va_loss:.4f} acc={va_acc:.2%} | lr={lr:.7f} | best={best_epoch} | wait={bad}/{config['patience']}", flush=True)
        if stopped:
            print("Early stopping triggered.", flush=True)
            break
        if epoch < config["epochs"]:
            scheduler.step()
    save(out/"cnn_final.pt", epoch, va_loss, va_acc, "final")
    training_curves(history, out/"cnn_curves.png")
    summary = {"model":"CNN", "augmentation":True, "best_epoch":best_epoch,
               "best_val_loss":best_loss, "best_val_accuracy":history[best_epoch-1]["val_accuracy"],
               "final_epoch":epoch, "max_epochs":config["epochs"], "early_stopping_triggered":stopped,
               "stop_reason":"early_stopping" if stopped else "max_epochs",
               "training_seconds":time.perf_counter()-start}
    write_json(out/"training_summary.json", summary)
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def comparison_plot(rows, path, gain=False):
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    specs = [("rotation","Rotation","Angle (degrees)",0), ("noise","Gaussian noise","Sigma (0-255)",0),
             ("blur","Gaussian blur","Sigma (pixels)",0), ("brightness","Brightness","Brightness factor",1)]
    clean = next(r for r in rows if r["family"]=="clean")
    for ax,(family,title,xlabel,clean_x) in zip(axes.flat,specs):
        group=[r for r in rows if r["family"]==family]
        if gain:
            pts=sorted([(r["value"],r["accuracy_gain_pp"]) for r in group]+[(clean_x,clean["accuracy_gain_pp"])],reverse=family=="brightness")
            ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",color="#256E63",label="Augmented minus baseline")
            ax.axhline(0,color="#777777",lw=1)
        else:
            for name,label,color in [("baseline","Baseline CNN","#68778C"),("augmented","CNN + augmentation","#256E63")]:
                pts=sorted([(r["value"],r[f"{name}_accuracy_pct"]) for r in group]+[(clean_x,clean[f"{name}_accuracy_pct"])],reverse=family=="brightness")
                ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",label=label,color=color)
            ax.set_ylim(0,100)
        if family=="brightness":
            ax.invert_xaxis()
        ax.set(title=title,xlabel=xlabel,ylabel="Accuracy gain (percentage points)" if gain else "Accuracy (%)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def compare_models(images, labels, baseline_path, augmented_path, out, device, batch_size, noise_seed,
                   expected_manifest=None):
    out.mkdir(parents=True, exist_ok=False)
    models, checkpoints, before = {}, {}, {}
    for name,path,aug in [("baseline",baseline_path,False),("augmented",augmented_path,True)]:
        ck = read_checkpoint(path, augmentation=aug)
        models[name] = frozen_model(ck,device)
        checkpoints[name] = {"path":str(path.resolve()), "file_sha256":file_digest(path),
                             "selected_epoch":ck["epoch"], "selected_val_loss":ck["val_loss"]}
        before[name] = state_digest(models[name])
    data_hash = data_digest(images,labels)
    all_rows, detail, classes, manifest = [], [], [], []
    predictions={"y_true":labels}
    matrices={}
    clean={}
    expected={r["id"]:r["pixels_float32_0_255_sha256"] for r in (expected_manifest or [])}
    if expected and set(expected)!={c["id"] for c in conditions()}:
        raise ValueError("Task 4 condition set differs from this evaluator.")
    for i,case in enumerate(conditions(),1):
        print(f"\nComparison [{i}/16]: {case['label']}",flush=True)
        pixels=corrupt(images,case,noise_seed)
        digest=array_digest(pixels)
        if expected and expected[case["id"]]!=digest:
            raise ValueError(f"Task 4 pixel hash mismatch for {case['id']}; check seed/data/OpenCV/NumPy versions.")
        manifest.append({**case,"pixels_float32_0_255_sha256":digest})
        row={"condition":case["id"],"family":case["family"],"value":case["value"]}
        for name,model in models.items():
            loss,pred=evaluate(model,pixels,labels,batch_size,device,f"{name} {case['id']}")
            met,per,cm=metrics(labels,pred)
            if case["family"]=="clean":
                clean[name]=met["accuracy"]*100
            accuracy=met["accuracy"]*100
            row.update({f"{name}_accuracy_pct":accuracy,f"{name}_accuracy_drop_pp":clean[name]-accuracy,
                        f"{name}_macro_f1":met["macro_f1"]})
            detail.append({"model":name,"condition":case["id"],"family":case["family"],
                           "n_images":len(labels),"loss":loss,**met,"accuracy_pct":accuracy,
                           "accuracy_drop_pp":clean[name]-accuracy})
            classes.extend({"model":name,"condition":case["id"],**r} for r in per)
            predictions[f"{name}__{case['id']}"]=pred
            matrices[f"{name}__{case['id']}"]=cm
        row["accuracy_gain_pp"]=row["augmented_accuracy_pct"]-row["baseline_accuracy_pct"]
        all_rows.append(row)
        write_csv(out/"comparison.csv",all_rows)
        print(f"Baseline {row['baseline_accuracy_pct']:.2f}% | Augmented {row['augmented_accuracy_pct']:.2f}% | gain {row['accuracy_gain_pp']:+.2f} pp",flush=True)
    state_checks={name:{"before":before[name],"after":state_digest(model)} for name,model in models.items()}
    if any(r["before"]!=r["after"] for r in state_checks.values()):
        raise RuntimeError("Model state changed during fixed-weight comparison.")
    family_rows=[]
    for family in ["rotation","noise","blur","brightness"]:
        group=[r for r in all_rows if r["family"]==family]
        family_rows.append({"family":family,"settings_count":len(group),
                           "baseline_accuracy_pct":float(np.mean([r["baseline_accuracy_pct"] for r in group])),
                           "augmented_accuracy_pct":float(np.mean([r["augmented_accuracy_pct"] for r in group])),
                           "accuracy_gain_pp":float(np.mean([r["accuracy_gain_pp"] for r in group]))})
    corrupted=[r for r in all_rows if r["family"]!="clean"]
    summary=[]
    for name in ["baseline","augmented"]:
        worst=min(corrupted,key=lambda r:r[f"{name}_accuracy_pct"])
        summary.append({"model":name,"n_images":len(labels),"clean_accuracy_pct":clean[name],
                        "clean_accuracy_gain_pp":clean[name]-clean["baseline"],
                        "mean_corrupted_accuracy_pct_15_settings":float(np.mean([r[f"{name}_accuracy_pct"] for r in corrupted])),
                        "mean_corrupted_accuracy_pct_equal_families":float(np.mean([r[f"{name}_accuracy_pct"] for r in family_rows])),
                        "mean_accuracy_drop_pp":float(np.mean([r[f"{name}_accuracy_drop_pp"] for r in corrupted])),
                        "mean_corrupted_gain_pp":float(np.mean([r["accuracy_gain_pp"] for r in corrupted])) if name=="augmented" else 0.0,
                        "worst_condition":worst["condition"],"worst_accuracy_pct":worst[f"{name}_accuracy_pct"]})
    write_csv(out/"summary.csv",summary)
    write_csv(out/"family_summary.csv",family_rows)
    write_csv(out/"detailed_metrics.csv",detail)
    write_csv(out/"per_class_metrics.csv",classes)
    np.savez_compressed(out/"predictions.npz",**predictions)
    np.savez_compressed(out/"confusion_matrices.npz",**matrices)
    write_json(out/"condition_manifest.json",manifest)
    write_json(out/"evaluation_config.json",{"n_images":len(labels),"data_sha256":data_hash,
               "noise_seed":noise_seed,"batch_size":batch_size,"conditions":conditions(),
               "checkpoints":checkpoints,"state_hashes":state_checks,"same_pixels_for_both_models":True,
               "task4_pixels_verified":bool(expected),"torch_version":str(torch.__version__),
               "numpy_version":str(np.__version__),"opencv_version":str(cv2.__version__),
               "rotation":"center13.5/13.5, bilinear, black padding, 28x28",
               "blur":"sigmaX=sigmaY, kernel automatic, BORDER_REFLECT_101",
               "noise":"default_rng(seed), float32 standard_normal, sigma in 0-255, clipping, no uint8 rounding",
               "summary_weighting":"15-setting mean has six rotations; equal-family mean also supplied"})
    comparison_plot(all_rows,out/"accuracy_comparison.png")
    comparison_plot(all_rows,out/"accuracy_gain.png",gain=True)
    return summary


def train_mode(args,device):
    baseline=read_checkpoint(args.baseline,augmentation=False)
    controls,changes=resolve_controls(args,baseline)
    profile=profile_from_args(args)
    print("Training controls loaded from baseline:",json.dumps(controls,indent=2),flush=True)
    if changes:
        print("Overrides relative to baseline:",json.dumps(changes,indent=2),flush=True)
    print("Augmentation:",json.dumps(profile,indent=2),flush=True)
    data=load_training(args.data_dir,args.split_file,baseline["config"].get("split_sha256"))
    train_x,train_y,val_x,val_y,ti,vi,split_hash=data
    out=args.output_dir/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    baseline_copy=out/"baseline_best.pt"
    shutil.copy2(args.baseline,baseline_copy)
    shutil.copy2(Path(__file__),out/"training_source.py")
    np.savez_compressed(out/"split_indices.npz",train_indices=ti,val_indices=vi)
    cfg={**controls,"task":5,"model":"CNN","architecture_version":"cnn_v2","augmentation":True,
         "augmentation_profile":profile,"training_from_scratch":True,"finetuning":False,
         "checkpoint_selection":"lowest clean validation loss, matching baseline policy",
         "normalization":"float32(pixel)/255.0","split_sha256":split_hash,
         "validation_sha256":data_digest(val_x,val_y),
         "test_sha256":baseline["config"].get("test_sha256"),
         "training_data_sha256":hashlib.sha256(train_x.tobytes()+train_y.astype("<i8").tobytes()).hexdigest(),
         "source_split_file":str(args.split_file.resolve()),"data_dir":str(args.data_dir.resolve()),
         "source_baseline_checkpoint":str(args.baseline.resolve()),
         "baseline_checkpoint_file_sha256":file_digest(baseline_copy),
         "baseline_training_config":baseline["config"],"control_overrides":changes,
         "training_controls_match_baseline":not bool(changes),"scheduler_T_max":max(1,controls["epochs"]-1),
         "torch_version":str(torch.__version__),"opencv_version":str(cv2.__version__),
         "numpy_version":str(np.__version__),"device":str(device),
         "eval_batch_size":args.eval_batch_size,"evaluation_noise_seed":args.noise_seed,
         "evaluation_conditions":conditions(),
         "test_evaluated":False,"test_set_used_for_training_or_selection":False}
    augmentation_preview(train_x,train_y,profile,controls["seed"],out/"augmentation_examples.png")
    print(f"Training {len(train_y):,} | Validation {len(val_y):,} | output: {out}",flush=True)
    fit_augmented(data,cfg,out,device)
    cfg["selected_augmented_checkpoint_sha256"]=file_digest(out/"cnn_best.pt")
    write_json(out/"config.json",cfg)
    print("\nEvaluating BOTH selected checkpoints on VALIDATION data only...",flush=True)
    result=compare_models(val_x,val_y,baseline_copy,out/"cnn_best.pt",out/"validation",device,
                          args.eval_batch_size,args.noise_seed)
    write_csv(out/"validation_summary.csv",result)
    print(f"\nTraining and validation complete. No official test files were read.\n{out}",flush=True)
    print("Review validation_summary.csv and validation/comparison.csv to select settings.",flush=True)
    print("When settings are fixed, run the following command for final testing:",flush=True)
    print(f'python task5_cnn_augmentation.py evaluate --run-dir "{out}"',flush=True)


def find_task4_manifest(root,test_hash,baseline_hash,noise_seed):
    if not root.is_dir():
        return None,None
    for folder in sorted((p for p in root.iterdir() if p.is_dir()),reverse=True):
        if not (folder/"config.json").is_file() or not (folder/"condition_manifest.json").is_file():
            continue
        try:
            cfg=json.loads((folder/"config.json").read_text(encoding="utf-8"))
            matching=(cfg.get("test_sha256")==test_hash and cfg.get("noise_seed")==noise_seed
                      and cfg.get("models",{}).get("CNN",{}).get("checkpoint_file_sha256")==baseline_hash)
            if matching:
                return json.loads((folder/"condition_manifest.json").read_text(encoding="utf-8")),folder
        except (ValueError,OSError):
            continue
    return None,None


def evaluate_mode(args,device):
    if args.run_dir is None:
        raise ValueError('Specify the selected run: evaluate --run-dir "outputs/task5/YOUR_RUN"')
    run=args.run_dir
    cfg=json.loads((run/"config.json").read_text(encoding="utf-8"))
    if cfg.get("task")!=5 or not cfg.get("augmentation"):
        raise ValueError("run-dir must be a completed Task 5 augmentation run.")
    if cfg.get("evaluation_conditions")!=conditions():
        raise ValueError("Evaluation conditions changed since training; restore the original evaluation protocol.")
    if not (run/"validation_summary.csv").is_file():
        raise ValueError("This run has not completed validation comparison.")
    baseline_path,augmented_path=run/"baseline_best.pt",run/"cnn_best.pt"
    if file_digest(baseline_path)!=cfg["baseline_checkpoint_file_sha256"]:
        raise ValueError("The saved baseline checkpoint changed after training.")
    augmented=read_checkpoint(augmented_path,augmentation=True)
    if file_digest(augmented_path)!=cfg.get("selected_augmented_checkpoint_sha256"):
        raise ValueError("The augmented best checkpoint changed after validation selection.")
    if augmented["config"].get("split_sha256")!=cfg.get("split_sha256"):
        raise ValueError("The augmented checkpoint does not belong to the recorded split.")
    images,labels=load_test(args.data_dir)
    test_hash=data_digest(images,labels)
    if cfg.get("test_sha256") and test_hash!=cfg["test_sha256"]:
        raise ValueError("Official test data hash differs from the Task 3 baseline.")
    # Use the frozen run settings, not any edited augmentation constants in this file.
    noise_seed=cfg["evaluation_noise_seed"]
    manifest,folder=find_task4_manifest(args.task4_dir,test_hash,cfg["baseline_checkpoint_file_sha256"],noise_seed)
    if folder:
        print(f"Matching Task 4 results found: {folder}. Corrupted pixel hashes will be verified.",flush=True)
    out=run/("test_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    summary=compare_models(images,labels,baseline_path,augmented_path,out,device,args.eval_batch_size,
                           noise_seed,expected_manifest=manifest)
    write_json(out/"selected_run.json",{"training_run":str(run.resolve()),"task4_reference":str(folder.resolve()) if folder else None,
               "selection_method":"user selects augmentation settings using validation before this explicit test command",
               "training_config":cfg})
    print(f"\nFinal test comparison saved to: {out}",flush=True)
    for row in summary:
        print(f"{row['model']}: clean {row['clean_accuracy_pct']:.2f}% | mean corrupted {row['mean_corrupted_accuracy_pct_15_settings']:.2f}% "
              f"| corrupted gain {row['mean_corrupted_gain_pp']:+.2f} pp | clean change {row['clean_accuracy_gain_pp']:+.2f} pp",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode",nargs="?",choices=["train","evaluate"],default="train")
    parser.add_argument("--data-dir",type=Path,default=DATA_DIR)
    parser.add_argument("--split-file",type=Path,default=SPLIT_FILE)
    parser.add_argument("--baseline",type=Path,default=BASELINE_CHECKPOINT)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--task4-dir",type=Path,default=BASE_DIR/"outputs/task4")
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default=DEVICE)
    for flag,dtype,default in [("epochs",int,EPOCHS),("batch-size",int,BATCH_SIZE),("lr",float,LEARNING_RATE),
            ("min-lr",float,MIN_LEARNING_RATE),("weight-decay",float,WEIGHT_DECAY),
            ("dropout",float,DROPOUT_RATE),("conv-dropout",float,CONV_DROPOUT),
            ("patience",int,PATIENCE),("min-delta",float,MIN_DELTA),("seed",int,RANDOM_SEED)]:
        parser.add_argument("--"+flag,type=dtype,default=default)
    for flag,default in [("probability",AUGMENT_PROBABILITY),("rotation-max",ROTATION_MAX_DEGREES),
            ("noise-max",NOISE_SIGMA_MAX),("blur-min",BLUR_SIGMA_MIN),("blur-max",BLUR_SIGMA_MAX),
            ("brightness-min",BRIGHTNESS_MIN)]:
        parser.add_argument("--"+flag,type=float,default=default)
    parser.add_argument("--eval-batch-size",type=int,default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--noise-seed",type=int,default=EVALUATION_NOISE_SEED)
    args=parser.parse_args()
    if args.eval_batch_size<1 or args.noise_seed<0:
        parser.error("eval-batch-size must be positive and noise-seed non-negative")
    if args.device=="cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if args.device=="auto" else args.device))
    if device.type=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    print(f"Task 5 mode={args.mode} | device={device} | torch={torch.__version__} | OpenCV={cv2.__version__}",flush=True)
    if args.mode=="train":
        train_mode(args,device)
    else:
        evaluate_mode(args,device)


if __name__ == "__main__":
    main()
