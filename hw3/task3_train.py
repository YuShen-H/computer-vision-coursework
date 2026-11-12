
import argparse
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import random
import struct
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"  
SPLIT_FILE = BASE_DIR / "outputs/task2/split_indices.npz" 
OUTPUT_DIR = BASE_DIR / "outputs/task3" 
EPOCHS = 80 
BATCH_SIZE = 128  
LEARNING_RATE = 0.01 
MIN_LEARNING_RATE = 0.001 
WEIGHT_DECAY = 1e-4 
DROPOUT_RATE = 0.5 
RANDOM_SEED = 42  
DEVICE = "cuda"  

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class MLP(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Parameter names remain compatible with earlier checkpoints.
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # Only the fully connected ReLU produces a two-dimensional tensor.
            if isinstance(layer, nn.ReLU) and x.ndim == 2:
                x = self.dropout(x)
        return x


def read_idx(path, images):
    requested = path
    if not path.is_file():
        path = path.with_suffix("")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {requested} and extracted equivalent.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    size = 16 if images else 8
    if len(raw) < size:
        raise ValueError(f"Incomplete file: {path}")
    header = struct.unpack(">IIII" if images else ">II", raw[:size])
    if header[0] != (2051 if images else 2049):
        raise ValueError(f"Invalid IDX header: {path}")
    shape = header[1:]
    if images and shape[1:] != (28, 28):
        raise ValueError(f"Expected 28x28 images: {path}")
    values = np.frombuffer(raw, dtype=np.uint8, offset=size)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Incomplete IDX payload: {path}")
    return values.reshape(shape)


def load_data(data_dir, split_path):
    x = read_idx(data_dir / "train-images-idx3-ubyte.gz", True)
    y = read_idx(data_dir / "train-labels-idx1-ubyte.gz", False)
    xt = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", True)
    yt = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", False)
    if len(x) != 60000 or len(y) != 60000 or len(xt) != 10000 or len(yt) != 10000:
        raise ValueError("Expected 60,000 training and 10,000 test images and labels.")
    if not np.array_equal(np.bincount(y, minlength=10), np.full(10, 6000)):
        raise ValueError("Original training labels must have 6,000 examples per class.")
    if not np.array_equal(np.bincount(yt, minlength=10), np.full(10, 1000)):
        raise ValueError("Test labels must have 1,000 examples per class.")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing Task 2 split file: {split_path}")
    with np.load(split_path, allow_pickle=False) as s:
        train_idx = s["train_indices"].copy()
        val_idx = s["val_indices"].copy()
    for idx, count in [(train_idx, 48000), (val_idx, 12000)]:
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) != count:
            raise ValueError("Invalid Task 2 split shape or dtype.")
        if idx.min() < 0 or idx.max() >= len(y):
            raise ValueError("Split index outside original training set.")
    if len(np.unique(np.concatenate((train_idx, val_idx)))) != 60000:
        raise ValueError("Split indices overlap or contain duplicates.")
    for idx, count in [(train_idx, 4800), (val_idx, 1200)]:
        if not np.array_equal(np.bincount(y[idx], minlength=10), np.full(10, count)):
            raise ValueError("Expected a stratified 48,000/12,000 Task 2 split.")
    def dataset(images, labels):
        return TensorDataset(torch.from_numpy(images.astype(np.float32)[:, None] / 255.0),
                             torch.from_numpy(labels.astype(np.int64)))
    return (dataset(x[train_idx], y[train_idx]), dataset(x[val_idx], y[val_idx]),
            dataset(xt, yt), train_idx, val_idx)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, shuffle, seed, cuda):
    # num_workers=0 avoids Windows multiprocessing startup issues.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=cuda, generator=torch.Generator().manual_seed(seed))


def run_epoch(model, loader, device, optimizer=None, tag="", collect=False):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    truths, predictions = [], []
    label = tag or ("Train" if training else ("Test" if collect else "Validation"))
    with torch.set_grad_enabled(training), tqdm(
            loader, desc=label, unit="batch", dynamic_ncols=True, mininterval=0.5,
            leave=False, ascii=True, file=sys.stdout) as progress:
        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)  # CrossEntropyLoss accepts logits; no Softmax here.
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
            correct += int((predicted == y).sum().item())
            if collect:
                truths.append(y.cpu().numpy())
                predictions.append(predicted.cpu().numpy())
            status = {"loss": f"{loss_sum / total:.4f}", "acc": f"{correct / total:.2%}"}
            if training:
                status["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(status, refresh=False)
    if not total:
        raise ValueError("Empty dataset.")
    result = {"loss": loss_sum / total, "accuracy": correct / total}
    if collect:
        result.update(y_true=np.concatenate(truths), y_pred=np.concatenate(predictions))
    return result


def classification_metrics(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted_count = cm.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros(10), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10),
                   where=(precision + recall) != 0)
    metrics = {"accuracy": float(tp.sum() / cm.sum()),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean())}
    rows = [{"label": i, "class": CLASSES[i], "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(10)]
    return metrics, rows, cm


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history, name, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = [r["epoch"] for r in history]
    for ax, metric in zip(axes[:2], ["loss", "accuracy"]):
        ax.plot(epochs, [r[f"train_{metric}"] for r in history], marker="o", markersize=3,
                label="Train (online)")
        ax.plot(epochs, [r[f"val_{metric}"] for r in history], marker="o", markersize=3,
                label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"{name} {metric.title()}")
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[2].plot(epochs, [r["learning_rate"] for r in history], marker="o", markersize=3)
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title=f"{name} Cosine Annealing LR")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(cm, name, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, label="Number of images", fraction=0.046, pad=0.04)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title=f"{name} Clean Test Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_model(name, model_type, datasets, args, device, out):
    set_seed(args.seed)
    train_loader = make_loader(datasets[0], args.batch_size, True, args.seed, device.type == "cuda")
    val_loader = make_loader(datasets[1], args.batch_size, False, args.seed, device.type == "cuda")
    model = model_type(dropout_rate=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Step between epochs so the final training epoch itself uses eta_min.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 1), eta_min=args.min_lr)
    history, best_loss, best_epoch = [], float("inf"), 0
    checkpoint_path = out / f"{name.lower()}_best.pt"
    final_path = out / f"{name.lower()}_final.pt"

    def save_checkpoint(path, epoch, validation, lr, kind):
        torch.save({"model_name": name, "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "checkpoint_type": kind, "epoch": epoch,
                    "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                    "learning_rate": lr, "initial_learning_rate": args.lr,
                    "min_learning_rate": args.min_lr, "scheduler": "CosineAnnealingLR",
                    "weight_decay": args.weight_decay, "dropout_rate": args.dropout,
                    "classes": CLASSES, "normalization": "float32(pixel) / 255.0",
                    "input_shape": [1, 28, 28], "seed": args.seed}, path)

    start = time.perf_counter()
    print(f"\nTraining {name}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, device, optimizer,
                       tag=f"{name} Train {epoch:02d}/{args.epochs}")
        va = run_epoch(model, val_loader, device, tag=f"{name} Val {epoch:02d}/{args.epochs}")
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
               "val_loss": va["loss"], "val_accuracy": va["accuracy"],
               "learning_rate": current_lr,
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if va["loss"] < best_loss:
            best_loss, best_epoch = va["loss"], epoch
            save_checkpoint(checkpoint_path, epoch, va, current_lr, "best")
        write_csv(out / f"{name.lower()}_history.csv", history)
        print(f"{name} [{epoch:02d}/{args.epochs}] train loss={tr['loss']:.4f} "
              f"acc={tr['accuracy']:.2%} | val loss={va['loss']:.4f} acc={va['accuracy']:.2%} "
              f"| lr={current_lr:.7f} | best epoch={best_epoch} | {row['epoch_seconds']:.1f}s", flush=True)
        if epoch < args.epochs:
            scheduler.step()
    # Save the last epoch before reloading any best checkpoint for evaluation.
    save_checkpoint(final_path, args.epochs, va, current_lr, "final")
    print(f"Saved {name} checkpoints: {checkpoint_path.name}, {final_path.name}", flush=True)
    seconds = time.perf_counter() - start
    plot_history(history, name, out / f"{name.lower()}_curves.png")
    architecture = str(model)
    count = sum(p.numel() for p in model.parameters())
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model": name, "parameters": count, "best_epoch": best_epoch,
            "weight_decay": args.weight_decay, "dropout_rate": args.dropout,
            "best_val_loss": best_loss, "best_val_accuracy": history[best_epoch - 1]["val_accuracy"],
            "final_epoch": args.epochs, "final_val_loss": va["loss"],
            "final_val_accuracy": va["accuracy"], "final_learning_rate": current_lr,
            "training_seconds": seconds, "architecture": architecture}


def main():
    print("Task 3 started. Preparing training configuration...", flush=True)
    parser = argparse.ArgumentParser(description="Train MLP and CNN for HW3 Task 3")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if not 0 <= args.min_lr <= args.lr:
        parser.error("min-lr must be between 0 and the initial lr")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and non-negative")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0,1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"PyTorch: {torch.__version__} | Device: {device}", flush=True)
    print(f"Settings: epochs={args.epochs} per model | batch size={args.batch_size} | "
          f"initial lr={args.lr} | min lr={args.min_lr} | seed={args.seed} | "
          "Adam + CrossEntropyLoss + CosineAnnealingLR", flush=True)
    print(f"Regularization: weight decay={args.weight_decay:g} | dropout={args.dropout}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading data from: {args.data_dir}", flush=True)
    datasets = load_data(args.data_dir, args.split_file)
    print("Split: Train 48,000 | Validation 12,000 | Test 10,000 | Pixels: [0,1]", flush=True)
    out = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Output folder: {out}", flush=True)
    np.savez_compressed(out / "split_indices.npz", train_indices=datasets[3], val_indices=datasets[4])
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
              "scheduler": "CosineAnnealingLR", "min_learning_rate": args.min_lr,
              "scheduler_T_max": max(1, args.epochs - 1),
              "scheduler_step": "once between epochs; no warm restarts",
              "checkpoint_outputs": "best (lowest validation loss) and final (last epoch)",
              "test_checkpoint": "best",
              "weight_decay": args.weight_decay, "dropout_rate": args.dropout,
              "dropout_placement": "MLP: after both hidden ReLUs; CNN: after dense hidden ReLU",
              "seed": args.seed, "optimizer": "Adam", "loss": "CrossEntropyLoss",
              "checkpoint_selection": "lowest validation loss", "augmentation": False,
              "normalization": "float32(pixel)/255.0", "torch_version": str(torch.__version__),
              "device": str(device), "source_split_file": str(args.split_file.resolve()),
              "metric_scale": "fractions from 0 to 1", "confusion_matrix": "rows=true, columns=predicted",
              "macro_average": "unweighted mean across all 10 classes; zero division returns zero"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    specs = [("MLP", MLP), ("CNN", CNN)]
    trained = [train_model(name, cls, datasets, args, device, out) for name, cls in specs]
    print("\nBoth models trained. Evaluating selected checkpoints on the clean test set...", flush=True)
    test_loader = make_loader(datasets[2], args.batch_size, False, args.seed, device.type == "cuda")
    summaries = []
    for (name, model_type), info in zip(specs, trained):
        checkpoint = torch.load(out / f"{name.lower()}_best.pt", map_location=device, weights_only=True)
        model = model_type(dropout_rate=checkpoint.get("dropout_rate", 0.0)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        result = run_epoch(model, test_loader, device, tag=f"{name} Test", collect=True)
        metrics, rows, cm = classification_metrics(result["y_true"], result["y_pred"])
        write_csv(out / f"{name.lower()}_per_class.csv", rows)
        np.savetxt(out / f"{name.lower()}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        plot_confusion(cm, name, out / f"{name.lower()}_confusion_matrix.png")
        np.savez_compressed(out / f"{name.lower()}_test_predictions.npz",
                            y_true=result["y_true"], y_pred=result["y_pred"])
        info.update(test_loss=result["loss"], **metrics)
        (out / f"{name.lower()}_metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        summaries.append({k: v for k, v in info.items() if k != "architecture"})
        print(f"{name}: Accuracy={metrics['accuracy']:.2%}, "
              f"Macro-Precision={metrics['macro_precision']:.4f}, "
              f"Macro-Recall={metrics['macro_recall']:.4f}, Macro-F1={metrics['macro_f1']:.4f}", flush=True)
        del model, checkpoint
    write_csv(out / "comparison.csv", summaries)
    print(f"\nTask 3 complete. Results saved to: {out}", flush=True)
    print("Report: comparison.csv, *_curves.png, *_confusion_matrix.png, *_per_class.csv", flush=True)
    print("Saved *_best.pt and *_final.pt; test metrics above use the best checkpoints.", flush=True)
    print("Keep *_best.pt for Task 4 robustness testing.", flush=True)


if __name__ == "__main__":
    main()
