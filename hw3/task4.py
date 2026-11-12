
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV is missing. In your current environment run: python -m pip install opencv-python==4.10.0.84")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MLP_CHECKPOINT = BASE_DIR / "outputs/task3_mlp/main/mlp_best.pt"
CNN_CHECKPOINT = BASE_DIR / "outputs/task3_cnn/main/cnn_best.pt"
OUTPUT_DIR = BASE_DIR / "outputs/task4"
BATCH_SIZE = 256 
RANDOM_SEED = 42  
DEVICE = "auto"  

ROTATIONS = [-30, -20, -10, 10, 20, 30]
NOISE_SIGMAS = [10, 25, 40]  
BLUR_SIGMAS = [0.5, 1.0, 2.0]
BRIGHTNESS_FACTORS = [0.8, 0.6, 0.4]


CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class MLP(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 10))

    def forward(self, x):
        return self.layers(x)  # logits; CrossEntropyLoss already includes LogSoftmax

    def predict_proba(self, x):
        # Call model.eval() and use torch.no_grad() for inference.
        return torch.softmax(self(x), dim=1)


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


def load_model(name, path, device, test_sha256):
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} best checkpoint not found: {path}\n"
            f"Copy {name.lower()}_best.pt into the main folder, or set --{name.lower()}-checkpoint.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Task 3 checkpoint dictionary: {path}")
    config = checkpoint.get("config", {})
    expected_version = f"{name.lower()}_v2"
    version = checkpoint.get("architecture_version", config.get("architecture_version"))
    if version != expected_version:
        raise ValueError(f"{name}: expected {expected_version}, found {version!r}. Use the selected standalone Task 3 best model.")
    if checkpoint.get("model_name") != name:
        raise ValueError(f"Checkpoint model name does not match {name}: {path}")
    if checkpoint.get("checkpoint_type") != "best":
        raise ValueError(f"{name}: use the best checkpoint, not the final checkpoint.")
    if checkpoint.get("classes", CLASSES) != CLASSES:
        raise ValueError(f"Class order does not match Fashion-MNIST: {path}")
    expected_test = config.get("test_sha256")
    if expected_test and expected_test != test_sha256:
        raise ValueError(f"{name}: the test dataset differs from the Task 3 test dataset (hash mismatch).")
    dropout = float(checkpoint.get("dropout_rate", config.get("dropout_rate", 0.0)))
    conv_dropout = float(checkpoint.get("conv_dropout", config.get("conv_dropout", 0.0)))
    model = MLP(dropout) if name == "MLP" else CNN(dropout, conv_dropout)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    metadata = {"checkpoint": str(path.resolve()), "checkpoint_file_sha256": file_digest(path),
                "architecture_version": version, "selected_epoch": checkpoint.get("epoch"),
                "selected_val_loss": checkpoint.get("val_loss"),
                "dropout_rate": dropout, "conv_dropout": conv_dropout,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_config": config, "state_sha256_before": state_digest(model)}
    print(f"Loaded {name}: {path} | best epoch={metadata['selected_epoch']} | {metadata['parameters']:,} parameters", flush=True)
    del checkpoint
    return model, metadata


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


def summaries(details):
    family_rows, model_rows = [], []
    for model in ("MLP", "CNN"):
        clean = next(r for r in details if r["model"] == model and r["family"] == "clean")
        corrupted = [r for r in details if r["model"] == model and r["family"] != "clean"]
        worst = min(corrupted, key=lambda r: r["accuracy"])
        means = []
        for family in ("rotation", "noise", "blur", "brightness"):
            group = [r for r in corrupted if r["family"] == family]
            avg = float(np.mean([r["accuracy_pct"] for r in group]))
            means.append(avg)
            family_rows.append({"model": model, "family": family, "settings_count": len(group),
                                "clean_accuracy_pct": clean["accuracy_pct"],
                                "mean_corrupted_accuracy_pct": avg,
                                "mean_accuracy_drop_pp": clean["accuracy_pct"] - avg})
        avg = float(np.mean([r["accuracy_pct"] for r in corrupted]))
        model_rows.append({"model": model, "clean_accuracy_pct": clean["accuracy_pct"],
                           "mean_corrupted_accuracy_pct_15_settings": avg,
                           "mean_accuracy_drop_pp_15_settings": clean["accuracy_pct"] - avg,
                           "mean_corrupted_accuracy_pct_4_families_equal_weight": float(np.mean(means)),
                           "worst_condition": worst["condition"], "worst_accuracy_pct": worst["accuracy_pct"],
                           "max_accuracy_drop_pp": worst["accuracy_drop_pp"]})
    return family_rows, model_rows


def plot_curves(details, path, drop=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    names = {"rotation": ("Rotation", "Angle (degrees)", 0),
             "noise": ("Gaussian noise", "Sigma (0-255 scale)", 0),
             "blur": ("Gaussian blur", "Sigma (pixels)", 0),
             "brightness": ("Reduced brightness", "Brightness factor", 1)}
    metric = "accuracy_drop_pp" if drop else "accuracy_pct"
    for ax, (family, (title, xlabel, clean_x)) in zip(axes.flat, names.items()):
        for model, color in [("MLP", "#7251A1"), ("CNN", "#176D8F")]:
            clean = next(r for r in details if r["model"] == model and r["family"] == "clean")
            pts = [(r["value"], r[metric]) for r in details if r["model"] == model and r["family"] == family]
            pts.append((clean_x, clean[metric]))
            pts.sort(reverse=family == "brightness")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model, color=color)
        ax.set(title=title, xlabel=xlabel, ylabel="Accuracy decrease (percentage points)" if drop else "Accuracy (%)")
        if not drop:
            ax.set_ylim(0, 100)
        if family == "brightness":
            ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Fixed-weight robustness: accuracy decrease" if drop else "Fixed-weight robustness: test accuracy", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_examples(cases, examples, index, label, path):
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax, case, image in zip(axes.flat, cases, examples):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(case["label"], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Same test image, separate transformations | index={index} | {CLASSES[label]}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    print("Task 4 started: fixed-weight robustness evaluation.", flush=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--mlp-checkpoint", type=Path, default=MLP_CHECKPOINT)
    parser.add_argument("--cnn-checkpoint", type=Path, default=CNN_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seed < 0:
        parser.error("batch-size must be positive and seed must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(1)
    print(f"PyTorch {torch.__version__} | OpenCV {cv2.__version__} | device={device}", flush=True)
    images, labels = load_test(args.data_dir)
    # Same normalized image bytes and int64 label bytes as Task 3 config.json.
    test_digest = hashlib.sha256((images.astype(np.float32) / np.float32(255)).tobytes()
                                 + labels.astype("<i8").tobytes()).hexdigest()
    models, metadata = {}, {}
    for name, path in [("MLP", args.mlp_checkpoint), ("CNN", args.cnn_checkpoint)]:
        models[name], metadata[name] = load_model(name, path, device, test_digest)
    out = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    cases = conditions()
    config = {"task": 4, "test_images": len(images), "test_sha256": test_digest,
              "official_test_order": True, "batch_size": args.batch_size, "noise_seed": args.seed,
              "numpy_version": str(np.__version__), "opencv_version": str(cv2.__version__),
              "torch_version": str(torch.__version__), "device": str(device),
              "source_data_dir": str(args.data_dir.resolve()), "conditions": cases,
              "normalization": "float32 corrupted pixels on [0,255], then divide by 255",
              "corruption_composition": "each condition independently starts from original images",
              "rotation": {"center": [13.5, 13.5], "size": [28, 28], "scale": 1.0,
                           "interpolation": "INTER_LINEAR", "border": "BORDER_CONSTANT", "fill": 0},
              "noise": {"distribution": "Gaussian mean 0, sigma on 0-255 scale",
                        "generator": "numpy default_rng(seed).standard_normal(shape,dtype=float32)",
                        "shared_standard_normal_draws_across_sigma_levels": True,
                        "same_pixels_for_both_models": True, "realizations_per_sigma": 1},
              "blur": {"kernel": [0, 0], "sigmaX": "specified sigma", "sigmaY": "specified sigma",
                       "border": "BORDER_REFLECT_101"},
              "pixel_clipping": [0, 255], "round_to_uint8_after_corruption": False,
              "checkpoint_weights_fixed": True, "selection_or_training_performed": False,
              "drop_definition": "clean_accuracy_pct - corrupted_accuracy_pct; percentage points",
              "summary_weighting": "15-setting mean counts six rotation settings and three each of other families; equal-family mean also reported",
              "confusion_matrix_orientation": "rows=true, columns=predicted",
              "macro_average": "unweighted mean across all 10 classes, zero division=0",
              "models": metadata}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), out / "evaluation_source.py")
    print(f"Test set: {len(images):,} images | {len(cases)} conditions | output: {out}", flush=True)
    details, wide, per_class, manifest, examples = [], [], [], [], []
    predictions_archive = {"y_true": labels}
    confusion_archive = {}
    example_index = int(np.flatnonzero(labels == 6)[0])
    baseline = {}
    for ci, case in enumerate(cases, 1):
        print(f"\n[{ci:02d}/{len(cases)}] {case['label']}", flush=True)
        pixels = corrupt(images, case, args.seed)
        examples.append(pixels[example_index].copy())
        manifest.append({**case, "pixels_float32_0_255_sha256": array_digest(pixels)})
        comparison = {"condition": case["id"], "family": case["family"], "value": case["value"]}
        for name, model in models.items():
            loss, predicted = evaluate(model, pixels, labels, args.batch_size, device, f"{name} {case['id']}")
            overall, classes, cm = metrics(labels, predicted)
            if case["family"] == "clean":
                baseline[name] = overall["accuracy"]
            decrease = 100 * (baseline[name] - overall["accuracy"])
            row = {"model": name, "condition": case["id"], "family": case["family"], "value": case["value"],
                   "n_test": len(labels), "test_loss": loss, **overall,
                   "accuracy_pct": 100 * overall["accuracy"],
                   "clean_accuracy_pct": 100 * baseline[name], "accuracy_drop_pp": decrease}
            details.append(row)
            per_class.extend({"model": name, "condition": case["id"], **r} for r in classes)
            predictions_archive[f"{name.lower()}__{case['id']}"] = predicted
            confusion_archive[f"{name.lower()}__{case['id']}"] = cm
            comparison.update({f"{name.lower()}_accuracy_pct": row["accuracy_pct"],
                               f"{name.lower()}_accuracy_drop_pp": decrease,
                               f"{name.lower()}_macro_f1": overall["macro_f1"]})
            print(f"{name}: accuracy={overall['accuracy']:.2%} | decrease={decrease:+.2f} pp | macro-F1={overall['macro_f1']:.4f}", flush=True)
        wide.append(comparison)
        # Save incremental CSVs so completed conditions survive an interruption.
        write_csv(out / "robustness_results.csv", wide)
        write_csv(out / "detailed_metrics.csv", details)
        del pixels
    for name, model in models.items():
        after = state_digest(model)
        metadata[name]["state_sha256_after"] = after
        metadata[name]["state_unchanged"] = after == metadata[name]["state_sha256_before"]
        if not metadata[name]["state_unchanged"]:
            raise RuntimeError(f"{name} state changed during testing.")
    family_rows, model_rows = summaries(details)
    write_csv(out / "family_summary.csv", family_rows)
    write_csv(out / "model_summary.csv", model_rows)
    write_csv(out / "per_class_metrics.csv", per_class)
    np.savez_compressed(out / "test_predictions.npz", **predictions_archive)
    np.savez_compressed(out / "confusion_matrices.npz", **confusion_archive)
    (out / "condition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_curves(details, out / "robustness_accuracy.png")
    plot_curves(details, out / "robustness_drop.png", drop=True)
    plot_examples(cases, examples, example_index, int(labels[example_index]), out / "corruption_examples.png")
    (out / "README.txt").write_text(
        "Task 4: selected model weights are fixed. All official test images use the same order.\n"
        "robustness_results.csv: main MLP/CNN comparison, one row per condition.\n"
        "detailed_metrics.csv: accuracy, loss and macro metrics per model/condition.\n"
        "model_summary.csv: overall corruption averages and worst conditions.\n"
        "family_summary.csv: separate rotation/noise/blur/brightness averages.\n"
        "Accuracy columns ending in _pct are percentages. Accuracy decrease is in percentage points.\n"
        "Macro precision/recall/F1 and the raw accuracy column are fractions [0,1].\n"
        "The 15-setting average includes six rotations; the equal-family mean weights four families equally.\n"
        "Noise uses one reproducible realization per sigma, shared across models; no confidence intervals.\n"
        "config.json records checkpoint paths and state hashes. condition_manifest.json hashes the exact corrupted pixels.\n"
        "Use the same transformation settings and seed for Task 5. Do not tune models using these test results.\n",
        encoding="utf-8")
    print(f"\nTask 4 complete. Both model states are unchanged. Results saved to:\n{out}", flush=True)
    for row in model_rows:
        print(f"{row['model']}: clean={row['clean_accuracy_pct']:.2f}% | mean corrupted={row['mean_corrupted_accuracy_pct_15_settings']:.2f}% "
              f"| mean decrease={row['mean_accuracy_drop_pp_15_settings']:.2f} pp | worst={row['worst_condition']}", flush=True)


if __name__ == "__main__":
    main()
