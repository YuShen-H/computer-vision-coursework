
import argparse
import csv
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def read_idx(path, images):
    requested_path = path
    if not path.is_file() and path.suffix == ".gz":
        path = path.with_suffix("")
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {requested_path} (or the extracted file without .gz)\n"
                                "Put all four files in this folder or use --data-dir.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    header_size = 16 if images else 8
    if len(raw) < header_size:
        raise ValueError(f"Incomplete file: {path.name}")
    header = struct.unpack(">IIII" if images else ">II", raw[:header_size])
    if header[0] != (2051 if images else 2049):
        raise ValueError(f"Invalid IDX format: {path.name}")
    shape = header[1:] if images else (header[1],)
    values = np.frombuffer(raw, dtype=np.uint8, offset=header_size)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Incomplete IDX data: {path.name}")
    if images and shape[1:] != (28, 28):
        raise ValueError(f"Unexpected image size: {shape[1:]}")
    return values.reshape(shape)


def prepare_dataset(data_dir, seed=42):
    """Return normalized (x_train,y_train,x_val,y_val,x_test,y_test), indices.

    Image arrays have shape (N,28,28), dtype float32, range [0,1].
    MLP flattening / CNN channel dimensions will be added in Task 3.
    """
    x = read_idx(data_dir / "train-images-idx3-ubyte.gz", True)
    y = read_idx(data_dir / "train-labels-idx1-ubyte.gz", False)
    x_test = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", True)
    y_test = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", False)
    if len(x) != 60000 or len(y) != 60000 or len(x_test) != 10000 or len(y_test) != 10000:
        raise ValueError("Expected 60,000 training and 10,000 test images with matching labels.")
    if not np.array_equal(np.bincount(y, minlength=10), np.full(10, 6000)):
        raise ValueError("Expected 6,000 original training images in each of the ten classes.")
    if not np.array_equal(np.bincount(y_test, minlength=10), np.full(10, 1000)):
        raise ValueError("Expected 1,000 official test images in each class.")
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(y == label))
        train_indices.extend(indices[:4800])
        val_indices.extend(indices[4800:])
    train_indices = rng.permutation(np.asarray(train_indices))
    val_indices = rng.permutation(np.asarray(val_indices))
    arrays = (x[train_indices].astype(np.float32) / 255.0, y[train_indices],
              x[val_indices].astype(np.float32) / 255.0, y[val_indices],
              x_test.astype(np.float32) / 255.0, y_test)
    return arrays, (train_indices, val_indices)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare Fashion-MNIST for HW3 Task 2")
    parser.add_argument("--data-dir", type=Path, default=base / "data")
    parser.add_argument("--output-dir", type=Path, default=base / "outputs" / "task2")
    args = parser.parse_args()
    arrays, indices = prepare_dataset(args.data_dir)
    x_train, y_train, x_val, y_val, x_test, y_test = arrays
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "split_indices.npz", train_indices=indices[0],
                        val_indices=indices[1], seed=np.asarray(42))

    with (out / "class_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Class", "Train", "Validation", "Test"])
        for label, name in enumerate(CLASSES):
            writer.writerow([label, name, int(np.sum(y_train == label)),
                             int(np.sum(y_val == label)), int(np.sum(y_test == label))])
        writer.writerow(["", "Total", len(y_train), len(y_val), len(y_test)])

    fig, axes = plt.subplots(10, 3, figsize=(6, 14))
    for label, name in enumerate(CLASSES):
        for col, idx in enumerate(np.flatnonzero(y_train == label)[:3]):
            ax = axes[label, col]
            ax.imshow(x_train[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{label}: {name}", rotation=0, ha="right", va="center", labelpad=12)
    fig.suptitle("Fashion-MNIST Training Samples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out / "class_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {"seed": 42, "image_shape": [28, 28], "normalization": "float32(pixel) / 255.0",
               "classes": dict(enumerate(CLASSES)), "splits": {}}
    for name, values in [("Train", x_train), ("Validation", x_val), ("Test", x_test)]:
        stats = {"count": len(values), "dtype": str(values.dtype),
                 "min": float(values.min()), "max": float(values.max())}
        summary["splits"][name] = stats
        print(f"{name}: {len(values):,} images | shape={values.shape} | "
              f"dtype={values.dtype} | range=[{values.min():.3f}, {values.max():.3f}]")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Task 2 results to: {out.resolve()}")
    print("class_samples.png  - 3 training examples per class")
    print("class_counts.csv   - class labels and split counts")
    print("split_indices.npz  - reproducible split indices in the original training set")
    print("summary.json       - dataset size and normalization statistics")

if __name__ == "__main__":
    main()