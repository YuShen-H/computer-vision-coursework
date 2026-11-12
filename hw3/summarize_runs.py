
import argparse
import csv
import json
import math
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
FIELDS = [
    "model", "run_id", "status", "architecture_version", "max_epochs",
    "actual_epochs", "best_epoch", "best_val_loss", "best_val_accuracy_pct",
    "test_accuracy_pct", "macro_precision", "macro_recall", "macro_f1",
    "batch_size", "learning_rate", "min_learning_rate", "optimizer",
    "weight_decay", "dropout_rate", "conv_dropout", "early_stopping",
    "patience", "min_delta", "stop_reason", "seed", "augmentation",
    "parameters", "training_seconds", "scheduler", "scheduler_T_max",
    "split_sha256", "test_sha256", "source_split_file", "run_folder",
    "best_model_path", "curves_path", "notes",
]


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def first(*values):
    return next((x for x in values if x is not None and x != ""), "")


def percent(value):
    x = number(value)
    return round(100 * x, 4) if x is not None else ""


def read_json(path, warnings):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return {}


def read_csv(path, warnings):
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, ValueError, csv.Error) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return []


def collect_run(folder, warnings):
    config = read_json(folder / "config.json", warnings)
    summary = read_json(folder / "training_summary.json", warnings)
    tables = read_csv(folder / "test_results.csv", warnings)
    if not tables:
        tables = read_csv(folder / "comparison.csv", warnings)
    rows = []
    for model in ("MLP", "CNN"):
        lower = model.lower()
        metrics = read_json(folder / f"{lower}_metrics.json", warnings)
        table = next((r for r in tables if r.get("model", "").upper() == model), {})
        history = read_csv(folder / f"{lower}_history.csv", warnings)
        model_summary = summary if str(summary.get("model", "")).upper() == model else {}
        declared = str(config.get("model", "")).upper() == model
        if not (metrics or table or history or model_summary or declared or
                (folder / f"{lower}_best.pt").exists()):
            continue
        # Most specific completed-model metadata takes priority over run settings.
        info = {**config, **table, **model_summary, **metrics}
        best_history = min((r for r in history if number(r.get("val_loss")) is not None),
                           key=lambda r: number(r["val_loss"]), default={})
        epochs = [number(r.get("epoch")) for r in history]
        actual = max((x for x in epochs if x is not None), default=None)
        tested = number(first(metrics.get("accuracy"), table.get("accuracy"))) is not None
        completed = bool(tested or model_summary)
        notes = []
        if not completed:
            notes.append("May be running/interrupted; history is provisional")
        if not info.get("split_sha256"):
            notes.append("Split hash unavailable; verify identical validation data before comparison")
        if not info.get("architecture_version"):
            notes.append("Architecture version was not saved")
        row = {k: info.get(k, "") for k in FIELDS}
        row.update(
            model=model, run_id=folder.name,
            status="completed_tested" if tested else ("completed_no_test" if completed else "incomplete_or_unknown"),
            max_epochs=config.get("epochs", ""),
            actual_epochs=first(info.get("final_epoch"), int(actual) if actual is not None else None),
            best_epoch=first(info.get("best_epoch"), best_history.get("epoch")),
            best_val_loss=first(info.get("best_val_loss"), best_history.get("val_loss")),
            best_val_accuracy_pct=percent(first(info.get("best_val_accuracy"), best_history.get("val_accuracy"))),
            test_accuracy_pct=percent(first(metrics.get("accuracy"), table.get("accuracy"))),
            learning_rate=first(config.get("learning_rate"), info.get("initial_learning_rate")),
            run_folder=str(folder.resolve()),
            best_model_path=str((folder / f"{lower}_best.pt").resolve()) if (folder / f"{lower}_best.pt").exists() else "",
            curves_path=str((folder / f"{lower}_curves.png").resolve()) if (folder / f"{lower}_curves.png").exists() else "",
            notes="; ".join(notes),
        )
        rows.append(row)
    return rows


def sort_key(row):
    val = number(row.get("best_val_loss"))
    return (row["model"], val if val is not None else float("inf"), row["run_folder"])


def write_table(path, rows):
    # Replace only the generated summary. Original run folders remain untouched.
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        temp.replace(path)
    except PermissionError:
        raise SystemExit(f"Please close {path.name} in Excel, then run again. Temporary output: {temp}")


def main():
    parser = argparse.ArgumentParser(description="Summarize saved MLP/CNN runs; no retraining required")
    parser.add_argument("--root", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs/run_summary")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Output root does not exist: {args.root}")
    folders = set()
    for pattern in ("config.json", "*_metrics.json", "*_history.csv"):
        folders.update(p.parent for p in args.root.rglob(pattern))
    rows, warnings = [], []
    for folder in sorted(folders):
        rows.extend(collect_run(folder, warnings))
    rows.sort(key=sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "all_runs.csv", rows)
    for model in ("MLP", "CNN"):
        selected = [r for r in rows if r["model"] == model]
        write_table(args.output_dir / f"{model.lower()}_runs.csv", selected)
        complete = [r for r in selected if r["status"].startswith("completed") and number(r["best_val_loss"]) is not None]
        print(f"{model}: {len(selected)} saved runs, {len(complete)} completed runs with validation results")
    print(f"\nSaved summaries to: {args.output_dir.resolve()}")
    print("Rows are sorted by model and lowest validation loss. Compare runs using the SAME validation split.")
    print("Use validation results to select settings; test results are for final reporting.")
    print("Blank values mean the original run did not save that information.")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not rows:
        print("No saved MLP/CNN runs found. Check --root points to your training outputs.")


if __name__ == "__main__":
    main()
