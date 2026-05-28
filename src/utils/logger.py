import csv
from pathlib import Path

def init_csv_logger(log_path, fieldnames):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def log_to_csv(log_path, row):
    log_path = Path(log_path)

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)