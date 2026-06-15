"""
run_all.py — 批量训练 interval=1~12，保存权重，最后打印汇总表
用法：python run_all.py
"""

import os
import json
import sys
import numpy as np
import torch
from config import Config
from dataset import build_dataloaders
from models import MODEL_REGISTRY, build_model
from train import train_model, test_model, NO_TRAIN_MODELS

# ── 要跑的 interval 范围（修改此处可自定义） ──
START = 1
END   = 12


def run_one(interval, cfg, all_results):
    print(f"\n{'#'*60}")
    print(f"  INTERVAL = {interval}")
    print(f"{'#'*60}")
    sys.stdout.flush()

    cfg.interval = interval
    device = cfg.get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, scaler_params = build_dataloaders(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    interval_results = {}

    for name in MODEL_REGISTRY.keys():
        if name in NO_TRAIN_MODELS:
            print(f"\n  [NO-TRAIN] {name}")
            model = build_model(name, cfg).to(device)
            history = {"train": [], "val": []}
        else:
            model, history = train_model(name, cfg, train_loader, val_loader, device)

        metrics, preds, trues = test_model(
            model, name, cfg.seq_len, test_loader, scaler_params, device
        )

        # 保存权重
        wpath = os.path.join(cfg.save_dir, f"{name}_interval{interval}.pt")
        torch.save(model.state_dict(), wpath)

        # 保存预测值
        np.savez(
            os.path.join(cfg.save_dir, f"{name}_interval{interval}_preds.npz"),
            preds=preds, trues=trues
        )

        # 保存训练历史
        hpath = os.path.join(cfg.save_dir, f"{name}_interval{interval}_history.json")
        with open(hpath, "w") as f:
            json.dump(history, f)

        interval_results[name] = metrics
        print(f"  {name:<14} MAE={metrics['MAE']:.6f}  R2={metrics['R2']:.6f}")
        sys.stdout.flush()

    # 保存本 interval 汇总
    spath = os.path.join(cfg.save_dir, f"summary_interval{interval}.json")
    with open(spath, "w") as f:
        json.dump(interval_results, f, indent=2)

    all_results[interval] = interval_results
    print(f"  >> interval {interval} saved to {spath}")
    sys.stdout.flush()


def print_table(all_results, key="MAE"):
    intervals = sorted(all_results.keys())
    models    = list(MODEL_REGISTRY.keys())
    title = "MAE" if key == "MAE" else "R2"

    print(f"\n{'='*90}")
    print(f"  {title} Summary: Intervals {intervals[0]}~{intervals[-1]}")
    print(f"{'='*90}")
    hdr = f"  {'Model':<14}" + "".join(f"  I{i:02d}" for i in intervals)
    print(hdr)
    print("  " + "-" * (14 + 8 * len(intervals)))
    for name in models:
        row = f"  {name:<14}"
        for i in intervals:
            v = all_results[i][name][key]
            row += f"  {v:>6.4f}"
        print(row)


def main():
    cfg = Config()
    print(f"Device: {cfg.get_device()} | intervals: {START}~{END}")

    all_results = {}
    for iv in range(START, END + 1):
        run_one(iv, cfg, all_results)

    # 打印两张汇总表
    print_table(all_results, key="MAE")
    print_table(all_results, key="R2")

    # 保存总汇总
    total_path = os.path.join(cfg.save_dir, "summary_all_intervals.json")
    with open(total_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  All done! Weights + summaries saved to: {cfg.save_dir}")
    print(f"  Total summary: {total_path}")


if __name__ == "__main__":
    main()
