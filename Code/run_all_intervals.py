"""
run_all_intervals.py — 批量运行 interval=1~12，保存所有模型权重和结果汇总

用法：
  python run_all_intervals.py              # 跑全部 interval 1~12
  python run_all_intervals.py --start 6   # 从 interval=6 开始跑
  python run_all_intervals.py --end 5     # 只跑 interval 1~5
"""
import os
import json
import argparse
import numpy as np
import torch
from config import Config
from dataset import build_dataloaders
from models import MODEL_REGISTRY, build_model
from train import train_model, test_model, NO_TRAIN_MODELS


def run_interval(interval: int, cfg: Config, all_results: dict, log_fh=None):
    tag = f"[I{interval:02d}]"
    print(f"\n{'#'*65}")
    print(f"  STARTING {tag}")
    print(f"{'#'*65}")

    cfg.interval = interval
    device = cfg.get_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, scaler_params = build_dataloaders(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model_names = list(MODEL_REGISTRY.keys())
    interval_results = {}

    for name in model_names:
        if name in NO_TRAIN_MODELS:
            print(f"  {tag} Evaluating (no-train): {name}")
            model = build_model(name, cfg).to(device)
            history = {"train": [], "val": []}
        else:
            # 抑制 train_model 的 epoch 输出（可选）
            model, history = train_model(name, cfg, train_loader, val_loader, device)

        metrics, preds, trues = test_model(
            model, name, cfg.seq_len,
            test_loader, scaler_params, device
        )

        # 保存模型权重
        weight_path = os.path.join(cfg.save_dir, f"{name}_interval{interval}.pt")
        torch.save(model.state_dict(), weight_path)

        # 保存预测结果
        np.savez(
            os.path.join(cfg.save_dir, f"{name}_interval{interval}_preds.npz"),
            preds=preds, trues=trues
        )

        # 保存训练曲线
        hist_path = os.path.join(cfg.save_dir, f"{name}_interval{interval}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

        interval_results[name] = metrics
        print(f"  {tag} {name:<16} MAE={metrics['MAE']:.6f}  R2={metrics['R2']:.6f}")

    # 保存本 interval 汇总
    summary_path = os.path.join(cfg.save_dir, f"summary_interval{interval}.json")
    with open(summary_path, "w") as f:
        json.dump(interval_results, f, indent=2)

    all_results[interval] = interval_results
    print(f"  {tag} DONE — results saved to {summary_path}")


def print_summary_table(all_results: dict):
    """打印所有 interval 的 MAE 汇总表"""
    intervals = sorted(all_results.keys())
    models = list(MODEL_REGISTRY.keys())

    print(f"\n{'='*90}")
    print(f"  MAE Summary: All Intervals (1~{max(intervals)})")
    print(f"{'='*90}")

    # 表头
    hdr = f"  {'Model':<16}" + "".join(f" I{i:02d}" for i in intervals)
    print(hdr)
    print("  " + "-" * (16 + 6 * len(intervals)))

    for name in models:
        row = f"  {name:<16}"
        for i in intervals:
            mae = all_results[i][name]["MAE"]
            row += f" {mae:>6.4f}"
        print(row)

    # R2 汇总表
    print(f"\n{'='*90}")
    print(f"  R2 Summary: All Intervals (1~{max(intervals)})")
    print(f"{'='*90}")

    hdr = f"  {'Model':<16}" + "".join(f" I{i:02d}" for i in intervals)
    print(hdr)
    print("  " + "-" * (16 + 6 * len(intervals)))

    for name in models:
        row = f"  {name:<16}"
        for i in intervals:
            r2 = all_results[i][name]["R2"]
            row += f" {r2:>6.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1,
                        help="起始 interval（默认1）")
    parser.add_argument("--end", type=int, default=12,
                        help="结束 interval（默认12）")
    args = parser.parse_args()

    cfg = Config()
    device = cfg.get_device()
    print(f"Device: {device} | intervals: {args.start}~{args.end}")

    all_results = {}
    for interval in range(args.start, args.end + 1):
        run_interval(interval, cfg, all_results)

    # 打印汇总表
    print_summary_table(all_results)

    # 保存总汇总 JSON
    total_path = os.path.join(cfg.save_dir, "summary_all_intervals.json")
    with open(total_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  All results saved to: {cfg.save_dir}")
    print(f"  Total summary: {total_path}")


if __name__ == "__main__":
    main()
