"""
run_msctcn.py — 批量训练 MSCTCN interval=1~12
"""
import os
import json
import sys
import numpy as np
import torch
from config import Config
from dataset import build_dataloaders
from models import build_model
from train import train_model, test_model

START = 1
END   = 12


def run_one(interval, cfg):
    print(f"\n{'#'*60}")
    print(f"  [MSCTCN] INTERVAL = {interval}")
    print(f"{'#'*60}")
    sys.stdout.flush()

    cfg.interval = interval
    device = cfg.get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, scaler_params = build_dataloaders(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model, history = train_model("MSCTCN", cfg, train_loader, val_loader, device)
    metrics, preds, trues = test_model(
        model, "MSCTCN", cfg.seq_len, test_loader, scaler_params, device
    )

    # 保存权重
    wpath = os.path.join(cfg.save_dir, f"MSCTCN_interval{interval}.pt")
    torch.save(model.state_dict(), wpath)

    # 保存预测值
    np.savez(
        os.path.join(cfg.save_dir, f"MSCTCN_interval{interval}_preds.npz"),
        preds=preds, trues=trues
    )

    # 保存训练历史
    hpath = os.path.join(cfg.save_dir, f"MSCTCN_interval{interval}_history.json")
    with open(hpath, "w") as f:
        json.dump(history, f)

    print(f"  [MSCTCN] I{interval:02d}  MAE={metrics['MAE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
          f"MAPE={metrics['MAPE']:.4f}%  R2={metrics['R2']:.6f}")
    sys.stdout.flush()
    return metrics


def main():
    cfg = Config()
    print(f"Device: {cfg.get_device()} | MSCTCN intervals: {START}~{END}")

    results = {}
    for iv in range(START, END + 1):
        results[iv] = run_one(iv, cfg)

    # 汇总表
    print(f"\n{'='*70}")
    print(f"  MSCTCN Summary: Intervals {START}~{END}")
    print(f"{'='*70}")
    print(f"  {'Interval':>8} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'R2':>10}")
    print(f"  " + "-"*50)
    for iv in range(START, END + 1):
        m = results[iv]
        print(f"  {iv:>8} {m['MAE']:>10.6f} {m['RMSE']:>10.6f} "
              f"{m['MAPE']:>10.4f} {m['R2']:>10.6f}")

    # 与 PEGRU 对比（读取已有的 summary）
    print(f"\n{'='*70}")
    print(f"  MSCTCN vs PEGRU (MAE)")
    print(f"{'='*70}")
    print(f"  {'Interval':>8} {'MSCTCN':>10} {'PEGRU':>10} {'Delta':>10}")
    print(f"  " + "-"*40)
    for iv in range(START, END + 1):
        msctcn_mae = results[iv]['MAE']
        pegrupath = os.path.join(cfg.save_dir, f"summary_interval{iv}.json")
        if os.path.exists(pegrupath):
            with open(pegrupath) as f:
                pegr = json.load(f)
            pegr_mae = pegr.get("PEGRU", {}).get("MAE", float('nan'))
            delta = msctcn_mae - pegr_mae
            print(f"  {iv:>8} {msctcn_mae:>10.6f} {pegr_mae:>10.6f} {delta:>10.6f}")
        else:
            print(f"  {iv:>8} {msctcn_mae:>10.6f} {'N/A':>10} {'N/A':>10}")

    # 保存
    out_path = os.path.join(cfg.save_dir, "summary_msctcn_all.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
