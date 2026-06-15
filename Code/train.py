"""
train.py — 训练、评估与结果输出

用法：
  python train.py                 # 训练全部模型
  python train.py --model PEGRU MLP   # 训练指定模型
  python train.py --interval 3      # 覆盖 config 中的 interval
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from dataset import build_dataloaders
from models import MODEL_REGISTRY, build_model

# 无需训练的规则模型（直接跳过训练循环）
NO_TRAIN_MODELS = {"LinearExtrap"}


# ════════════════════════════════════════════════════════
# baseline 模型只用前 seq_len(14) 维，PEGRU 用全部 27 维
# ════════════════════════════════════════════════════════

def get_model_input(model_name: str, X: torch.Tensor, seq_len: int) -> torch.Tensor:
    if model_name in ("PEGRU", "MSCTCN", "CNN1DDiff", "CNN1DDiffGated"):
        return X
    else:
        return X[:, :seq_len]


# ════════════════════════════════════════════════════════
# 训练单个 epoch
# ════════════════════════════════════════════════════════

def train_one_epoch(model, model_name, seq_len, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_in = get_model_input(model_name, X, seq_len)
        optimizer.zero_grad()
        pred = model(X_in)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, model_name, seq_len, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_in = get_model_input(model_name, X, seq_len)
        pred = model(X_in)
        loss = criterion(pred, y)
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / n


# ════════════════════════════════════════════════════════
# 完整训练流程（含 early stopping）
# ════════════════════════════════════════════════════════

def train_model(name: str, cfg: Config, train_loader, val_loader, device):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    model = build_model(name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=5, min_lr=1e-6)

    best_val = float('inf')
    best_state = None
    wait = 0
    history = {"train": [], "val": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, name, cfg.seq_len, train_loader,
                                     criterion, optimizer, device, cfg.grad_clip)
        val_loss   = evaluate(model, name, cfg.seq_len, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        star = "  *" if wait == 0 else ""
        if epoch % 20 == 0 or wait == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.6f}  val={val_loss:.6f}{star}")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val:.6f}")
    return model, history


# ════════════════════════════════════════════════════════
# 测试集评估（反标准化后计算指标）
# ════════════════════════════════════════════════════════

@torch.no_grad()
def test_model(model, model_name, seq_len, test_loader, scaler_params, device):
    model.eval()
    preds, trues = [], []
    for X, y in test_loader:
        X = X.to(device)
        X_in = get_model_input(model_name, X, seq_len)
        pred = model(X_in)
        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())

    preds = np.concatenate(preds, axis=0).flatten()
    trues = np.concatenate(trues, axis=0).flatten()

    # 反标准化
    mean, std = scaler_params["mean"], scaler_params["std"]
    preds_real = preds * std + mean
    trues_real = trues * std + mean

    # 计算指标
    mae  = np.mean(np.abs(preds_real - trues_real))
    rmse = np.sqrt(np.mean((preds_real - trues_real) ** 2))
    mape = np.mean(np.abs((preds_real - trues_real) / (trues_real + 1e-8))) * 100
    ss_res = np.sum((trues_real - preds_real) ** 2)
    ss_tot = np.sum((trues_real - trues_real.mean()) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        "MAE":   round(float(mae),   6),
        "RMSE":  round(float(rmse),  6),
        "MAPE":  round(float(mape),  4),
        "R2":    round(float(r2),    6),
    }
    return metrics, preds_real, trues_real


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["CNN1D"],
                        help="模型名称列表，默认全部")
    parser.add_argument("--interval", type=int, default=None,
                        help="覆盖 config 中的 interval (1~6)")
    args = parser.parse_args()

    cfg = Config()
    if args.interval is not None:
        cfg.interval = args.interval

    device = cfg.get_device()
    print(f"Device: {device}  |  interval: {cfg.interval}")

    # 固定随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)


    # 数据
    train_loader, val_loader, test_loader, scaler_params = build_dataloaders(cfg)

    # 输出目录
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 确定要跑的模型列表
    model_names = args.model if args.model else list(MODEL_REGISTRY.keys())

    # 训练 & 评估
    all_results = {}

    for name in model_names:
        if name in NO_TRAIN_MODELS:
            # ── 无训练规则模型：直接构建 & 评估 ──
            print(f"\n{'='*60}")
            print(f"  Evaluating (no-train): {name}")
            print(f"{'='*60}")
            model = build_model(name, cfg).to(device)
            history = {"train": [], "val": []}
        else:
            model, history = train_model(name, cfg, train_loader, val_loader, device)

        metrics, preds, trues = test_model(model, name, cfg.seq_len,
                                            test_loader, scaler_params, device)

        print(f"\n  [{name}] Test Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        # 打印测试集每个样本的预测值 vs 真实值（反标准化后）
        print(f"\n  [{name}] Test Predictions vs True Values (denormalized):")
        print("  #         Predicted       True        Error")
        print("  " + "-" * 49)
        for i in range(len(preds)):
            err = preds[i] - trues[i]
            print(f"  {i+1:<5} {preds[i]:>12.6f} {trues[i]:>12.6f} {err:>12.6f}")

        # 保存模型权重（规则模型只有 buffer，也可以保存）
        torch.save(model.state_dict(),
                   os.path.join(cfg.save_dir, f"{name}_interval{cfg.interval}.pt"))

        # 保存训练曲线
        hist_path = os.path.join(cfg.save_dir, f"{name}_interval{cfg.interval}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f)

        # 保存预测结果
        np.savez(os.path.join(cfg.save_dir, f"{name}_interval{cfg.interval}_preds.npz"),
                 preds=preds, trues=trues)

        all_results[name] = metrics

    # ── 输出汇总表 ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Results Summary  (interval={cfg.interval})")
    print(f"{'='*70}")
    header = f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'R2':>10}"
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        print(f"{name:<12} {m['MAE']:>10.6f} {m['RMSE']:>10.6f} "
              f"{m['MAPE']:>10.4f} {m['R2']:>10.6f}")

    # 保存汇总表
    summary_path = os.path.join(cfg.save_dir, f"summary_interval{cfg.interval}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {cfg.save_dir}")


main()
