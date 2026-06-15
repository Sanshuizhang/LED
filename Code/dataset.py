"""
dataset.py — LED 时序数据加载与预处理

数据逻辑：
  1. 读取 75组扩充后的数据.xlsx → 75 条 LED 序列（407 个时间点 / 24h 间隔，0~9744h）
  2. 每条 LED 只生成 1 个样本：
     - X_raw = 从 time=0 开始，按 interval 间隔取 seq_len(14) 个点
       例：interval=1 → time 0,1,2,...,13（原始索引 0,1,2,...,13）
           interval=6 → time 0,6,12,...,78（原始索引 0,6,12,...,78）
     - 若 cfg.use_diff_feature=True：
         X = concat(X_raw[14], diff(X_raw)[13])  → shape (27,)
         差分 = X_raw[1:] - X_raw[:-1]（反映衰减速率）
       否则 X = X_raw → shape (14,)
     - Y = time=9744 的数值（原始索引 406，即最后一个时间点）
  3. 75 条 LED 按 60/5/10 划分 train / val / test
  4. 用训练集统计做标准化（zero-mean, unit-variance）
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


class LEDSeriesDataset(Dataset):
    """单个样本：(seq_len,) → (1,)"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ────────────────────────────────────────────────────────
# 核心函数
# ────────────────────────────────────────────────────────

def load_led_data(cfg: Config):
    """
    读取 Excel，返回 dict: {led_name: 1D-np.array}
    每条序列长度为 407（time=0 到 time=9744，步长 24h）
    """
    df = pd.read_excel(cfg.data_path)
    led_cols = [c for c in df.columns if c != "time"]
    data = {}
    for col in led_cols:
        data[col] = df[col].values.astype(np.float64)
    return data


def make_one_sample(series: np.ndarray, interval: int, seq_len: int,
                    use_diff: bool = False):
    """
    从单条 LED 序列生成 1 个样本：
      X_raw = series[0], series[interval], ..., series[(seq_len-1)*interval]  shape(14,)
      若 use_diff=True：
        diff  = X_raw[1:] - X_raw[:-1]          shape(13,)  衰减速率
        X     = concat(X_raw, diff)              shape(27,)
      否则 X = X_raw                             shape(14,)
      Y = series[-1]  （time=9744 的值）
    """
    indices = [i * interval for i in range(seq_len)]  # 0, interval, ..., 13*interval
    X_raw = series[indices]           # (seq_len,)
    Y = series[-1]                    # 最后一个时间点
    if use_diff:
        diff = X_raw[1:] - X_raw[:-1]  # (seq_len-1,) = (13,)
        X = np.concatenate([X_raw, diff])  # (27,)
    else:
        X = X_raw
    return X, Y


def split_led_indices(cfg: Config):
    """
    将 75 条 LED 随机划分为 train / val / test
    返回三个 list: train_indices, val_indices, test_indices
    """
    rng = np.random.RandomState(cfg.seed)
    indices = np.arange(75)
    rng.shuffle(indices)
    train_idx = indices[: cfg.n_train].tolist()
    val_idx   = indices[cfg.n_train : cfg.n_train + cfg.n_val].tolist()
    test_idx  = indices[cfg.n_train + cfg.n_val :].tolist()
    return train_idx, val_idx, test_idx


def build_dataloaders(cfg: Config):
    """
    主入口：构建 train / val / test DataLoader
    返回 (train_loader, val_loader, test_loader, scaler_params)
      scaler_params = {'mean': float, 'std': float}  （训练集统计量）
    """
    # 1) 读取数据
    raw_data = load_led_data(cfg)
    led_names = sorted(raw_data.keys())  # UN1 ~ UN75
    assert len(led_names) == 75, f"Expected 75 LEDs, got {len(led_names)}"

    # 2) 划分 LED
    train_idx, val_idx, test_idx = split_led_indices(cfg)

    # 3) 每条 LED 生成 1 个样本
    use_diff = getattr(cfg, "use_diff_feature", False)

    def collect(indices):
        Xs, ys = [], []
        for idx in indices:
            name = led_names[idx]
            X, y = make_one_sample(raw_data[name], cfg.interval, cfg.seq_len, use_diff)
            Xs.append(X)
            ys.append(y)
        return np.array(Xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    X_train, y_train = collect(train_idx)
    X_val,   y_val   = collect(val_idx)
    X_test,  y_test  = collect(test_idx)

    # 4) 标准化（用训练集统计量）
    mean = X_train.mean()
    std  = X_train.std() + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std
    y_train = (y_train - mean) / std
    y_val   = (y_val   - mean) / std
    y_test  = (y_test  - mean) / std

    scaler_params = {"mean": float(mean), "std": float(std)}

    # 5) 构建 DataLoader
    train_ds = LEDSeriesDataset(X_train, y_train)
    val_ds   = LEDSeriesDataset(X_val,   y_val)
    test_ds  = LEDSeriesDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    feat_dim = X_train.shape[1]
    print(f"[Data] interval={cfg.interval} | "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} | "
          f"feat_dim={feat_dim} (diff={'on' if use_diff else 'off'}) | "
          f"mean={mean:.6f}  std={std:.6f}")

    return train_loader, val_loader, test_loader, scaler_params
