"""
models.py — 全部模型定义

所有模型统一接口：
  输入 (batch_size, input_dim) → 输出 (batch_size, 1)
  input_dim = seq_len(14)                若 use_diff_feature=False
  input_dim = seq_len*2-1(27)            若 use_diff_feature=True

包含：
  1. PEGRUPredictor    — 可学习位置编码增强 GRU（本文模型，轻量化版）
  2. MLPPredictor      — 多层感知机
  3. BiGRUPredictor    — 双向 GRU
  4. BiLSTMPredictor   — 双向 LSTM
  5. CNN1DPredictor    — 一维卷积网络
  6. TCNPredictor      — 时间卷积网络
  7. CNNLSTMPredictor  — CNN + LSTM 混合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# ═════════════════════════════════════════════════════════
# 工具模块
# ═════════════════════════════════════════════════════════

class Chomp1d(nn.Module):
    """裁剪右侧填充以保持因果性"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """膨胀因果卷积 + 残差连接"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        return self.relu(out + self.downsample(x))


# ═════════════════════════════════════════════════════════
# 1. PE-GRU（本文模型，轻量化）
# ═════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    """
    可学习位置编码（论文中使用的方式）
    输入  (B, seq_len, d_model)
    输出  (B, seq_len, d_model)
    """
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class ResidualGRUCell(nn.Module):
    """残差 GRUCell：h_out = h_gru + skip(x)"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_x_rz = nn.Linear(input_size,  2 * hidden_size, bias=False)
        self.W_h_rz = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.W_x_n  = nn.Linear(input_size,  hidden_size, bias=False)
        self.W_h_n  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.skip   = nn.Linear(input_size,  hidden_size, bias=False)

    def forward(self, x, h):
        rz = torch.sigmoid(self.W_x_rz(x) + self.W_h_rz(h))
        r, z = rz.chunk(2, dim=-1)
        n = torch.tanh(self.W_x_n(x) + r * self.W_h_n(h))
        h_gru = (1 - z) * n + z * h
        return h_gru + self.skip(x)


class MultiScaleConv(nn.Module):
    """多尺度1D卷积：用 kernel_size=3/5/7 同时提取不同感受野的局部特征"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        ch = out_channels // 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout))
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout))
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        # x: (B, in_ch, T) → (B, out_ch, T)
        return torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)


class PEGRUPredictor(nn.Module):
    """
    PE-GRU 预测器（论文模型，双流 + 多尺度卷积版，无PE无Attention）

    路径一（原始亮度流，CNN增强）：
      raw(B,14) → MultiScaleConv(k=3,5,7) → ResidualGRUCell×num_layers → 最后时步隐状态 → context(B,hidden)

    路径二（差分速率流，独立2层标准GRU）：
      diff(B,13) → nn.GRU(1, hidden, 2layers) → 最后时步隐状态 → diff_out(B,hidden)
      仅当 input_dim > seq_len（即 use_diff_feature=True）时启用

    路径三（TCN 跳跃连接，基于原始亮度）：
      raw(B,14) → TemporalBlock×2 → early(B,hidden)

    输出：
      concat(context, diff_out, early)  → fc → 1
      若无差分：concat(context, early)  → fc → 1
    """
    def __init__(self, seq_len=14, input_dim=14, input_size=1, d_model=16,
                 hidden_size=32, num_layers=1, dropout=0.3, **kw):
        super().__init__()
        self.seq_len     = seq_len
        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.has_diff    = (input_dim > seq_len)
        self.diff_len    = input_dim - seq_len   # 13 or 0

        # ── 路径一：原始亮度 → MultiScaleConv + ResidualGRU（无PE无Attention）──
        self.ms_conv     = MultiScaleConv(1, d_model, dropout=dropout)
        self.gru_cells   = nn.ModuleList([
            ResidualGRUCell(d_model if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.drop        = nn.Dropout(dropout)

        # ── 路径二：差分速率 → 独立单层标准GRU ──
        if self.has_diff:
            self.diff_gru = nn.GRU(
                input_size  = 1,
                hidden_size = hidden_size,
                num_layers  = 1,
                batch_first = True,
            )
            self.diff_ln = nn.LayerNorm(hidden_size)

        # ── 路径三：CNN1D 残差跳跃连接（替换TCN）──
        # Block1: (B,1,14) → (B, hidden_size, 14)
        self.skip_cnn1 = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Block2: (B, hidden_size, 14) → (B, hidden_size, 14)
        self.skip_cnn2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
        )
        # 残差投影：当 in_ch=1 ≠ out_ch=hidden_size 时需要1x1卷积
        self.skip_residual = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.skip_ln = nn.LayerNorm(hidden_size)

        # ── 输出 MLP ──
        merge_dim = hidden_size * 3 if self.has_diff else hidden_size * 2
        self.fc1  = nn.Linear(merge_dim, hidden_size)
        self.bn1  = nn.BatchNorm1d(hidden_size)
        self.fc2  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, input_dim)
        B   = x.size(0)
        raw = x[:, :self.seq_len]        # (B, 14)

        # ── 路径一：原始亮度（CNN增强，无PE无Attention）──
        inp = self.ms_conv(raw.unsqueeze(1))        # (B, 1, 14) → (B, d_model, 14)
        inp = inp.permute(0, 2, 1)                  # (B, 14, d_model)
        h   = [torch.zeros(B, self.hidden_size, device=x.device)
               for _ in range(self.num_layers)]
        for t in range(self.seq_len):
            inp_t = inp[:, t, :]
            for li in range(self.num_layers):
                h[li] = self.gru_cells[li](inp_t, h[li])
                h[li] = self.layer_norms[li](h[li])
                inp_t = self.drop(h[li])
        context = h[-1]                                # 最后时步隐状态 → (B, hidden)

        # ── 路径二：差分速率（独立双层GRU）──
        if self.has_diff:
            diff      = x[:, self.seq_len:]             # (B, 13)
            diff_inp  = diff.unsqueeze(-1)              # (B, 13, 1)
            _, diff_h = self.diff_gru(diff_inp)         # diff_h: (2, B, hidden)
            diff_out  = self.diff_ln(diff_h[-1])        # 取最后层 → (B, hidden)
            diff_out  = self.drop(diff_out)

        # ── 路径三：CNN1D 残差跳跃连接 ──
        skip_inp = raw.unsqueeze(1)                      # (B, 1, 14)
        skip_main = self.skip_cnn1(skip_inp)             # (B, hidden, 14)
        skip_main = self.skip_cnn2(skip_main)             # (B, hidden, 14)
        skip_res  = self.skip_residual(skip_inp)          # (B, hidden, 14) 1x1投影
        skip_out  = F.relu(self.skip_ln((skip_main + skip_res).permute(0, 2, 1).mean(dim=1)))
        # (B, hidden, 14) → 残差相加 → permute → mean pool → (B, hidden)

        # ── 合并 & 输出 ──
        if self.has_diff:
            merged = torch.cat([context, diff_out, skip_out], dim=-1)  # (B, hidden*3)
        else:
            merged = torch.cat([context, skip_out], dim=-1)            # (B, hidden*2)

        out = self.drop(F.relu(self.bn1(self.fc1(merged))))
        out = self.fc2(out)                              # (B, 1)
        return out


# ═════════════════════════════════════════════════════════
# 2. MLP
# ═════════════════════════════════════════════════════════

class MLPPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_size=128, dropout=0.2, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ═════════════════════════════════════════════════════════
# 3. GRU
# ═════════════════════════════════════════════════════════

class GRUPredictor(nn.Module):
    def __init__(self, input_dim=14, input_size=1, hidden_size=128,
                 num_layers=1, dropout=0.2, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)                    # (B, input_dim, 1)
        out, _ = self.gru(x)                    # (B, input_dim, hidden*2)
        out = out[:, -1, :]                     # 取最后时间步
        return self.fc(out)


# ═════════════════════════════════════════════════════════
# 4. LSTM
# ═════════════════════════════════════════════════════════

# class LSTMPredictor(nn.Module):
#     def __init__(self, input_dim=14, input_size=1, hidden_size=128,
#                  num_layers=1, dropout=0.2, **kw):
#         super().__init__()
#         self.input_dim = input_dim
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
#                             batch_first=True, bidirectional=False,
#                             dropout=dropout if num_layers > 1 else 0)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         return self.fc(out)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, input_dim=14, hidden_size=32, num_layers=1, dropout=0.2, **kw):
        super().__init__()
        self.hidden_size = hidden_size
        self.FC1 = nn.Linear(1, hidden_size)  # 将        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(hidden_size, hidden_size)
        self.FC2 = nn.Linear(hidden_size, 1)
    def forward(self, x, h0=None, c0=None):
        """
        x: [batch, seq_len, input_size]
        return:
            outputs: [batch, seq_len, hidden_size]  所有时刻h
            h_last:  [batch, hidden_size]           最后时刻h
            c_last:  [batch, hidden_size]           最后时刻c
        """
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)

        x = self.FC1(x)

        # 初始化 h0, c0
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h, c = h0, c0
        outputs = []

        # 手动循环时间步
        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c = self.lstm_cell(x_t, h, c)
            outputs.append(h)


        h = self.FC2(h)
        return h


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入 → 四个门
        self.Wi = nn.Linear(input_size, 4 * hidden_size)
        # 隐状态 → 四个门
        self.Ui = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        手动算一步 LSTM
        x_t:    [batch, input_size]  时刻 t 的输入
        h_prev: [batch, hidden_size] 上一时刻隐状态
        c_prev: [batch, hidden_size] 上一时刻细胞状态
        return: h_new, c_new
        """
        # 一次性算出 4 个门的输出
        gates = self.Wi(x_t) + self.Ui(h_prev)  # [batch, 4*hidden]

        # 拆成 4 个部分：i, f, g, o
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        # 激活
        i_t = torch.sigmoid(i_t)  # 输入门
        f_t = torch.sigmoid(f_t)  # 遗忘门
        g_t = torch.tanh(g_t)     # 候选细胞
        o_t = torch.sigmoid(o_t)  # 输出门

        # 更新细胞状态 C
        c_new = f_t * c_prev + i_t * g_t

        # 更新隐状态 H
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new



# ═════════════════════════════════════════════════════════
# 3. BiGRU
# ═════════════════════════════════════════════════════════

class BiGRUPredictor(nn.Module):
    def __init__(self, input_dim=14, input_size=1, hidden_size=128,
                 num_layers=1, dropout=0.2, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)                    # (B, input_dim, 1)
        out, _ = self.gru(x)                    # (B, input_dim, hidden*2)
        out = out[:, -1, :]                     # 取最后时间步
        return self.fc(out)


# ═════════════════════════════════════════════════════════
# 4. BiLSTM
# ═════════════════════════════════════════════════════════

# class BiLSTMPredictor(nn.Module):
#     def __init__(self, input_dim=14, input_size=1, hidden_size=128,
#                  num_layers=1, dropout=0.2, **kw):
#         super().__init__()
#         self.input_dim = input_dim
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
#                             batch_first=True, bidirectional=True,
#                             dropout=dropout if num_layers > 1 else 0)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         return self.fc(out)


class BiLSTMPredictor(nn.Module):
    def __init__(self, input_dim=14, input_channels=1, cnn_channels=32,
                 hidden_size=32, num_layers=2, dropout=0.2, **kw):
        super().__init__()
        self.hidden_size = hidden_size
        self.FC1 = nn.Linear(1, hidden_size)
        # 正向 LSTM
        self.lstm_forward = LSTMCell(hidden_size, hidden_size)

        # 反向 LSTM
        self.lstm_backward = LSTMCell(hidden_size, hidden_size)
        self.FC2 = nn.Linear(64, 1)
    def forward(self, x, h0=None, c0=None):
        """
        x: [batch, seq_len, input_size]
        return:
            outputs: [batch, seq_len, 2*hidden_size]  双向拼接结果
        """
        B, T = x.shape
        x = x.unsqueeze(-1)
        x = self.FC1(x)

        # ========================
        # 1. 正向传播（从左到右）
        # ========================
        h_f = torch.zeros(B, self.hidden_size, device=x.device)
        c_f = torch.zeros(B, self.hidden_size, device=x.device)
        forward_out = []

        for t in range(T):
            xt = x[:, t, :]
            h_f, c_f = self.lstm_forward(xt, h_f, c_f)
            forward_out.append(h_f)

        forward_out = torch.stack(forward_out, dim=1)  # [B, T, H]

        # ========================
        # 2. 反向传播（从右到左）
        # ========================
        h_b = torch.zeros(B, self.hidden_size, device=x.device)
        c_b = torch.zeros(B, self.hidden_size, device=x.device)
        backward_out = []

        for t in reversed(range(T)):
            xt = x[:, t, :]
            h_b, c_b = self.lstm_backward(xt, h_b, c_b)
            backward_out.append(h_b)

        h = torch.cat([h_f, h_b], dim=-1)   # b, 64
        h = self.FC2(h)

        return h

# ═════════════════════════════════════════════════════════
# 5. 1D-CNN
# ═════════════════════════════════════════════════════════

class CNN1DPredictor(nn.Module):
    def __init__(self, input_dim=14, input_channels=1, hidden_size=128, dropout=0.2, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B, 1, input_dim)       16, 1, 14
        x = self.conv(x)                       # (B, hidden, input_dim)
        x = self.pool(x).squeeze(-1)           # (B, hidden)
        return self.fc(x)

# class CNN1DPredictor(nn.Module):
#     def __init__(self, seq_len=14, input_channels=1,
#                  hidden_size=128, dropout=0.2, **kw):
#         super().__init__()
#         self.seq_len   = seq_len
#         # ── 主路：原始亮度 CNN1D ──
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#         )
#         self.pool = nn.AdaptiveMaxPool1d(1)
#
#         # ── 输出 MLP ──
#         fc_in = hidden_size
#         self.fc = nn.Sequential(
#             nn.Linear(fc_in, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, 1),
#         )
#
#     def forward(self, x):
#         raw = x[:, :self.seq_len]                     # (B, 14)
#
#         # 主路
#         main_feat = self.conv(raw.unsqueeze(1))        # (B, 1, 14) → (B, hidden, 14)
#         main_feat = self.pool(main_feat).squeeze(-1)   # (B, hidden)
#
#         return self.fc(main_feat)

# ═════════════════════════════════════════════════════════
# 5b. CNN1D-Diff（CNN1D + 差分支路）
# ═════════════════════════════════════════════════════════

class CNN1DDiffPredictor(nn.Module):
    """
    CNN1D + 差分双分支预测器

    主路：原始亮度序列 → 3层Conv1D → AdaptiveMaxPool → main_feat(B, hidden)
    差分路：一阶差分序列 → 2层Conv1D → AdaptiveMaxPool → diff_feat(B, hidden)
    输出：concat(main_feat, diff_feat) → FC → 1
    """
    def __init__(self, seq_len=14, input_dim=27, input_channels=1,
                 hidden_size=128, dropout=0.2, **kw):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.has_diff  = (input_dim > seq_len)

        # ── 主路：原始亮度 CNN1D ──
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)


        # ── 差分路：差分 CNN1D ──
        if self.has_diff:
            self.diff_conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )

        # ── 输出 MLP ──
        fc_in = hidden_size * 2 if self.has_diff else hidden_size
        # fc_in = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        raw = x[:, :self.seq_len]                     # (B, 14)

        # 主路
        main_feat = self.conv(raw.unsqueeze(1))        # (B, 1, 14) → (B, hidden, 14)
        main_feat = self.pool(main_feat).squeeze(-1)   # (B, hidden)

        # 差分路
        if self.has_diff:
            diff      = x[:, self.seq_len:]            # (B, 13)
            diff_feat = self.diff_conv(diff.unsqueeze(1))   # (B, 1, 13) → (B, hidden, 13)
            diff_feat = self.pool(diff_feat).squeeze(-1)    # (B, hidden)
            merged    = torch.cat([main_feat, diff_feat], dim=-1)  # (B, hidden*2)
        else:
            merged = main_feat
        # merged = main_feat
        return self.fc(merged)


# ═════════════════════════════════════════════════════════
# 5c. CNN1D-Diff-Gated（CNN1D + 门控差分支路）
# ═════════════════════════════════════════════════════════

class CNN1DDiffGatedPredictor(nn.Module):
    """
    CNN1D + 门控差分双分支预测器

    主路：原始亮度 → 3层Conv1D → MaxPool → main_feat(B, hidden)
    差分路：一阶差分 → 2层Conv1D → MaxPool → diff_feat(B, hidden)
    门控：gate = sigmoid(FC(concat(main, diff)))  → (B, hidden)
          diff_gated = gate ⊙ diff_feat
    输出：concat(main_feat, diff_gated) → FC → 1

    门控机制让模型自动学习差分信号的信任度：
    短间隔/长间隔差分质量差时 gate→0，退化为纯CNN1D
    中间隔差分信息有用时 gate→1，充分利用差分
    """
    def __init__(self, seq_len=14, input_dim=27, input_channels=1,
                 hidden_size=128, dropout=0.2, **kw):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.has_diff  = (input_dim > seq_len)

        # ── 主路：原始亮度 CNN1D ──
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

        # ── 差分路：差分 CNN1D ──
        if self.has_diff:
            self.diff_conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )
            # ── 门控：根据主路+差分路特征决定差分信任度 ──
            self.gate_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid(),
            )

        # ── 输出 MLP ──
        fc_in = hidden_size * 2 if self.has_diff else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        raw = x[:, :self.seq_len]                     # (B, 14)

        # 主路
        main_feat = self.conv(raw.unsqueeze(1))        # (B, 1, 14) → (B, hidden, 14)
        main_feat = self.pool(main_feat).squeeze(-1)   # (B, hidden)

        # 差分路 + 门控
        if self.has_diff:
            diff      = x[:, self.seq_len:]            # (B, 13)
            diff_feat = self.diff_conv(diff.unsqueeze(1))   # (B, 1, 13) → (B, hidden, 13)
            diff_feat = self.pool(diff_feat).squeeze(-1)    # (B, hidden)

            # 门控：自适应调节差分信号强度
            gate      = self.gate_fc(torch.cat([main_feat, diff_feat], dim=-1))  # (B, hidden)
            diff_gated = gate * diff_feat                          # 门控差分

            merged = torch.cat([main_feat, diff_gated], dim=-1)   # (B, hidden*2)
        else:
            merged = main_feat

        return self.fc(merged)

        return self.fc(merged)


# ═════════════════════════════════════════════════════════
# 6. TCN（时间卷积网络）
# ═════════════════════════════════════════════════════════

class TCNPredictor(nn.Module):
    def __init__(self, input_dim=14, input_channels=1, num_channels=None,
                 kernel_size=3, dropout=0.2, **kw):
        super().__init__()
        self.input_dim = input_dim
        if num_channels is None:
            num_channels = [64, 64, 64]
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch  = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B, 1, input_dim)
        out = self.network(x)                   # (B, channels, input_dim)
        out = out[:, :, -1]                     # 取最后时间步
        return self.fc(out)


# ═════════════════════════════════════════════════════════
# 7. CNN-LSTM
# ═════════════════════════════════════════════════════════

class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=14, input_channels=1, cnn_channels=32,
                 hidden_size=128, num_layers=2, dropout=0.2, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B, 1, input_dim)
        x = self.cnn(x)                        # (B, ch, input_dim)
        x = x.permute(0, 2, 1)                 # (B, input_dim, ch)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])                  # (B, 1)


# ═════════════════════════════════════════════════════════
# 8. DilatedConvBlock（膨胀卷积残差块，对称填充）
# ═════════════════════════════════════════════════════════

class DilatedConvBlock(nn.Module):
    """膨胀卷积残差块（对称填充，双向感受野，适合特征提取）

    与 TCN 的 TemporalBlock 区别：
      - 使用对称填充（非因果），利用前后文信息
      - 双层 Conv + BN + 残差 + Dropout
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # 对称填充
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T) → (B, out_ch, T)
        res = self.skip(x)
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.drop(F.relu(out + res))


# ═════════════════════════════════════════════════════════
# 9. MSCTCN（Multi-Scale CNN + Dilated Conv，纯卷积模型）
# ═════════════════════════════════════════════════════════

class MSCTCNPredictor(nn.Module):
    """
    多尺度卷积 + 膨胀卷积融合预测器（纯卷积，无RNN，无残差跳跃）

    路径一：多尺度CNN特征提取 + 膨胀卷积时序建模
      raw(B,1,14) → MultiScaleConv(k=3,5,7) → (B,d_model,14)
                   → DilatedConvBlock(d=1) → DilatedConvBlock(d=2) → GAP → context(B,hidden)

    路径二：差分CNN
      diff(B,1,13) → Conv1D+BN+ReLU → Conv1D+BN+ReLU → GAP → diff_out(B,hidden)

    输出：concat(context, diff_out) → MLP → 1
    """
    def __init__(self, seq_len=14, input_dim=14, input_size=1, d_model=18,
                 hidden_size=32, num_layers=1, dropout=0.3, **kw):
        super().__init__()
        self.seq_len     = seq_len
        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.has_diff    = (input_dim > seq_len)
        self.drop        = nn.Dropout(dropout)

        # ── 路径一：多尺度CNN + 膨胀卷积 ──
        self.ms_conv = MultiScaleConv(1, d_model, dropout=dropout)
        self.dcb1    = DilatedConvBlock(d_model, hidden_size,
                                        kernel_size=3, dilation=1, dropout=dropout)
        self.dcb2    = DilatedConvBlock(hidden_size, hidden_size,
                                        kernel_size=3, dilation=2, dropout=dropout)

        # ── 路径二：差分CNN ──
        if self.has_diff:
            self.diff_cnn = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )

        # ── 输出 MLP ──
        merge_dim = hidden_size * 2 if self.has_diff else hidden_size
        self.fc1  = nn.Linear(merge_dim, hidden_size)
        self.bn1  = nn.BatchNorm1d(hidden_size)
        self.fc2  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, input_dim)
        B   = x.size(0)
        raw = x[:, :self.seq_len]        # (B, 14)

        # ── 路径一：多尺度CNN + 膨胀卷积 ──
        feat = self.ms_conv(raw.unsqueeze(1))   # (B, 1, 14) → (B, d_model, 14)
        feat = self.dcb1(feat)                   # (B, hidden, 14)
        feat = self.dcb2(feat)                   # (B, hidden, 14)
        context = feat.mean(dim=2)               # GAP → (B, hidden)

        # ── 路径二：差分CNN ──
        if self.has_diff:
            diff      = x[:, self.seq_len:]       # (B, 13)
            diff_feat = self.diff_cnn(diff.unsqueeze(1))  # (B, 1, 13) → (B, hidden, 13)
            diff_out  = diff_feat.mean(dim=2)     # GAP → (B, hidden)

        # ── 合并 & 输出 ──
        if self.has_diff:
            merged = torch.cat([context, diff_out], dim=-1)  # (B, hidden*2)
        else:
            merged = context                                  # (B, hidden)

        out = self.drop(F.relu(self.bn1(self.fc1(merged))))
        out = self.fc2(out)                              # (B, 1)
        return out


# ═════════════════════════════════════════════════════════
# 10. LinearExtrapolation（无参数规则 baseline）
# ═════════════════════════════════════════════════════════

class LinearExtrapolation(nn.Module):
    """
    线性外推 baseline（无可训练参数）

    逻辑：
      取输入序列的最后 n_tail 个点（默认5个），用最小二乘法拟合线性斜率，
      然后向前外推 predict_steps 步（默认1步）得到预测值。

    时间轴假设：
      相邻点之间间隔 1 个单位（即 t = 0, 1, ..., n_tail-1），
      外推目标在 t = n_tail 处（再往前1步）。
    """
    def __init__(self, seq_len=14, n_tail=5, predict_steps=1, **kw):
        super().__init__()
        self.seq_len       = seq_len
        self.n_tail        = n_tail
        self.predict_steps = predict_steps

        # 预计算最小二乘权重（固定，不参与梯度）
        # t = [0, 1, ..., n_tail-1]
        t = torch.arange(n_tail, dtype=torch.float32)
        t_mean = t.mean()
        denom  = ((t - t_mean) ** 2).sum()       # Σ(t - t̄)²

        # 斜率权重 w_k：pred_slope = Σ w_k * y_k
        w_slope = (t - t_mean) / denom           # shape (n_tail,)

        # 外推点时刻
        t_pred = float(n_tail - 1 + predict_steps)

        # 截距权重 w_b：pred_intercept = (1/n) * Σ y_k - slope * t̄
        # 最终预测 = slope * t_pred + intercept
        #           = Σ [ w_slope_k * (t_pred - t̄) + 1/n ] * y_k
        w_final = w_slope * (t_pred - t_mean) + 1.0 / n_tail  # (n_tail,)

        # 注册为 buffer（不参与训练，但随模型 .to(device) 移动）
        self.register_buffer("w_final", w_final)

    def forward(self, x):
        # x: (B, input_dim)，只用前 seq_len 维的最后 n_tail 个点
        tail = x[:, :self.seq_len][:, -self.n_tail:]   # (B, n_tail)
        pred = (tail * self.w_final).sum(dim=1, keepdim=True)  # (B, 1)
        return pred


# ═════════════════════════════════════════════════════════
# 模型工厂
# ═════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "MLP":         MLPPredictor,
    "LSTM":        LSTMPredictor,
    "BiLSTM":      BiLSTMPredictor,
    "CNN1D":       CNN1DPredictor,
    "CNN1DDiff":   CNN1DDiffPredictor,
    "TCN":         TCNPredictor,
    "CNNLSTM":     CNNLSTMPredictor,
    "LinearExtrap": LinearExtrapolation,
    "PEGRU":       PEGRUPredictor,
}
# MODEL_REGISTRY = {
#     "PEGRU":       PEGRUPredictor,
#     "MSCTCN":      MSCTCNPredictor,
#     "MLP":         MLPPredictor,
#     "BiGRU":       BiGRUPredictor,
#     "BiLSTM":      BiLSTMPredictor,
#     "CNN1D":       CNN1DPredictor,
#     "CNN1DDiff":   CNN1DDiffPredictor,
#     "CNN1DDiffGated": CNN1DDiffGatedPredictor,
#     "TCN":         TCNPredictor,
#     "CNNLSTM":     CNNLSTMPredictor,
#     "LinearExtrap": LinearExtrapolation,
# }

def build_model(name: str, cfg) -> nn.Module:
    """根据名称构建模型，自动传入 cfg 中的参数
    PEGRU / MSCTCN 使用完整 input_dim（含差分 27 维），其他 baseline 只用 seq_len(14) 维
    """
    cls = MODEL_REGISTRY[name]
    use_diff   = getattr(cfg, "use_diff_feature", False)
    # PEGRU / MSCTCN 用差分特征，baseline 不用
    if name in ("PEGRU", "MSCTCN", "CNN1DDiff", "CNN1DDiffGated"):
        input_dim = cfg.seq_len * 2 - 1 if use_diff else cfg.seq_len
    else:
        input_dim = cfg.seq_len   # baseline 固定 14 维
    return cls(
        seq_len     = cfg.seq_len,
        input_dim   = input_dim,
        hidden_size = cfg.hidden_size,
        num_layers  = cfg.num_layers,
        dropout     = cfg.dropout,
        d_model     = cfg.d_model,
    )
