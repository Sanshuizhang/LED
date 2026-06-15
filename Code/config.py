"""
config.py — 全局配置文件
修改 interval 参数即可切换采样间隔（1~6）
"""

class Config:
    # ===================== 数据 =====================
    data_path   = r".\75组扩充后的数据.xlsx"
    interval    =  1           # 采样间隔因子（1=24h, 2=48h, ..., 6=144h）
    seq_len     = 14          # 输入序列长度（每个样本包含 14 个时序点）

    # 差分特征开关：True → X = concat(原始14, 差分13) = shape(27,)
    use_diff_feature = True

    # ===================== 数据集划分 =====================
    n_train     = 60          # 训练集 LED 数量
    n_val       = 5           # 验证集 LED 数量
    n_test      = 10          # 测试集 LED 数量
    seed        = 40          # 随机种子

    # ===================== 模型通用参数 =====================
    # PEGRU 专用（轻量化）
    d_model     = 18          # 多尺度卷积输出维度（需被3整除，3个尺度各6通道）
    hidden_size = 32          # GRU 隐状态维度（原128→32）
    num_layers  = 1           # GRU 层数（原3→1）
    dropout     = 0.3         # Dropout（原0.2→0.3，更强正则）

    # ===================== 训练 =====================
    batch_size  = 16          # 样本少时用小 batch（原32→16）
    epochs      = 300         # 多一点轮次
    lr          = 5e-4        # 适当降低学习率
    weight_decay= 1e-4        # 稍微加大 L2 正则
    patience    = 30          # Early stopping 耐心值（原20→30）
    grad_clip   = 5.0         # 梯度裁剪阈值

    # ===================== 输出 =====================
    save_dir    = r"..\res_LSTMandBILSTM"

    # ===================== 运行设备 =====================
    @classmethod
    def get_device(cls):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def input_dim(cls):
        """实际输入到模型的特征维度"""
        return cls.seq_len * 2 - 1 if cls.use_diff_feature else cls.seq_len
