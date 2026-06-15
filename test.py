
import torch
import numpy as np
from openpyxl import Workbook# 加载 .npz 文件

data = np.load("./results/CNN1DDiff_interval8_preds.npz")

# 查看里面有哪些数组（最重要！）
print("文件里包含的键：", data.files)
print(data["preds"])
print(data["trues"])

wb = Workbook()
ws = wb.active

# 循环：第i个元素放在 A列 第i+1行（Excel行从1开始）
for row_idx, value in enumerate(data["preds"], start=1):
    ws.cell(row=row_idx, column=1, value=value)

# 保存文件
wb.save("竖排数据.xlsx")
print("写入完成")