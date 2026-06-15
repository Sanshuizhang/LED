"""跑 CNN1DDiff 的 interval=1~12"""
import subprocess, json, os, sys, re

results = []
for interval in range(1, 13):
    print(f"\n{'='*60}")
    print(f"  CNN1DDiff | interval={interval}")
    print(f"{'='*60}")
    
    r = subprocess.run(
        [sys.executable, "train.py", "--model", "CNN1DDiff", "--interval", str(interval)],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    
    # 从输出提取指标（分行的 MAE: xxx 格式）
    metrics = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        m = re.match(r"MAE:\s*([\d.]+)", line)
        if m: metrics["MAE"] = float(m.group(1))
        m = re.match(r"RMSE:\s*([\d.]+)", line)
        if m: metrics["RMSE"] = float(m.group(1))
        m = re.match(r"MAPE:\s*([\d.]+)", line)
        if m: metrics["MAPE"] = float(m.group(1))
        m = re.match(r"R2:\s*([-\d.]+)", line)
        if m: metrics["R2"] = float(m.group(1))
    
    metrics["interval"] = interval
    results.append(metrics)
    print(f"  I{interval:02d}  MAE={metrics.get('MAE','?')}  R2={metrics.get('R2','?')}")

# 保存
os.makedirs("results", exist_ok=True)
with open("results/summary_cnn1ddiff_all.json", "w") as f:
    json.dump(results, f, indent=2)

# 打印汇总
print(f"\n{'='*60}")
print(f"  CNN1DDiff Summary: Intervals 1~12")
print(f"{'='*60}")
print(f"  {'Interval':>8}  {'MAE':>10}  {'RMSE':>10}  {'MAPE(%)':>8}  {'R2':>10}")
print(f"  {'-'*52}")
for r in results:
    print(f"  {r['interval']:>8}  {r['MAE']:>10.6f}  {r['RMSE']:>10.6f}  {r['MAPE']:>8.2f}  {r['R2']:>10.6f}")

# 对比 PEGRU
pegru_path = "results/summary_all_intervals.json"
if os.path.exists(pegru_path):
    with open(pegru_path) as f:
        all_data = json.load(f)
    print(f"\n{'='*60}")
    print(f"  CNN1DDiff vs PEGRU (MAE)")
    print(f"{'='*60}")
    print(f"  {'Interval':>8}  {'CNN1DDiff':>10}  {'PEGRU':>10}  {'Delta':>10}")
    print(f"  {'-'*44}")
    for r in results:
        iv = str(r["interval"])
        if iv in all_data and "PEGRU" in all_data[iv]:
            p_mae = all_data[iv]["PEGRU"]["MAE"]
            print(f"  {iv:>8}  {r['MAE']:>10.6f}  {p_mae:>10.6f}  {r['MAE']-p_mae:>10.6f}")

print(f"\n  Results saved to: results/summary_cnn1ddiff_all.json")
