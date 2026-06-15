"""跑 CNN1DDiffGated 的 interval=1~12，对比 CNN1D"""
import subprocess, json, os, sys, re

results = []
for interval in range(1, 13):
    print(f"\n{'='*60}")
    print(f"  CNN1DDiffGated | interval={interval}")
    print(f"{'='*60}")
    
    r = subprocess.run(
        [sys.executable, "train.py", "--model", "CNN1DDiffGated", "--interval", str(interval)],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    
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
with open("results/summary_cnn1ddiff_gated_all.json", "w") as f:
    json.dump(results, f, indent=2)

# 打印汇总
print(f"\n{'='*60}")
print(f"  CNN1DDiffGated Summary")
print(f"{'='*60}")
print(f"  {'I':>4}  {'MAE':>10}  {'RMSE':>10}  {'MAPE%':>8}  {'R2':>10}")
print(f"  {'-'*48}")
for r in results:
    print(f"  {r['interval']:>4}  {r['MAE']:>10.6f}  {r['RMSE']:>10.6f}  {r['MAPE']:>8.2f}  {r['R2']:>10.6f}")

# 对比 CNN1D
cnn1d_path = "results/summary_all_intervals.json"
if os.path.exists(cnn1d_path):
    with open(cnn1d_path) as f:
        all_data = json.load(f)
    print(f"\n{'='*60}")
    print(f"  CNN1DDiffGated vs CNN1D (MAE)")
    print(f"{'='*60}")
    print(f"  {'I':>4}  {'Gated':>10}  {'CNN1D':>10}  {'Delta':>10}  {'Win':>6}")
    print(f"  {'-'*46}")
    gated_wins = 0
    cnn1d_wins = 0
    for r in results:
        iv = str(r["interval"])
        if iv in all_data and "CNN1D" in all_data[iv]:
            c_mae = all_data[iv]["CNN1D"]["MAE"]
            delta = r["MAE"] - c_mae
            w = "Gated" if delta < 0 else "CNN1D" if delta > 0 else "Tie"
            if delta < 0: gated_wins += 1
            elif delta > 0: cnn1d_wins += 1
            print(f"  {iv:>4}  {r['MAE']:>10.6f}  {c_mae:>10.6f}  {delta:>10.6f}  {w:>6}")
    print(f"\n  Gated wins: {gated_wins}  |  CNN1D wins: {cnn1d_wins}  |  Ties: {12-gated_wins-cnn1d_wins}")

# 对比 CNN1DDiff（无门控）
diff_path = "results/summary_cnn1ddiff_all.json"
if os.path.exists(diff_path):
    with open(diff_path) as f:
        diff_data = json.load(f)
    print(f"\n{'='*60}")
    print(f"  CNN1DDiffGated vs CNN1DDiff (MAE)")
    print(f"{'='*60}")
    print(f"  {'I':>4}  {'Gated':>10}  {'NoGate':>10}  {'Delta':>10}  {'Win':>6}")
    print(f"  {'-'*46}")
    for r in results:
        iv = r["interval"]
        d_mae = [d["MAE"] for d in diff_data if d["interval"]==iv]
        if d_mae:
            d_mae = d_mae[0]
            delta = r["MAE"] - d_mae
            w = "Gated" if delta < 0 else "NoGate" if delta > 0 else "Tie"
            print(f"  {iv:>4}  {r['MAE']:>10.6f}  {d_mae:>10.6f}  {delta:>10.6f}  {w:>6}")

print(f"\n  Results saved to: results/summary_cnn1ddiff_gated_all.json")
