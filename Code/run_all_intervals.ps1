# run_all_intervals.ps1
# 批量运行 interval=1~12，保存所有模型权重和预测结果

$CodeDir = "C:\Users\ASUS\Desktop\论文：LED的寿命预测\Code3"
$saveDir = Join-Path $CodeDir "results"

if (-not (Test-Path $saveDir)) { New-Item -ItemType Directory -Path $saveDir | Out-Null }

$intervals = 1..12
$models    = @("PEGRU","MLP","BiGRU","BiLSTM","CNN1D","TCN","CNNLSTM","LinearExtrap")

$allResults = @{}

foreach ($interval in $intervals) {
    Write-Host "`n============================================================="
    Write-Host "  INTERVAL = $interval"
    Write-Host "============================================================="

    # 为当前 interval 训练/评估所有模型
    $argList = "--interval", "$interval"
    $output = & "python" $CodeDir\train.py @argList 2>&1

    # 解析输出中的指标行（格式：Model   MAE   RMSE   MAPE   R2）
    # 从 results/summary_interval{interval}.json 读取更可靠
    $summaryPath = Join-Path $saveDir "summary_interval$interval.json"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath | ConvertFrom-Json
        $allResults[$interval] = @{}
        foreach ($m in $models) {
            if ($summary.PSObject.Properties.Name -contains $m) {
                $allResults[$interval][$m] = @{
                    MAE  = $summary.$m.MAE
                    RMSE = $summary.$m.RMSE
                    MAPE = $summary.$m.MAPE
                    R2   = $summary.$m.R2
                }
            }
        }
        Write-Host "  Interval $interval done. Models: $($allResults[$interval].Keys -join ', ')"
    } else {
        Write-Host "  WARNING: $summaryPath not found!"
    }
}

# 打印 MAE 汇总表
Write-Host "`n============================================================="
Write-Host "  MAE Summary (all intervals)"
Write-Host "============================================================="
$header = "Model          " + (($intervals | ForEach-Object { " I{0:D2}" -f $_ }) -join "")
Write-Host $header
Write-Host ("-" * $header.Length)

foreach ($m in $models) {
    $row = "{0,-15}" -f $m
    foreach ($i in $intervals) {
        if ($allResults[$i] -and $allResults[$i][$m]) {
            $row += " {0,6:F4}" -f $allResults[$i][$m].MAE
        } else {
            $row += "     -"
        }
    }
    Write-Host $row
}

# 打印 R2 汇总表
Write-Host "`n============================================================="
Write-Host "  R2 Summary (all intervals)"
Write-Host "============================================================="
$header = "Model          " + (($intervals | ForEach-Object { " I{0:D2}" -f $_ }) -join "")
Write-Host $header
Write-Host ("-" * $header.Length)

foreach ($m in $models) {
    $row = "{0,-15}" -f $m
    foreach ($i in $intervals) {
        if ($allResults[$i] -and $allResults[$i][$m]) {
            $row += " {0,6:F4}" -f $allResults[$i][$m].R2
        } else {
            $row += "     -"
        }
    }
    Write-Host $row
}

Write-Host "`nAll done! Results in: $saveDir"
