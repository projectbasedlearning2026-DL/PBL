# run_all.ps1 - run several datasets x seeds in one go; every run gets its own folder and log.
# Runs that already have a metrics.csv are skipped, so you can stop and restart safely.
#
#   .\run_all.ps1                                               # all 3 datasets, seeds 42-46, 100 series, default model
#   .\run_all.ps1 -Datasets m5 -Seeds 42,43,44 -Tag dense -Extra "--topk_ratio","1.0"
#   .\run_all.ps1 -Datasets m5 -MaxSeries 300 -Tag n300
#
# Results: runs\<dataset>_n<series>_<tag>_s<seed>\   Logs: runs\logs\
param(
    [string[]]$Datasets = @("m5", "walmart", "favorita"),
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [int]$MaxSeries = 100,
    [string]$Tag = "default",
    [string[]]$Extra = @(),
    [string]$OutRoot = "runs"
)

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# dataset-specific settings (same as the validated single runs)
$base = @{
    "m5"       = @("--source", "m5", "--data_dir", "m5", "--epochs", "8")
    "walmart"  = @("--source", "walmart", "--data_dir", "walmart", "--epochs", "30",
                   "--freq", "W-FRI", "--season", "52", "--horizon", "4", "--encoder_len", "26")
    "favorita" = @("--source", "favorita", "--data_dir", "favorita", "--epochs", "8", "--horizon", "16")
}

New-Item -ItemType Directory -Force "$OutRoot\logs" | Out-Null
$failed = @()

foreach ($ds in $Datasets) {
    if (-not $base.ContainsKey($ds)) { Write-Host "Unknown dataset '$ds' (use m5, walmart, favorita)"; continue }
    foreach ($s in $Seeds) {
        $name = "${ds}_n${MaxSeries}_${Tag}_s${s}"
        $out  = "$OutRoot\$name"
        $log  = "$OutRoot\logs\$name.log"
        if (Test-Path "$out\metrics.csv") { Write-Host "SKIP $name (already done)"; continue }
        Write-Host "`n>>> $name  ($(Get-Date -Format 'HH:mm:ss'))"
        $argList = @("-u", "train.py") + $base[$ds] + @("--max_series", "$MaxSeries", "--seed", "$s", "--out", $out) + $Extra
        & python @argList 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$out\metrics.csv")) {
            Write-Host "FAILED: $name (see $log)"
            $failed += $name
        }
    }
}

Write-Host "`nDone at $(Get-Date -Format 'HH:mm:ss')."
if ($failed.Count -gt 0) { Write-Host "Failed runs: $($failed -join ', ')" } else { Write-Host "All runs finished." }
