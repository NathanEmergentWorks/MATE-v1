# ============================================
# Parallel PPO Training Launcher for MATE v1
# 8 seeds at a time, with timestamps + ETA
# ============================================

$agentConfig = "configs/agent_ppo.yaml"
$envC0 = "configs/env_c0.yaml"
$envC1 = "configs/env_c1.yaml"

function Timestamp {
    return (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

function Run-SeedBatch {
    param(
        [string]$Condition,
        [string]$EnvConfig,
        [int[]]$Seeds,
        [int]$BatchNumber,
        [int]$TotalBatches
    )

    Write-Host ""
    Write-Host "[$(Timestamp)] Starting batch $BatchNumber of $TotalBatches for $Condition" -ForegroundColor Cyan
    Write-Host "Seeds: $($Seeds -join ', ')" -ForegroundColor DarkCyan

    $startTime = Get-Date

    foreach ($seed in $Seeds) {
        Start-Process -NoNewWindow -FilePath "python" `
            -ArgumentList "-m mate.training.train --env-config $EnvConfig --agent-config $agentConfig --condition $Condition"
        Start-Sleep -Milliseconds 300
    }

    Write-Host "[$(Timestamp)] Waiting for batch $BatchNumber to finish..." -ForegroundColor Yellow
    Wait-Process -Name "python"

    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Host "[$(Timestamp)] Batch $BatchNumber complete. Duration: $($duration.ToString())" -ForegroundColor Green
}

function Split-IntoBatches {
    param([int[]]$Seeds, [int]$BatchSize)

    $batch = @()
    foreach ($s in $Seeds) {
        $batch += $s
        if ($batch.Count -eq $BatchSize) {
            ,$batch
            $batch = @()
        }
    }
    if ($batch.Count -gt 0) { ,$batch }
}

# ============================
# Run C0
# ============================

$C0Seeds = 0..19
$C0Batches = Split-IntoBatches -Seeds $C0Seeds -BatchSize 8
$batchIndex = 1

foreach ($batch in $C0Batches) {
    Run-SeedBatch -Condition "C0" -EnvConfig $envC0 -Seeds $batch -BatchNumber $batchIndex -TotalBatches $C0Batches.Count
    $batchIndex++
}

# ============================
# Run C1
# ============================

$C1Seeds = 0..19
$C1Batches = Split-IntoBatches -Seeds $C1Seeds -BatchSize 8
$batchIndex = 1

foreach ($batch in $C1Batches) {
    Run-SeedBatch -Condition "C1" -EnvConfig $envC1 -Seeds $batch -BatchNumber $batchIndex -TotalBatches $C1Batches.Count
    $batchIndex++
}

Write-Host ""
Write-Host "[$(Timestamp)] All training complete." -ForegroundColor Magenta
