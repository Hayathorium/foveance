# Self-restarting monitor for the full 6-arm x 3-model trajectory benchmark on CPU Ollama.
# Ollama hangs run_bench's socket after many calls; this watches Ollama's CPU (which keeps rising
# while it is genuinely computing, even a slow 57s call) and, when it goes flat for several minutes,
# treats it as a stall: it kills the hung run_bench, restarts Ollama, and relaunches with --resume.
# Runs until by_seed.csv reaches the target row count. Logs to scripts/full6_monitor.log.

$ErrorActionPreference = "SilentlyContinue"
$root   = Split-Path $PSScriptRoot -Parent
$py     = "C:\Users\Dell\AppData\Local\Programs\Python\Python312\python.exe"
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$bySeed = Join-Path $root "bench\results\by_seed.csv"
$log    = Join-Path $root "scripts\full6_monitor.log"
$target  = 270   # 6 arms x 3 models x 3 budgets x 5 seeds
$runArgs = @('bench\run_bench.py','--backend','ollama',
            '--models','llama3.2:1b,qwen2.5:1.5b,gemma2:2b',
            '--budgets','400,700,1200','--tasks','5','--turns','8',
            '--n-facts','3','--block-lines','18','--drift','0.7',
            '--arms','full,recency,truncate,uniform,reactive_afm,foveance',
            '--outdir','bench\results')

function Log($m) { "$((Get-Date).ToString('HH:mm:ss'))  $m" | Tee-Object -FilePath $log -Append }
function Rows { try { (Import-Csv $bySeed).Count } catch { 0 } }
function OllamaCpu { try { (Get-Process ollama | Measure-Object CPU -Sum).Sum } catch { 0 } }

Set-Location $root
Log "monitor start; target=$target rows; current=$(Rows)"

$round = 0
while ((Rows) -lt $target) {
    $round++
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_bench' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-Process ollama | Stop-Process -Force
    Start-Sleep 4
    Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep 6
    $p = Start-Process -FilePath $py -ArgumentList $runArgs -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $root "scripts\full6_run.out") `
            -RedirectStandardError  (Join-Path $root "scripts\full6_run.err")
    Log "round ${round}: launched run_bench pid $($p.Id); rows=$(Rows)"

    $prevCpu = OllamaCpu; $idle = 0
    while (-not $p.HasExited -and (Rows) -lt $target) {
        Start-Sleep 60
        $cpu = OllamaCpu
        if (($cpu - $prevCpu) -lt 3) { $idle++ } else { $idle = 0 }
        $prevCpu = $cpu
        if ($idle -ge 4) { Log "round ${round}: Ollama idle ~4min (stall) at rows=$(Rows); restarting"; break }
    }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    else { Log "round ${round}: run_bench exited; rows=$(Rows)" }
    Start-Sleep 3
}
Log "DONE: rows=$(Rows) reached target=$target"
