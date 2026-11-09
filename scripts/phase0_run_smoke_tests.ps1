# Runs the remaining Phase 0 smoke tests (one method per process).
# Results land in outputs\phase0_smoke\<method>.json
$ErrorActionPreference = "Continue"
$env:HF_HOME = "D:\Projects\cache\hf"
$env:TORCH_HOME = "D:\Projects\cache\torch"
$env:CLIP_CACHE = "D:\Projects\cache\clip"
$py = "D:\Projects\envs\zsad\Scripts\python.exe"
$root = Split-Path -Parent $PSScriptRoot
$tp = "$root\third_party"
$img = "$tp\AnomalyVFM\test.png"
$s = "$root\scripts\phase0_smoke_test.py"
$log = "$root\outputs\phase0_smoke\run_log.txt"
New-Item -ItemType Directory -Force "$root\outputs\phase0_smoke" | Out-Null

"### anomalyvfm_radio" | Tee-Object -FilePath $log
& $py $s --method anomalyvfm_radio --image $img 2>&1 | Tee-Object -FilePath $log -Append
"### anomalyclip" | Tee-Object -FilePath $log -Append
& $py $s --method anomalyclip --image $img --checkpoint "$tp\AnomalyCLIP\checkpoints\9_12_4_multiscale_visa\epoch_15.pth" 2>&1 | Tee-Object -FilePath $log -Append
"### visualad" | Tee-Object -FilePath $log -Append
& $py $s --method visualad --image $img --checkpoint "$tp\VisualAD\weight\train_on_visa\CLIP.pth" 2>&1 | Tee-Object -FilePath $log -Append
"Done. Log: $log"
