#!/usr/bin/env bash
# 夜间实验总调度: baseline val -> tune(封顶90min) -> parse -> final train -> final val -> summary
# 只用一块 GPU（物理卡 1），绝不触碰卡 0。任一步失败都继续往下走，保证有模型产出。
set -u
cd /workspaces/huawei

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

mkdir -p logs
LOG=logs/night_run.log

log() { echo "[$(date '+%F %T')] $*"; }

{
  log "=== NIGHT RUN START ==="
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  log "--- step1: baseline val ---"
  python3 experiments/night_baseline_val.py || log "step1 FAILED"

  log "--- step2: tuning (cap 90min) ---"
  timeout 5400 python3 experiments/night_tune.py || log "step2 ended (timeout/error)"

  # tune 若被 timeout 中断，可能残留 Ray 子进程占卡；强制清理，保证正式训练能拿到 GPU
  log "--- step2b: cleanup GPU after tune ---"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f "ray::ImplicitFunc.train" >/dev/null 2>&1 || true
  sleep 8

  log "--- step3: parse best hyperparameters ---"
  python3 experiments/night_parse_best.py || log "step3 FAILED (will use fallback)"

  log "--- step4: final training ---"
  python3 experiments/night_final_train.py || log "step4 FAILED"

  log "--- step5: final val ---"
  python3 experiments/night_final_val.py || log "step5 FAILED"

  log "--- step6: summary ---"
  python3 experiments/night_summary.py || log "step6 FAILED"

  log "=== NIGHT RUN DONE ==="
} >> "$LOG" 2>&1
