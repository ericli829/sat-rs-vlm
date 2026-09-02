#!/usr/bin/env bash
# 过夜任务：32 条验收 → 32 张真实消融 → 全量 320。
# 已完成的步骤会跳过。关聊天/断 Cursor 不会停。

set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/autodl-tmp/26揭榜挂帅-太空智算/autodl-tmp/cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export COUNTING_AUTODL_FS=/root/autodl-tmp/26揭榜挂帅-太空智算/autodl-fs
ROOT=/root/autodl-tmp/26揭榜挂帅-太空智算/04_counting_system_plan
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p outputs/logs
STATUS=outputs/logs/STATUS.txt
LOG=outputs/logs/overnight_gpu.log

stamp() { date '+%F %T'; }
note() {
  echo "[$(stamp)] $*" | tee -a "$STATUS"
}

run_step() {
  local name="$1"
  local marker="$2"
  shift 2
  if [[ -f "$marker" ]]; then
    note "SKIP ${name}（已有 ${marker}）"
    return 0
  fi
  note "START ${name}"
  if python3 "$@"; then
    note "DONE ${name} exit=0"
    return 0
  fi
  local rc=$?
  note "FAIL ${name} exit=${rc}"
  return "$rc"
}

note "===== overnight pid=$$ host=$(hostname) ====="
note "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"

run_step "xlrs_benchmark_32" outputs/xlrs_benchmark_32/metrics.json \
  scripts/run_benchmark.py --max-samples 32 --backend auto --out outputs/xlrs_benchmark_32

run_step "experiments_xlrs_32" outputs/experiments_xlrs_32/report.json \
  scripts/run_experiments.py --backend auto --max-samples 32 --gate --out outputs/experiments_xlrs_32

run_step "xlrs_benchmark_320" outputs/xlrs_benchmark_all/metrics.json \
  scripts/run_benchmark.py --max-samples 320 --backend auto --no-overlay --out outputs/xlrs_benchmark_all

note "===== ALL STEPS FINISHED $(stamp) ====="
if [[ -f outputs/xlrs_benchmark_8/metrics.json ]]; then
  note "metrics8=$(python3 -c "import json;d=json.load(open('outputs/xlrs_benchmark_8/metrics.json'));print('exact',d.get('exact_accuracy'),'mae',d.get('mae'),'backend',d.get('backend'))")"
fi
if [[ -f outputs/xlrs_benchmark_32/metrics.json ]]; then
  note "metrics32=$(python3 -c "import json;d=json.load(open('outputs/xlrs_benchmark_32/metrics.json'));print('exact',d.get('exact_accuracy'),'mae',d.get('mae'),'n',d.get('num_samples'),'sec',round(d.get('elapsed_sec',0),1),'gpu',d.get('gpu',{}).get('torch_max_allocated_mb'))")"
fi
if [[ -f outputs/xlrs_benchmark_all/metrics.json ]]; then
  note "metrics320=$(python3 -c "import json;d=json.load(open('outputs/xlrs_benchmark_all/metrics.json'));print('exact',d.get('exact_accuracy'),'mae',d.get('mae'),'n',d.get('num_samples'),'sec',round(d.get('elapsed_sec',0),1),'gpu',d.get('gpu',{}).get('torch_max_allocated_mb'))")"
fi
nvidia-smi >> "$STATUS" 2>/dev/null || true
note "log=$LOG"
