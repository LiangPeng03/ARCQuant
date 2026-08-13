#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
SUMMARY="$LOG_DIR/arcquant_mrgptq_protocol.tsv"
mkdir -p "$ROOT/models" "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/kernels/build:$ROOT/model${PYTHONPATH:+:$PYTHONPATH}"

# Existing HF-cache snapshots; symlinks consume no additional model storage.
ln -sfn \
  /home/pengliang/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
  "$ROOT/models/llama-3-8b"

ln -sfn \
  /home/pengliang/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  "$ROOT/models/llama-3.2-1b"

if [[ ! -f "$SUMMARY" ]]; then
  printf 'timestamp\tmodel\twikitext2_ppl\tc4_ppl\tpiqa\twinogrande\tboolq\tarc_challenge\thellaswag\tmmlu\tarc_easy\tlambada_openai\tgsm8k\tmbpp\traw_log\n' > "$SUMMARY"
fi

ensure_calibration() {
  local model_path="$1"
  local model_id="$2"
  local prefix="$ROOT/saved/${model_id}_"

  if [[ -f "${prefix}reorder_index_c4_max.pt" &&
        -f "${prefix}select_num_c4_max.pt" &&
        -f "${prefix}average_bits_c4_max.pt" ]]; then
    echo "[$model_id] C4 calibration: reuse existing saved statistics."
    return
  fi

  echo "[$model_id] C4 calibration: starting."
  python "$ROOT/reorder_indices.py" \
    --model "$model_path" \
    --dataset c4 \
    --samples 128 \
    --seqlen 2048 \
    --act_sort_metric max
}

evaluate() {
  local model_path="$1"
  local model_id="$2"
  local stamp raw_log

  stamp="$(date +%Y%m%d_%H%M%S)"
  raw_log="$LOG_DIR/${model_id}_arcquant_${stamp}.full.log"

  echo "[$model_id] ARCQuant NVFP4 evaluation: starting. Raw log: $raw_log"

  if ! python "$ROOT/model/evaluate_mrgptq_protocol.py" \
      --model "$model_path" \
      --mode arcquant \
      --dtype bfloat16 \
      --calib-dataset c4 \
      --act-sort-metric max \
      --seed 0 \
      --tasks piqa winogrande boolq arc_challenge hellaswag mmlu arc_easy lambada_openai gsm8k mbpp \
      > "$raw_log" 2>&1; then
    echo "[$model_id] FAILED; last 80 lines:"
    tail -80 "$raw_log"
    return 1
  fi

  python - "$raw_log" "$SUMMARY" "$stamp" "$model_id" <<'PY'
import json
import sys

raw_log, summary, stamp, model_id = sys.argv[1:]
line = next(
    (item.split(" ", 1)[1] for item in reversed(open(raw_log, encoding="utf-8").read().splitlines())
     if item.startswith("[RESULT_SUMMARY] ")),
    None,
)
if line is None:
    raise SystemExit(f"No RESULT_SUMMARY in {raw_log}")

data = json.loads(line)
r = data["results"]

def fmt(name):
    value = r.get(name)
    return "NA" if value is None else f"{value:.6f}"

row = [
    stamp, model_id,
    f"{data['wikitext2_ppl']:.6f}",
    f"{data['c4_ppl']:.6f}",
    *[fmt(name) for name in (
        "piqa", "winogrande", "boolq", "arc_challenge", "hellaswag",
        "mmlu", "arc_easy", "lambada_openai", "gsm8k", "mbpp",
    )],
    raw_log,
]
with open(summary, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print("\nRESULT:", "\t".join(row[:-1]))
PY
}

ensure_calibration "$ROOT/models/llama-3-8b" "llama-3-8b"
evaluate "$ROOT/models/llama-3-8b" "llama-3-8b"

ensure_calibration "$ROOT/models/llama-3.2-1b" "llama-3.2-1b"
evaluate "$ROOT/models/llama-3.2-1b" "llama-3.2-1b"

echo
echo "Completed. Concise summary:"
column -ts $'\t' "$SUMMARY"
