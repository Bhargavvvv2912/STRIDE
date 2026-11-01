#!/bin/bash
#
# Turn the fine-tuned adapter into an Ollama model, qwen14b-aura, quantised to
# Q4_K_M -- the same quantisation as the base ollama qwen2.5-coder:14b, so the
# comparison differs only in the fine-tuning. This Ollama neither quantises
# imported safetensors to K-quants nor loads LoRA adapters, so llama.cpp does
# the conversion: merge -> GGUF f16 -> llama-quantize Q4_K_M -> ollama create
# from the GGUF, with the base model's own template and parameters.
#
#   sbatch scripts/slurm/export_ollama.sbatch
#   NAME=qwen14b-aura-kto sbatch scripts/slurm/export_ollama.sbatch
#
#SBATCH --job-name=aura-export
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/export_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export HF_HOME="$AURA_SCRATCH/hf"
NAME="${NAME:-qwen14b-aura}"          # NAME=qwen14b-aura-kto for the KTO model
RUN="$AURA_SCRATCH/ft/$NAME"
PY="$AURA_SCRATCH/ft-venv/bin/python"
LCPP="$AURA_SCRATCH/llama.cpp"
free_gb() { df -BG --output=avail "$AURA_SCRATCH" | tail -1 | tr -dc 0-9; }

echo "== tools $(date +%H:%M)"
[ -d "$LCPP" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$LCPP"
uv pip install -q -p "$PY" -e "$LCPP/gguf-py"
QUANT="$LCPP/bin/llama-quantize"
if [ ! -x "$QUANT" ]; then
  # "releases/latest" is not a build; take the newest build's plain CPU binaries.
  URL=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5" \
        | "$PY" -c "import json,sys; a=[x['browser_download_url'] for r in json.load(sys.stdin) for x in r['assets'] if x['name'].endswith('-bin-ubuntu-x64.tar.gz')]; print(a[0] if a else '')")
  mkdir -p "$LCPP/bin" "$LCPP/dl"
  if [ -n "$URL" ]; then
    echo "prebuilt: $URL"
    curl -sL "$URL" -o "$LCPP/dl/rel"
    "$PY" -c "
import sys, tarfile, zipfile
f = sys.argv[1]
(zipfile.ZipFile(f) if zipfile.is_zipfile(f) else tarfile.open(f)).extractall(sys.argv[2])" "$LCPP/dl/rel" "$LCPP/dl"
    Q=$(find "$LCPP/dl" -name llama-quantize -type f | head -1)
    [ -n "$Q" ] && cp -r "$(dirname "$Q")"/* "$LCPP/bin/" && chmod +x "$LCPP/bin/"llama-* 2>/dev/null || true
  fi
  if [ ! -x "$QUANT" ]; then
    echo "no prebuilt binary; building llama-quantize"
    uv pip install -q -p "$PY" cmake
    CMAKE="$(dirname "$PY")/cmake"
    "$CMAKE" -S "$LCPP" -B "$LCPP/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
    "$CMAKE" --build "$LCPP/build" --target llama-quantize -j 8 >/dev/null
    cp "$LCPP/build/bin/llama-quantize" "$QUANT"
  fi
fi
export LD_LIBRARY_PATH="$LCPP/bin:${LD_LIBRARY_PATH:-}"
chmod +x "$QUANT"; ls -l "$QUANT"

echo "== merge $(date +%H:%M)   free $(free_gb)G"
[ -f "$RUN/merged/config.json" ] || "$PY" scripts/ft/merge_adapter.py --run "$RUN"

echo "== gguf f16 $(date +%H:%M)   free $(free_gb)G"
if [ "$(free_gb)" -lt 35 ]; then
  # The base model's HF copy is only needed to merge (re-download: ~6 min).
  echo "low disk: removing the HF copy of the base model"; rm -rf "$HF_HOME/hub/models--Qwen--Qwen2.5-Coder-14B-Instruct"
fi
"$PY" "$LCPP/convert_hf_to_gguf.py" "$RUN/merged" --outtype f16 --outfile "$RUN/model-f16.gguf"
rm -rf "$RUN/merged"

echo "== quantize Q4_K_M $(date +%H:%M)   free $(free_gb)G"
"$QUANT" "$RUN/model-f16.gguf" "$RUN/$NAME-Q4_K_M.gguf" Q4_K_M 8
rm -f "$RUN/model-f16.gguf"
ls -lh "$RUN/$NAME-Q4_K_M.gguf"

echo "== ollama import $(date +%H:%M)"
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
"$OLLAMA_BIN" serve > "slurm_logs/ollama_export_${SLURM_JOB_ID}.log" 2>&1 &
trap 'kill $! 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
"$OLLAMA_BIN" show qwen2.5-coder:14b --modelfile | grep -v '^FROM ' > "$RUN/Modelfile.base"
{ echo "FROM $RUN/$NAME-Q4_K_M.gguf"; cat "$RUN/Modelfile.base"; } > "$RUN/Modelfile"
"$OLLAMA_BIN" create "$NAME" -f "$RUN/Modelfile"
"$OLLAMA_BIN" list | grep -E "qwen2.5-coder:14b|$NAME"
# Ollama keeps its own copy; the GGUF stays only if there is room for it.
[ "$(free_gb)" -lt 20 ] && rm -f "$RUN/$NAME-Q4_K_M.gguf"

echo "== smoke test $(date +%H:%M)"
for M in qwen2.5-coder:14b "$NAME"; do
  echo "-- $M"
  curl -s "http://$OLLAMA_HOST/api/generate" -d '{"model":"'"$M"'","stream":false,"options":{"temperature":0},"prompt":"Two Python packages have incompatible version requirements.\nChoose versions that resolve the conflict.\n\nERROR\n    transformers 4.30.0 depends on tokenizers<0.14 and >=0.11.1\n    The user requested tokenizers==0.15.0\n\nPACKAGES YOU MAY CHANGE\n  transformers             installed 4.30.0       newest 4.36.2\n  tokenizers               installed 0.13.3       newest 0.15.0  <-- MUST APPEAR IN YOUR ANSWER\n\nRULES\n- Every version must be HIGHER than \"installed\" and NOT higher than \"newest\".\n\nReply with one JSON object and nothing else."}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('response'))"
done
echo "== done $(date +%H:%M)   free $(free_gb)G"
