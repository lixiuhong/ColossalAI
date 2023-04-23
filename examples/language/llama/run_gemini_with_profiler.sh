set -x
export BS=${BS:-16}
export MEMCAP=${MEMCAP:-0}
# Acceptable values include `7b`, `13b`, `33b`, `65b`. For `7b`
export MODEL=${MODEL:-"7b"}
export GPUNUM=${GPUNUM:-8}
export USE_SHARD_INIT=${USE_SHARD_INIT:-"false"}

# make directory for logs
mkdir -p ./logs

if [ ${USE_SHARD_INIT} = "true" ]; then
  USE_SHARD_INIT="--shardinit"
else
  USE_SHARD_INIT=""
fi

export MODLE_PATH="decapoda-research/llama-7b-hf"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  train_gemini_llama_with_profiler.py \
  --mem_cap ${MEMCAP} \
  --model_name_or_path ${MODLE_PATH} \
  ${USE_SHARD_INIT} \
  --batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log

