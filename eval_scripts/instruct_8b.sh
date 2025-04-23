#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <task> [n_shots]"
    echo "Note: model_path is ignored for this script as it uses a fixed model"
    echo "Available tasks: gsm8k_cot_llama, wmt16-de-en, ifeval, arc_challenge_chat, mmlu_llama, nq_open"
    exit 1
fi

MODEL_PATH=$1  # This is ignored, kept for consistency
TASK=$2
MODEL=meta-llama/Llama-3.1-8B-Instruct

# Define available tasks
declare -a TASKS=(
    "gsm8k_cot_llama"
    "wmt16-de-en"
    "ifeval"
    "arc_challenge_chat"
    "mmlu_llama"
    "nq_open"
)

# Validate if the provided task is in the list of available tasks
valid_task=false
for valid_task_name in "${TASKS[@]}"; do
    if [ "$TASK" == "$valid_task_name" ]; then
        valid_task=true
        break
    fi
done

if [ "$valid_task" = false ]; then
    echo "Error: Invalid task '$TASK'"
    echo "Available tasks: ${TASKS[*]}"
    exit 1
fi

# Set N_SHOT based on task or use provided value
if [ "$TASK" == "arc_challenge_chat" ]; then
    N_SHOT=0
elif [ $# -ge 3 ]; then
    N_SHOT=$3
    echo "Using provided N_SHOT value: $N_SHOT"
else
    N_SHOT=4
    echo "No N_SHOT provided, using default value: 4"
fi

# Default batch size
BATCH_SIZE=1

echo "=> evaluating 8b-instruct model on ${TASK} with ${N_SHOT} shots"
lm_eval \
    --model hf \
    --model_args pretrained=$MODEL \
    --tasks $TASK \
    --num_fewshot $N_SHOT \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --output_path eval_results/instruct_8b_${TASK}_${N_SHOT}shots