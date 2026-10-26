#!/bin/bash --login
#SBATCH --job-name=roberta-pan-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h200:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=roberta_%j.out
#SBATCH --error=roberta_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load Miniforge3/25.11.0-1
conda activate roberta-h200
export HF_HOME="$SCRATCH/nlp_project/hf_cache"
export TOKENIZERS_PARALLELISM=false

python -u finetune_roberta_h200.py \
    --data-dir pan25-generative-ai-detection-task2-train \
    --output-dir roberta_pan_task2_full_finetune \
    --batch-size 32 \
    --epochs 3 \
    --max-length 256
