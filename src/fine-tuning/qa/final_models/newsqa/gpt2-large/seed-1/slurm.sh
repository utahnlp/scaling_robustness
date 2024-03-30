#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=gpt2-large_newsqa_1
#SBATCH --output=final_models//newsqa/gpt2-large/seed-1/slurm_output.txt
#SBATCH --error=final_models//newsqa/gpt2-large/seed-1/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness/qa
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source ../envs/py385_qa//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python finetune_gpt_qa_trainer.py --model_name_or_path gpt2-large --dataset_name mrqa^newsqa  --do_train --learning_rate 1e-05 --max_seq_length 512 --output_dir final_models//newsqa/gpt2-large/seed-1 --per_device_train_batch_size 8 --num_train_epochs 3 --doc_stride 128 --save_steps 1000000 --overwrite_output_dir --per_device_eval_batch_size 8 --num_beams 5"

python finetune_gpt_qa_trainer.py --model_name_or_path gpt2-large --dataset_name mrqa^newsqa  --do_train --learning_rate 1e-05 --max_seq_length 512 --output_dir final_models//newsqa/gpt2-large/seed-1 --per_device_train_batch_size 8 --num_train_epochs 3 --doc_stride 128 --save_steps 1000000 --overwrite_output_dir --per_device_eval_batch_size 8 --num_beams 5



