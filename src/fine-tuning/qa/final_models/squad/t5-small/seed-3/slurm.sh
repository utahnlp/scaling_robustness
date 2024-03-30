#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=t5-small_squad_3
#SBATCH --output=final_models//squad/t5-small/seed-3/slurm_output.txt
#SBATCH --error=final_models//squad/t5-small/seed-3/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness/qa
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source ../envs/py385_qa//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python run_seq2seq_qa.py --model_name_or_path t5-small --dataset_name mrqa^squad --context_column context --question_column question --answer_column detected_answers --do_train --do_eval --per_device_train_batch_size 8 --learning_rate 5e-05 --num_train_epochs 3 --max_seq_length 512 --doc_stride 128 --output_dir final_models//squad/t5-small/seed-3 --save_steps 1000000 --overwrite_output_dir"

python run_seq2seq_qa.py --model_name_or_path t5-small --dataset_name mrqa^squad --context_column context --question_column question --answer_column detected_answers --do_train --do_eval --per_device_train_batch_size 8 --learning_rate 5e-05 --num_train_epochs 3 --max_seq_length 512 --doc_stride 128 --output_dir final_models//squad/t5-small/seed-3 --save_steps 1000000 --overwrite_output_dir



