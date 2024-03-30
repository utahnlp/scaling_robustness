#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=0
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=t5-11b_naturalquestionsShort_1
#SBATCH --output=final_models//naturalquestionsShort/t5-11b/seed-1/slurm_output.txt
#SBATCH --error=final_models//naturalquestionsShort/t5-11b/seed-1/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np
#SBATCH --requeue




WORK_DIR=$HOME/scr/robustness/t5_robustness/qa
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source ../envs/py385_ds_2//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python run_seq2seq_qa.py --model_name_or_path t5-11b --dataset_name mrqa^naturalquestionsShort --context_column context --question_column question --answer_column detected_answers --do_train --do_eval --per_device_train_batch_size 1 --learning_rate 5e-05 --num_train_epochs 3 --max_seq_length 512 --doc_stride 128 --output_dir final_models//naturalquestionsShort/t5-11b/seed-1 --save_steps 1000000 --overwrite_output_dir"

deepspeed run_seq2seq_qa.py --deepspeed ../ds_config_zero3.json --model_name_or_path t5-11b --dataset_name mrqa^naturalquestionsShort --context_column context --question_column question --answer_column detected_answers --do_train --per_device_train_batch_size 4 --learning_rate 5e-05 --num_train_epochs 3 --max_seq_length 512 --doc_stride 128 --output_dir final_models//naturalquestionsShort/t5-11b/seed-1 --save_steps 1000000 --overwrite_output_dir --bf16 --gradient_accumulation_steps 2



