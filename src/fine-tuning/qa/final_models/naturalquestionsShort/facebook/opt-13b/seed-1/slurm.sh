#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=0
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=facebook_opt-13b_naturalquestionsShort_1
#SBATCH --output=final_models//naturalquestionsShort/facebook/opt-13b/seed-1/slurm_output.txt
#SBATCH --error=final_models//naturalquestionsShort/facebook/opt-13b/seed-1/slurm_output.txt
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




echo "deepspeed finetune_gpt_qa_trainer.py --deepspeed ../ds_config_zero3.json --model_name_or_path facebook/opt-13b --dataset_name mrqa^naturalquestionsShort --do_train --max_seq_length 512 --doc_stride 128 --output_dir final_models/naturalquestionsShort/facebook/opt-13b/seed-1/ --per_device_eval_batch_size 8 --num_beams 5 --overwrite_output_dir --per_device_train_batch_size 8 --learning_rate 5e-06 --bf16"

deepspeed finetune_gpt_qa_trainer.py --deepspeed ../ds_config_zero3.json --model_name_or_path facebook/opt-13b --dataset_name mrqa^naturalquestionsShort --do_train --max_seq_length 512 --doc_stride 128 --output_dir final_models/naturalquestionsShort/facebook/opt-13b/seed-1/ --per_device_eval_batch_size 8 --num_beams 5 --overwrite_output_dir --per_device_train_batch_size 8 --learning_rate 5e-06 --bf16




