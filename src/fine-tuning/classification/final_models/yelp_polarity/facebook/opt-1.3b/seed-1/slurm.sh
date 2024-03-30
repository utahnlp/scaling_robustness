#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:4

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=facebook_opt-1.3b_yelp_polarity_1
#SBATCH --output=final_models//yelp_polarity/facebook/opt-1.3b/seed-1/slurm_output.txt
#SBATCH --error=final_models//yelp_polarity/facebook/opt-1.3b/seed-1/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source env_t5_textattack//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python run_glue.py --model_name_or_path facebook/opt-1.3b --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --dataset_name yelp_polarity --do_train --save_steps 1000000 --max_seq_length 512 --overwrite_output_dir --output_dir final_models//yelp_polarity/facebook/opt-1.3b/seed-1 --learning_rate 5e-06"



python run_glue.py --model_name_or_path facebook/opt-1.3b --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --dataset_name yelp_polarity --do_train --save_steps 1000000 --max_seq_length 512 --overwrite_output_dir --output_dir final_models//yelp_polarity/facebook/opt-1.3b/seed-1 --learning_rate 5e-06



