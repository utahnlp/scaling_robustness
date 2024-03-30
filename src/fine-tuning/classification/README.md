## Code for training and evaluating classification models

### File Descriptions


- `run_glue.py` - For classification tasks (including non-glue) for non-generative models
- `run_glue_no_trainer.py` - For classification tasks (including non-glue) for non-generative models - without the trainer - mostly used for debugging, etc.
- `utils.py` - Utility functions to load evaluation files, etc.
- `constants.py` - Constants like parameter counts, column mappings, etc.
- `t5_run_glue.py` - Non-trainer version to train text-to-text models for classification (ex: T5).
- `t5_run_glue_trainer.py` - Trainer version to train text-to-text models for classification (ex: T5).
- `t5_run_glue_trainer_boolq.py` - Trainer version to train text-to-text models for classification (ex: T5). The only difference from `t5_run_glue_trainer.py` is that this file works with other datasets like `boolq`, and `quoref`, etc. For simpler classification tasks, you can use either of them.

Important Notes:

1. Note that in order to run larger models using `deepspeed`, you need to use the trainer versions. The non-trainer versions are here only for debugging purposes and should not otherwise be used.
2. Due to a recent update to the `glue` datasets on huggingface, they have converted it to a no-script Parquet format and therefore, you need to add an additional argument of `revision='script'` with `load_dataset` function for the `datasets` version we used. I have made the necessary change in all the files. In future, this might be discontinued, and you might have to upgrade the `datasets` version to make it work. Please refer to this [discussion](https://huggingface.co/datasets/nyu-mll/glue/discussions/17).

### Training

For training any of the encoder-only or decoder-only classification models:
```
facebook/opt-*
gpt2*
microsoft/deberta-v3*
roberta-*
```
use the `run_glue.py` script:
```
export TASK=mnli
export MODEL_NAME=facebook/opt-125m
export SEED=1
export BATCH_SIZE=32
export SEQ_LEN=256
export LEARNING_RATE=5e-5
export OUTPUT_DIR=final_models/${TASK}/${MODEL_NAME}/seed-${SEED}/

mkdir -p $OUTPUT_DIR

python run_glue.py --model_name_or_path $MODEL_NAME \ 
                    --per_device_train_batch_size $BATCH_SIZE \ 
                    --num_train_epochs 3 --seed $SEED --task_name $TASK \ 
                    --do_train --do_eval --save_steps 1000000 --max_seq_length $SEQ_LEN \ 
                    --overwrite_output_dir --output_dir $OUTPUT_DIR --learning_rate $LEARNING_RATE
```

Similary, for training text-to-text, encode-decoder models like `t5`, you can use the following command:
```
python t5_run_glue_trainer.py --model_name_or_path $MODEL_NAME \ 
                    --per_device_train_batch_size $BATCH_SIZE \ 
                    --num_train_epochs 3 --seed $SEED --task_name $TASK \
                    --do_train --do_eval --save_steps 1000000 --max_seq_length $SEQ_LEN \ 
                    --overwrite_output_dir --output_dir $OUTPUT_DIR --learning_rate $LEARNING_RATE \
                    --predict_with_generate
```

#### Training Scripts Used
We provide all the training scripts with the exact hyperparmeters (learning rates, batch sizes, etc.) that were used for training our models. These are available in the directory `final_models/`. The directory structure is `final_models/${task}/${model_name}/seed-${seed}/` with the slurm file `slurm.sh`. For example if you want to check the exact command used for training `gpt2-medium` on `ag_news` for `seed = 1`, you can refer to file [final_models/ag_news/gpt2-medium/seed-1/slurm.sh](final_models/ag_news/gpt2-medium/seed-1/slurm.sh).

### Evaluation

#### Generate predictions for a set of test files
To generate predictions for a test file `test.jsonl`, you can use the following command:

```
export TASK=mnli
export MODEL_NAME=facebook/opt-125m
export SEED=1
export BATCH_SIZE=32
export MODEL_DIR=final_models/${TASK}/${MODEL_NAME}/seed-${SEED}/

python run_glue.py --model_name_or_path $MODEL_DIR --output_dir ./ \ 
                    --test_file test.jsonl --do_predict --task_name $TASK \ 
                    --per_device_eval_batch_size $BATCH_SIZE
```

This will save the predictions in the file `test.jsonl.preds.jsonl` in the output directory.

You can also provide multiple test files separated by space: `--test_file test1.jsonl test2.jsonl ...`
