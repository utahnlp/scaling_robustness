task=$1
model_dir=$2

#cmd="awk '/^$task/' evaluation_files.tsv"
#cmd="awk '/^$task/' evaluation_files_leftover.tsv"
cmd="awk '/^$task/' evaluation_files_last.tsv"
files=$(eval $cmd | cut -f2)

#files=$(awk '/^mnli/' evaluation_files.tsv | cut -f2)
#files=$(awk '/^snli/' evaluation_files.tsv | cut -f2)

echo $files

if [[ $model_dir == *"t5"* ]]; then
    echo "Running for t5 model : $model_dir"
    if [[ $model_dir == *"11b"* ]]; then
        python t5_run_glue_trainer.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 8 --prediction_format 'default'
    else
        python t5_run_glue_trainer.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 32 --prediction_format 'default'
    fi
else
    echo "Running for non-t5 model : $model_dir"
    if [[ $model_dir == *"13b"* ]]; then
        python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --task_name $task --per_device_eval_batch_size 4
    else
        python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --task_name $task
    fi
fi

##python run_glue.py --model_name_or_path final_models/mnli/roberta-base/seed-1/ --output_dir final_models/mnli/roberta-base/seed-1/evaluations/ --test_file $files --do_predict --task_name mnli
#python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --task_name $task
#
#
#python t5_run_glue_trainer.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 32 --prediction_format 'default'
