task=$1
model_dir=$2
data_dir=$3

GLUE_TASKS=("snli" "sst2" "mnli" "qqp" "qnli" "mrpc")

eval_files_dir=$data_dir/$task/

# Read files from the directory
files=""
for file in $eval_files_dir/*; do
    if [ -f "$file" ]; then
        if [[ "$file" == *" "* ]]; then
            new_file="${file// /_}"
            cp "$file" "$new_file"
            files+="$new_file "
        else
            files+="$file "
        fi
    fi
done


echo $files

if [[ $model_dir == *"t5"* ]]; then
    echo "Running for t5 model : $model_dir"
    if [[ $task == *"boolq"* ]]; then
        echo "python t5_run_glue_trainer_boolq.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 8 --prediction_format 'default'"
        python t5_run_glue_trainer_boolq.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 8 --prediction_format 'default'
    else
        echo "python t5_run_glue_trainer.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 8 --prediction_format 'default'"
        python t5_run_glue_trainer.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --do_predict --test_file $files  --task_name $task --predict_with_generate --per_device_eval_batch_size 8 --prediction_format 'default'
    fi
else
    echo "Running for non-t5 model : $model_dir"
    # Check if the task is in the array of glue tasks
    if [[ " ${GLUE_TASKS[@]} " =~ " $task " ]]; then
        echo "python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --task_name $task"
        python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --task_name $task
    else
        echo "python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --dataset_name $task"
        python run_glue.py --model_name_or_path $model_dir --output_dir $model_dir/evaluations/ --test_file $files --do_predict --dataset_name $task
    fi
fi

