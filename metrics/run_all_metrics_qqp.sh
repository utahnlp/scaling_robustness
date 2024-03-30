task=qqp

results_file=$1

echo "train_dataset,model_name,model_full_name,params,seed,eval_dataset,category,sub_category,score,metric,f1" >> $results_file
for model in roberta-base roberta-large gpt2 gpt2-medium gpt2-large gpt2-xl t5-3b t5-small t5-base t5-large t5-11b microsoft/deberta-v3-base microsoft/deberta-v3-large facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b;
do
    seed_list=(1 2 3)
    if [[ "$model" == "t5-11b" ||  "$model" == "facebook/opt-13b" ||  "$model" == "facebook/opt-6.7b" ]]; then
        seed_list=(1)
    fi
    for seed in "${seed_list[@]}";
    do
        echo "Runing for model: $model and seed: $seed"
        MODEL_DIR=../final_models/$task/$model/seed-$seed/
        EVALUATIONS_DIR=$MODEL_DIR/evaluations/
        #python mnli_scorer.py $EVALUATIONS_DIR $model $seed
        python ${task}_scorer.py $EVALUATIONS_DIR $results_file
    done
done

