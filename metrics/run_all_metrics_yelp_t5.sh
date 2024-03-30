task=$1

results_file=$2

#echo "train_dataset,model_name,model_full_name,params,seed,eval_dataset,category,sub_category,score,metric,f1" >> $results_file
for model in t5-3b t5-small t5-base t5-large t5-11b ;
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
        python yelp_polarity_t5.py $EVALUATIONS_DIR $results_file
    done
done

