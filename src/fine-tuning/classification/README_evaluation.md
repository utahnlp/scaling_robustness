## Evaluation of Finetuned Models

### Out-of-domain Evaluation

1. Please download the evaluations sets from [this drive link](https://drive.google.com/file/d/1K2tCKJeG8WfHA7QRtqszNj7_PjIAwwGp/view?usp=sharing), and extract into the `data` folder in the root of the repository.
    ```
    # make sure to copy all_evaluation_sets.tar.gz into the data folder before extracting
    tar -zxf all_evaluation_sets.tar.gz
    ```

2. Run the following script to run all evaluations for a particular task:
    ```
    export TASK=mnli
    export MODEL_DIR=final_models/mnli/roberta-base/seed-1/
    export DATA_DIR="../../../data/all_evaluation_sets/"
    
    sh run_model_evaluations $TASK $MODEL_DIR $DATA_DIR
    ```
    This will create a directory called `evaluations/` inside `$MODEL_DIR` and save all the predictions.

3. Calculate metrics: 