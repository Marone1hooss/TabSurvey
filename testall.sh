#!/bin/bash

#SBATCH --partition=gpu # partition name                                                                                                                                               
#SBATCH --account=MANAPY-1WABCJWE938-DEFAULT-GPU                                                                                                                                           
#SBATCH --export=NONE
#SBATCH --gres=gpu:1                     # Necessary to activate the gpu card (The number of GPUs allowed by node is 1)
##SBATCH --nodes=1
#SBATCH -n 1                               # number of cores ( max 44 per node)
#SBATCH --time=1-00:00:00                      # wall time to finish the job



#SBATCH --job-name=X          # job name
#SBATCH --output=tabsurvey-%j.log         # output file

module load Anaconda3 CUDA/11.1.1

N_TRIALS=2
EPOCHS=3

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

declare -A MODELS
MODELS=( ["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
         # ["SVM"]=$SKLEARN_ENV
         ["DecisionTree"]=$SKLEARN_ENV
         ["RandomForest"]=$SKLEARN_ENV
         ["XGBoost"]=$GBDT_ENV
         ["CatBoost"]=$GBDT_ENV
         ["LightGBM"]=$GBDT_ENV
         ["MLP"]=$TORCH_ENV
         ["TabNet"]=$TORCH_ENV
         ["VIME"]=$TORCH_ENV
         ["TabTransformer"]=$TORCH_ENV
         ["ModelTree"]=$GBDT_ENV
         ["NODE"]=$TORCH_ENV
         ["DeepGBM"]=$TORCH_ENV
         ["RLN"]=$KERAS_ENV
         ["DNFNet"]=$KERAS_ENV
         ["STG"]=$TORCH_ENV
         ["NAM"]=$TORCH_ENV
         ["DeepFM"]=$TORCH_ENV
         ["SAINT"]=$TORCH_ENV
         ["DANet"]=$TORCH_ENV
          )

CONFIGS=( "config/adult.yml"
          "config/covertype.yml"
          "config/california_housing.yml"
          "config/higgs.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"


for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS 

    conda deactivate

  done

done
