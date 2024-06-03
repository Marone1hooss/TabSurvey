#!/bin/bash      

#SBATCH --partition=gpu # partition name                                                                                                                                               
#SBATCH --account=MANAPY-1WABCJWE938-DEFAULT-GPU                                                                                                                                           
#SBATCH --export=NONE
#SBATCH --gres=gpu:1                     # Necessary to activate the gpu card (The number of GPUs allowed by node is 1)
##SBATCH --nodes=1
#SBATCH -n 1                               # number of cores ( max 44 per node)

#SBATCH --time=1-00:00:00 
#SBATCH --job-name test2
#SBATCH --output /home/marwan.housni/lustre/manapy-um6p-st-msda-1wabcjwe938/users/marwan.housni/test%j.log
#SBATCH  --array=1-21
# Load modules

module load CUDA
N_TRIALS=2
EPOCHS=3

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

Listmodels=(
    "LinearModel"
    "KNN"
    "DecisionTree"
    "RandomForest"
    "XGBoost"
    "CatBoost"
    "LightGBM"
    "MLP"
    "TabNet"
    "VIME"
    "TabTransformer"
    "ModelTree"
    "NODE"
    "DeepGBM"
    "RLN"
    "DNFNet"
    "STG"
    "NAM"
    "DeepFM"
    "SAINT"
    "DANet"
)

Listenv=(
    $SKLEARN_ENV
    $SKLEARN_ENV
    $SKLEARN_ENV
    $SKLEARN_ENV
    $GBDT_ENV
    $GBDT_ENV
    $GBDT_ENV
    $TORCH_ENV
    $TORCH_ENV
    $TORCH_ENV
    $TORCH_ENV
    $GBDT_ENV
    $TORCH_ENV
    $TORCH_ENV
    $KERAS_ENV
    $KERAS_ENV
    $TORCH_ENV
    $TORCH_ENV
    $TORCH_ENV
    $TORCH_ENV
    $TORCH_ENV
)


CONFIGS=( "config/adult.yml"
          "config/california_housing.yml"
          "config/higgs.yml"
          "config/covertype.yml"

          )
# Calculate the array index for num_samples and core count
array_index=$((SLURM_ARRAY_TASK_ID - 1))
#array_index=16
echo $array_index

#echo ${!MODELS[$array_index]}

#num_samples_index=$((array_index % ${#!MODELS[@]}))


# Get the current values of num_samples and core count
#current_model=${MODELS[$array_index]}
#current_config=${CONFIGS[$num_samples_index]}

eval "$(conda shell.bash hook)"


for config in "${CONFIGS[@]}"; do
    
    #${Listmodels[$array_index]}, ${Listenv[$array_index]}
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "${Listmodels[$array_index]}" "$config"  "${Listenv[$array_index]}"

    conda activate "${Listenv[$array_index]}"

    python train.py --config "$config" --model_name "${Listmodels[$array_index]}" --n_trials $N_TRIALS --epochs $EPOCHS 

    conda deactivate

done


# conda init bash
#eval "$(conda shell.bash hook)"

#echo $current_model





#Laplacian1024.mtx  Laplacian16384.mtx  Laplacian262144.mtx  Laplacian262144.zip  Laplacian32400.mtx  Laplacian4096.mtx  Laplacian65536.mtx


#matrix_samples=(Laplacian1024.mtx Laplacian4096.mtx Laplacian16384.mtx Laplacian32400.mtx Laplacian65536.mtx  3DProblem_3375.mtx 3D_Problem29791.mtx  3D_Problem59319.mtx)






