#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --partition=performance
#SBATCH --job-name="22FS_I4DS11"
#SBATCH --error=output/err-%j.err
#SBATCH --out=output/out-%j.log
basePath=${PWD}/service
dataPath=${PWD}/data
SECONDS=0
output="output/$SLURM_JOB_ID-$SLURM_JOB_NAME"
mkdir $output
output=${PWD}/$output
sampleOfData=None
filename='input_ML_tabula_Muris_modified.csv'
modelVersion='aparent'
dataVersion='Datenset1'


if [ -n "$1" ]; then
    sampleOfData=$1
fi    

# singularity exec -B $basePath:$basePath aparent.sif \
python3 $basePath/startTrainNeuralNetwork.py \
-b $basePath \
-o $output \
-s $sampleOfData \
-d $dataPath \
-f $filename \
-m $modelVersion \
-dv $dataVersion 
/

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

mv "output/out-$SLURM_JOB_ID.log" "$output/out.log"
mv "output/err-$SLURM_JOB_ID.err" "$output/err.err"