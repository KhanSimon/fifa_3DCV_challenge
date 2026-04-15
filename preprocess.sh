#!/bin/bash -l
#SBATCH --job-name=sam3d_kpts
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100S:1
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=slurms/slurm_%A_%a.out
#SBATCH --error=slurms/slurm_%A_%a.err

module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 4D-humans

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge

# récupère la séquence correspondant à l'index
SEQ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" data/sequences_full.txt)

echo "===== RUN $SEQ ====="

nvidia-smi

python preprocess.py --root data --sequence "$SEQ"

echo "===== DONE $SEQ ====="

: << 'COMMENT'
VERSION PAS BATCH PAR BATCH, TOUT SUR LE MEME JOB, MAIS CA MARCHE

#!/bin/bash -l
#SBATCH --job-name=sam3d_kpts
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100S:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 4D-humans

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge

echo "===== GPU CHECK ====="
nvidia-smi
python - <<EOF
import torch
print("device:", torch.cuda.get_device_name(0))
print("total mem (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
EOF

python preprocess.py

COMMENT
