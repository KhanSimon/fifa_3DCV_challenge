#!/bin/bash -l
#SBATCH --job-name=sam3d
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 4D-humans

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

echo "===== ENV CHECK ====="
python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
EOF

echo "===== IMPORT CHECK ====="
python - <<EOF
import sam_3d_body
print("sam_3d_body OK")
EOF


python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
EOF



echo "===== RUN SCRIPT ====="

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge


python sam_3d_body/demo.py \
    --image_folder sam_3d_body/example_data/images \
    --output_folder sam_3d_body/example_data/output \
    --checkpoint_path /home/BeeGFS/Laboratories/IBHGC/skhan/models/sam-3d-body-dinov3/model.ckpt \
    --mhr_path /home/BeeGFS/Laboratories/IBHGC/skhan/models/sam-3d-body-dinov3/assets/mhr_model.pt \
    --fov_path /home/BeeGFS/Laboratories/IBHGC/skhan/models/moge-2-vitl-normal/model.pt

echo "===== DONE ====="