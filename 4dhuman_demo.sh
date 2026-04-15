#!/bin/bash -l
#SBATCH --job-name=4dh_demo
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 4D-humans

# Garde ces variables au cas où detectron2 charge des extensions compilées
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

echo "===== ENV CHECK ====="
which python
python --version
python - <<EOF
import torch, detectron2, chumpy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("detectron2:", detectron2.__file__)
print("chumpy ok")
EOF

echo "===== CACHE CHECK ====="
ls -lh /home/BeeGFS/Laboratories/IBHGC/skhan/.cache/4DHumans || true
ls -lh /home/BeeGFS/Laboratories/IBHGC/skhan/.cache/4DHumans/detectron2 || true
ls -lh /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge/4D-Humans/data || true

echo "===== RUN DEMO ====="
cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge/4D-Humans

python demo.py \
  --img_folder example_data/images \
  --out_folder demo_test \
  --batch_size 1 \
  --detector vitdet

echo "===== DONE ====="