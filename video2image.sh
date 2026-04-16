#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Activer environnement
source ~/.bashrc
conda activate 4D-humans

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge

mkdir -p logs

# Loop sur toutes les vidéos (triées pour stabilité)
videos=$(ls data/videos/*.mp4 data/videos/*.mov 2>/dev/null | sort)

for video in $videos; do
    # Nom du fichier sans extension
    name=$(basename "$video")
    name="${name%.*}"

    output_dir="data/images/${name}"

    echo "Processing $video → $output_dir"

    python video2image.py \
        --video_path "$video" \
        --output_folder "$output_dir"
done

echo "All videos processed."