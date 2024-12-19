#!/bin/bash

# Make this script executable
chmod +x "$0"

echo "Starting main script..."

# Process ANN directory
echo "Processing directory: ANN"
cd ANN || { echo "Failed to enter directory ANN"; exit 1; }

# List of .sh files in ANN directory
scripts_ann=(
    "qdrant_ann_gist_cos.sh"
    "qdrant_ann_gist_dot.sh"
    "qdrant_ann_gist_euc.sh"
    "qdrant_ann_sift_cos.sh"
    "qdrant_ann_sift_dot.sh"
    "qdrant_ann_sift_euc.sh"
    "weaviate_ann_gist_cos.sh"
    "weaviate_ann_gist_dot.sh"
    "weaviate_ann_gist_euc.sh"
    "weaviate_ann_sift_cos.sh"
    "weaviate_ann_sift_dot.sh"
    "weaviate_ann_sift_euc.sh"
)

for script in "${scripts_ann[@]}"; do
    if [[ -f "$script" ]]; then
        chmod +x "$script"
        echo "Preparing to run: $script"
        ./$script
    else
        echo "Script $script not found in ANN."
    fi
done
cd .. >/dev/null || exit

# Process AF directory
echo "Processing directory: AF"
cd AF || { echo "Failed to enter directory AF"; exit 1; }

# List of .sh files in AF directory
scripts_af=(
    "qdrant_af_gist_cos.sh"
    "qdrant_af_gist_dot.sh"
    "qdrant_af_gist_euc.sh"
    "qdrant_af_sift_cos.sh"
    "qdrant_af_sift_dot.sh"
    "qdrant_af_sift_euc.sh"
    "weaviate_af_gist_cos.sh"
    "weaviate_af_gist_dot.sh"
    "weaviate_af_gist_euc.sh"
    "weaviate_af_sift_cos.sh"
    "weaviate_af_sift_dot.sh"
    "weaviate_af_sift_euc.sh"
)

for script in "${scripts_af[@]}"; do
    if [[ -f "$script" ]]; then
        chmod +x "$script"
        echo "Preparing to run: $script"
        ./$script
    else
        echo "Script $script not found in AF."
    fi
done
cd .. >/dev/null || exit

echo "Main script finished."
