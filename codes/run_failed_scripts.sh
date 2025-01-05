#!/bin/bash

# Make this script executable
chmod +x "$0"

echo "Starting main script..."

# Process ANN directory
echo "Processing directory: ANN"
cd ANN || { echo "Failed to enter directory ANN"; exit 1; }

# List of .sh files in ANN directory
scripts_ann=(
    "weaviate_ann_gist_cos.sh"
    "weaviate_ann_gist_dot.sh"
    "weaviate_ann_gist_euc.sh"
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

echo "Main script finished."
