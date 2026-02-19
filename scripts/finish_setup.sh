#!/bin/bash

# Exit on error
set -e

echo "Starting dataset setup..."

# 1. Generate the map
echo "Generating dataset map..."
python scripts/prepare_dataset_map.py

# 2. Split train/val
echo "Splitting dataset..."
python src/splits/train_val_split.py

echo "✅ Dataset setup complete!"
echo "You can now run training with:"
echo "python scripts/train.py --model swin --epochs 50 --batch_size 8"
