#!/bin/bash
# ============================================================
# FaceLift Pipeline - Environment Setup
# ============================================================
# Usage: bash setup_environment.sh
# Prerequisites: NVIDIA GPU + CUDA 12.x driver installed
# ============================================================

set -e

echo "============================================"
echo " FaceLift Pipeline - Environment Setup"
echo "============================================"

# --- 1. Create Conda environment ---
echo ""
echo "[Step 1/6] Creating conda environment..."

if conda info --envs | grep -q "facelift_pipeline"; then
    echo "Environment 'facelift_pipeline' already exists, skipping creation."
else
    conda create -n facelift_pipeline python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate facelift_pipeline

# --- 2. Install PyTorch + CUDA ---
echo ""
echo "[Step 2/6] Installing PyTorch..."

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- 3. Clone and install FaceLift ---
echo ""
echo "[Step 3/6] Setting up FaceLift..."

if [ ! -d "FaceLift" ]; then
    git clone https://github.com/weijielyu/FaceLift.git
fi
cd FaceLift

# Install FaceLift dependencies
pip install -r requirements.txt 2>/dev/null || true

# FaceLift core dependencies
pip install \
    diffusers==0.25.1 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    pytorch-lightning==2.1.3 \
    kornia==0.7.1 \
    trimesh==4.0.8 \
    pymeshlab==2023.12.post1 \
    xatlas==0.0.9 \
    plyfile==1.0.2 \
    rembg==2.0.50 \
    gradio==4.14.0 \
    huggingface_hub==0.20.3

# Install diff-gaussian-rasterization
if [ -d "submodules/diff-gaussian-rasterization" ]; then
    pip install submodules/diff-gaussian-rasterization
else
    echo "WARNING: diff-gaussian-rasterization not found. Install manually."
fi

cd ..

# --- 4. Install additional pipeline dependencies ---
echo ""
echo "[Step 4/6] Installing pipeline dependencies..."

pip install -r requirements.txt

# --- 5. Configure Kaggle API ---
echo ""
echo "[Step 5/6] Kaggle API setup..."

if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Kaggle credentials found."
else
    echo "WARNING: ~/.kaggle/kaggle.json not found."
    echo "Download from https://www.kaggle.com/settings -> API -> Create New Token"
    echo "Place kaggle.json in ~/.kaggle/ and run: chmod 600 ~/.kaggle/kaggle.json"
fi

# --- 6. Create project directory structure ---
echo ""
echo "[Step 6/6] Creating directory structure..."

mkdir -p data/{raw_faces,cropped_faces,splats,depth_maps,normal_maps,opacity_maps,rgb_rendered}
mkdir -p data/{postprocessed,dataset,paired_dataset}
mkdir -p logs checkpoints

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate with: conda activate facelift_pipeline"
echo " Run pipeline:  python run_pipeline.py"
echo "============================================"
