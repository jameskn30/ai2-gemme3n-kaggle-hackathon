#!/bin/bash

set -e

# Install CUDA 12.8 (Ubuntu example)
echo "Installing CUDA 12.8..."

if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA is already installed. Do you want to proceed with reinstallation? (y/N): "
    read -p "CUDA appears to be installed. Do you want to proceed with reinstallation? (y/N): " confirm
    confirm=${confirm,,} # tolower
    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "Aborting installation."
        exit 0
    fi
fi


echo "Removing previous CUDA installations..."
# Remove previous CUDA installations
sudo apt-get --purge remove -y '*cublas*' '*cufft*' '*curand*' '*cusolver*' '*cusparse*' '*npp*' '*nvjpeg*' 'cuda*' 'nsight*' || true
sudo apt-get autoremove -y
sudo apt-get autoclean -y

CUDA_VERSION="12.8.0"
CUDA_RUN_FILE="cuda_${CUDA_VERSION}_570.86.10_linux.run"

# Download and install CUDA 12.8
if [ ! -f $CUDA_RUN_FILE ]; then
    wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUN_FILE}
fi
sudo sh $CUDA_RUN_FILE --silent --toolkit

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Activate Python .venv
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo ".venv not found, creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Uninstall existing torch, torchvision, torchaudio if present
pip uninstall -y torch torchvision torchaudio || true

# Install latest PyTorch for CUDA 12.8
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "CUDA 12.8 and PyTorch (CUDA 12.8) installation complete."

rm $CUDA_RUN_FILE

