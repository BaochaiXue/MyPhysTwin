conda create -y -n phystwin python=3.10
conda activate phystwin

# we want to avoid the interference from ~/.local
export PYTHONNOUSERSITE=1
unset PIP_USER   

conda install -n phystwin -c nvidia/label/cuda-12.1.1 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
conda install -n phystwin -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y
export NVCC_CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"



# Install the packages
# If you only want to explore the interactive playground, you can skip installing Trellis, Grounding-SAM-2, RealSense, and SDXL.
bash ./env_install/env_install.sh > tmp.txt 2> tmp1.err

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh > tmp2.txt 2> tmp2.err