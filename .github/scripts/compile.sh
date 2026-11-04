#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -e
source /opt/dtk/env.sh

check_torch() {
   if ! python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
      echo "PyTorch is not installed. Please install PyTorch first."
      exit 1
   fi
}

check_dtk() {
    local dir
    local dtk_dir
    shopt -s nullglob
    dtk_dir=(/opt/dtk /opt/dtk-*)
    shopt -u nullglob
    for dir in "${dtk_dir[@]}"; do
        if [ -d "$dir" ]; then
            DTK_PATH="$dir"
            break
        fi
    done
    if [ -z "$DTK_PATH" ]; then
        echo "DTK is not installed. Please install DTK first."
        exit 1
    fi
    echo "DTK is installed: ${DTK_PATH}"
}

check_aicc() {
    if [ ! -e "${DTK_PATH}/bin/aicc" ]; then
        echo "AICC is not installed. Please install AICC first."
        exit 1
    fi
    echo "AICC is installed."
}

install_dependencies(){
    echo "install env"
    wget http://42.228.13.241:18000/customized/amdsmi/amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl -O amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl
    pip install amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl
    pip install scipy==1.14.1
    pip install cupy
    pip install pytest-timeout -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip install setuptools==79.0.1 wheel  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip install --upgrade pip
    pip install appdirs -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install  pluggy -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install  iniconfig  -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install  execnet  -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install  comet-ml tensorboard wandb datasets sphinx sphinx-rtd-theme mup pre-commit pytest-forked recommonmark -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip install  evaluate  -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple

    pip install platformdirs -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install docker_pycreds  -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pip install sentry-sdk -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    pkgs=("mpi4py" "tabulate" "pytorch-lightning" "transformers" "accelerate")
    for pkg in ${pkgs[@]};do
        pip install ${pkg} -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    done
    pip3 install -r  requirements/requirements-autotuning-ml.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip3 install -r  requirements/requirements-1bit-mpi.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip3 install -r  requirements/requirements-autotuning.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip3 install -r  requirements/requirements-inf.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip3 install -r  requirements/requirements-sd.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    sed -i 's/^numpy$/numpy==2.1.0/' requirements/requirements.txt
    pip install -r requirements/requirements.txt   -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    pip install -r requirements/requirements-dev.txt  --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
}

check_torch
check_dtk
#check_aicc
install_dependencies

export BUILD_ROOT=${GITHUB_WORKSPACE}
export PYTHON_VERSION=$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')
export LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages/torch/lib:$BUILD_ROOT/third_party/libaio_build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages/torch/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$BUILD_ROOT/third_party/libaio_build/include:$C_INCLUDE_PATH
export C_PLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export CFLAGS="-I$BUILD_ROOT/third_party/libaio_build/include/ -w"
export CXXFLAGS="-w"
export LDFLAGS="-L$BUILD_ROOT/third_party/libaio_build/lib/"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/hcusmi:$LD_LIBRARY_PATH
python3 --version
echo ${PYTHON_VERSION}
python -c "import torch, os; print(os.path.dirname(torch.__file__))"
echo "obtain the path of the torch"
TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
echo "${TORCH_PATH}"
export LD_LIBRARY_PATH=${TORCH_PATH}/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
export CXX=hipcc
export CC=hipcc
DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_CUTLASS_OPS=0 DS_BUILD_FP_QUANTIZER=0  DS_BUILD_OPS=1  DS_ACCELERATOR='cuda' python3 setup.py install bdist_wheel
pip install dist/*.whl