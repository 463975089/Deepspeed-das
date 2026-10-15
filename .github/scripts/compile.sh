#!/bin/bash

source /opt/dtk/env.sh
export PYTHON_VERSION=$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')
if [[ $PYTHON_VERSION == "3.10" ]];then
   pip install --force-reinstall torch==2.10.0+das.opt1.dtk2604.2607161508.gd52b39 -i https://pypi.sourcefind.cn/nightly/dtk/
fi

if [[ $PYTHON_VERSION == "3.12" ]];then
   pip install scipy==1.14.1
fi

IMAGE_NAME=""
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}" in
        rocky|rhel|centos|almalinux|ol|opencloudos|alinux)
            IMAGE_NAME="rockylinux"
            ;;
        ubuntu|debian)
            IMAGE_NAME="ubuntu"
            ;;
        *)
            IMAGE_NAME=""
            ;;
    esac
fi

if [[ "${IMAGE_NAME}" == "ubuntu" ]]; then
    cat > /etc/apt/sources.list <<'EOF'
deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
EOF
    apt-get update
    apt-get install libaio-dev -y
else
    yum install -y libaio-devel
    yum install -y libaio
fi

cd ${WORK_SPACE}
pip install http://42.228.13.241:18000/pkg/hcusmi-1.0.0%2B5d7586ac.dirty-py3-none-any.whl
pip3 install -r  requirements/requirements-autotuning-ml.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install -r  requirements/requirements-1bit-mpi.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install -r  requirements/requirements-autotuning.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install -r  requirements/requirements-inf.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install -r  requirements/requirements-sd.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install -r requirements/requirements.txt   -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install -r requirements/requirements-dev.txt  --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

export BUILD_ROOT=${WORK_SPACE}
echo $BUILD_ROOT
export LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages/torch/lib:$BUILD_ROOT/third_party/libaio_build/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$BUILD_ROOT/third_party/libaio_build/include:$C_INCLUDE_PATH
export C_PLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export CFLAGS="-I$BUILD_ROOT/third_party/libaio_build/include/ -w"
export CXXFLAGS="-w"
export LDFLAGS="-L$BUILD_ROOT/third_party/libaio_build/lib/"
export HSA_FORCE_FINE_GRAIN_PCIE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/hcusmi:$LD_LIBRARY_PATH
TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
export LD_LIBRARY_PATH=${TORCH_PATH}/lib:$LD_LIBRARY_PATH

export CXX=hipcc
export CC=hipcc
DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_CUTLASS_OPS=0 DS_BUILD_FP_QUANTIZER=0  DS_BUILD_OPS=1  DS_ACCELERATOR='cuda' python3 setup.py install bdist_wheel
