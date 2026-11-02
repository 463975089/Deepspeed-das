# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

wget http://42.228.13.241:18000/customized/amdsmi/amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl -O amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl
pip install amdsmi-24.5.3%2B02cbffb.dirty-py3-none-any.whl
PYTHON_VERSION=$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')

if [[ $PYTHON_VERSION == "3.12" ]];then
   pip install scipy==1.14.1
fi
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
pip install numpy==1.24.3
