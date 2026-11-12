#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# 生成 manylinux 版本标识
PLATFORM="manylinux_$(ldd --version | awk '{print $NF}' | head -n1 | tr '.' '_')_x86_64"

TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:${LD_LIBRARY_PATH}"

# Patterns are matched with fnmatch, so each one ends at ".so" + "*" instead of naming a
# concrete version: a bump like librocblas.so.0 -> .4 or libavcodec.so.58 -> .60 then
# stays excluded, where a pinned name would silently bundle the library into the wheel.
auditwheel repair --plat "${PLATFORM}" --strip \
  --exclude 'libgalaxyhip.so*' \
  --exclude 'libMIOpen.so*' \
  --exclude 'librccl.so*' \
  --exclude 'libhipblas.so*' \
  --exclude 'libhipblaslt.so*' \
  --exclude 'libhipfft.so*' \
  --exclude 'libhiprand.so*' \
  --exclude 'libhipsolver.so*' \
  --exclude 'libhipsparse.so*' \
  --exclude 'libhipnn.so*' \
  --exclude 'librocblas.so*' \
  --exclude 'librocsolver.so*' \
  --exclude 'librocfft.so*' \
  --exclude 'librocfft-device-*.so*' \
  --exclude 'librocrand.so*' \
  --exclude 'librocsparse.so*' \
  --exclude 'librocm_smi64.so*' \
  --exclude 'libc10*.so*' \
  --exclude 'libtorch*.so*' \
  --exclude '_torchaudio*.so*' \
  --exclude 'libavcodec.so*' \
  --exclude 'libavdevice.so*' \
  --exclude 'libavfilter.so*' \
  --exclude 'libavformat.so*' \
  --exclude 'libavutil.so*' \
  --exclude 'libsox.so*' \
  --exclude 'libomp.so*' \
  --exclude 'libgcvm*' \
  -w dist/ \
  dist/*.whl

cd dist/
pack=`ls |grep manylinux_2_27`
wheel unpack ${pack}
base=$(ls -t|head -n 1)
cd ${base}
dir=`ls |grep info`
sed -i '/manylinux_2_27/d' ${dir}/WHEEL
cd ../
wheel pack ${base}
rm -rf ${base}
rm -rf ${pack}
cd ..
