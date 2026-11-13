#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# The glibc of the build image decides the tag (rocky 8.6 -> manylinux_2_28).
# The repaired wheel keeps that tag: pip then refuses the install on machines
# with older glibc instead of failing later at import time with a missing
# symbol. No tail-stage retagging happens, so auditwheel's work is the product.
PLATFORM="manylinux_$(ldd --version | awk '{print $NF}' | head -n1 | tr '.' '_')_x86_64"

# The image does not necessarily ship auditwheel; setup_hcu.sh already pointed
# pip at the devpi index, which serves it.
pip install auditwheel
pip install patchelf
# Fail loudly right here if an older patchelf still wins PATH resolution.
patchelf --version
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

# auditwheel leaves the input wheel next to the repaired one. The deliverable
# keeps its manylinux tag, so the raw linux_x86_64 build has to go away or
# CIUpload and the wheel artifact would both receive two files.
for wheel in dist/*.whl; do
    case "${wheel}" in
        *manylinux*) ;;
        *) rm -f "${wheel}" ;;
    esac
done
