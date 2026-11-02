# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import os
from .async_io import AsyncIOBuilder


class GDSBuilder(AsyncIOBuilder):
    BUILD_VAR = "DS_BUILD_GDS"
    NAME = "gds"

    def __init__(self):
        super().__init__()

    def absolute_name(self):
        return f'deepspeed.ops.gds.{self.NAME}_op'

    def lib_sources(self):
        src_list = ['csrc/gds/py_lib/deepspeed_py_gds_handle.cpp', 'csrc/gds/py_lib/deepspeed_gds_op.cpp']
        return super().lib_sources() + src_list

    def sources(self):
        return self.lib_sources() + ['csrc/gds/py_lib/py_ds_gds.cpp']

    def cxx_args(self):
        if self.is_rocm_pytorch():
            return super().cxx_args() + ['-lhyfile']
        return super().cxx_args() + ['-lcufile']

    def include_paths(self):
        import torch
        import torch.utils.cpp_extension
        # 'third_party/libaio_build/include' carries the libaio headers shipped for the
        # HCU/DTK image, where the system libaio-dev package is not installed.
        SRC_INCLUDE = ['csrc/aio/py_lib', 'csrc/aio/common', 'csrc/pin_memory']
        LIBAIO_INCLUDE = ['third_party/libaio_build/include']
        if self.is_rocm_pytorch():
            HIP_INCLUDE = [os.path.join(torch.utils.cpp_extension.HIP_HOME, "include")]
            return SRC_INCLUDE + HIP_INCLUDE + LIBAIO_INCLUDE + ["/opt/hyhal/include"]
        CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        return SRC_INCLUDE + CUDA_INCLUDE + LIBAIO_INCLUDE

    def extra_ldflags(self):
        if self.is_rocm_pytorch():
            return super().extra_ldflags() + ['-L/opt/hyhal/lib', '-lhyfile']
        return super().extra_ldflags() + ['-lcufile']

    def exist_hyfile(self):
        if os.path.exists("/opt/hyhal/include/hipfile.h"):
            return True
        self.warning("Not exist hyfile.h, ignore gds compile")
        return False

    def is_compatible(self, verbose=False):
        # if self.is_rocm_pytorch():
        #     if verbose:
        #         self.warning(f'{self.NAME} is not compatible with ROCM')
        #     return False

        try:
            import torch.utils.cpp_extension
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile GDS")
            return False

        # CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
        # if CUDA_HOME is None:
        #     if verbose:
        #         self.warning("Please install torch CUDA if trying to pre-compile GDS with CUDA")
        #     return False
        # CUDA_HOME = torch.utils.cpp_extension.HIP_HOME
        # CUDA_LIB64 = os.path.join(CUDA_HOME, "lib64")
        # gds_compatible = self.has_function(funcname="cuFileDriverOpen",libraries=("cufile", ),library_dirs=(CUDA_HOME,CUDA_LIB64),
        #                                    verbose=verbose)

        cp = super().is_compatible(verbose) and self.exist_hyfile()
        return cp
