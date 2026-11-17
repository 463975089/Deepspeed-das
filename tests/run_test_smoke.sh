#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-300}

cd "${TESTS_DIR}"

printf '%s\n' '[smoke] 加速器初始化'
# This test resolves ds_config.json relative to the accelerator directory.
(
    cd accelerator
    pytest --timeout="${SMOKE_TIMEOUT}" ./
)

printf '%s\n' '[smoke] 核心配置、运行时、分布式、检查点和编译'
pytest --timeout="${SMOKE_TIMEOUT}" \
    ./unit/runtime/test_ds_config_model.py \
    ./unit/runtime/zero/test_zero_config.py \
    ./unit/runtime/pipe/test_pipe_schedule.py \
    ./unit/compile/test_inductor_aot_kwargs.py \
    ./unit/module_inject/test_tp_plan_converter.py \
    ./unit/runtime/test_ds_initialize.py::TestNoOptim \
    ./unit/comm/test_dist.py::TestInit \
    ./unit/checkpoint/test_latest_checkpoint.py \
    ./torch_compile/test_deepcompile_z3_release.py::TestDeepCompileZ3ReleaseStorage::test_storage_resized_to_zero_after_release_single_use

