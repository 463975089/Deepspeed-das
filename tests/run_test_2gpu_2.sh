# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from tests/run_test.sh by tests/gen_run_test_buckets.py.
# Do not edit by hand; edit run_test.sh and regenerate instead.

#
# Tests that need at most 2 GPU(s), shard 2 of 2.
# The shards of one bucket are disjoint; the CI matrix runs them in
# parallel and together they cover every target exactly once.
#
# transformers: no target here asserts a floor; any release that satisfies
# requirements/requirements-dev.txt (>= 4.51.3) is enough.
#
# Class-gated (these skip themselves if the installed transformers lacks
# the model class they probe):
#   unit/v1/autotp/test_tp_plan_e2e.py

set -uo pipefail

TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# CI sets REPORTS_DIR so every pytest call also emits JUnit XML for the job
# summary; unset locally the scripts just print.
REPORTS_DIR="${REPORTS_DIR:-}"
REPORT_TAG=$(basename "${BASH_SOURCE[0]}" .sh)
RECORD_SEQ=0

# 统计本机可见的加速卡数量，探测不出来时返回空串（调用方必须区分“0 张”和“不知道”）。
# 顺序：显式限卡变量 → torch → 各厂商 smi。容器内 torch 最可靠，因为设备插件只把
# 分配到的卡暴露给容器，torch 看到的数量就是本 Pod 真正拿到的卡数。
visible_gpu_count() {
    local list="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}}"
    if [ -n "${list}" ]; then
        # 支持 "0,1,2,3" 与 "0,1,2,3,4,5,6,7" 两种写法，空项不计。
        local count=0
        local item
        local IFS=','
        for item in ${list}; do
            [ -n "${item}" ] && count=$((count + 1))
        done
        echo "${count}"
        return 0
    fi

    local from_torch
    from_torch=$(python - <<'PY' 2>/dev/null || true
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else "")
PY
)
    if [ -n "${from_torch}" ]; then
        echo "${from_torch}"
        return 0
    fi

    local smi count
    for smi in hy-smi rocm-smi nvidia-smi; do
        command -v "${smi}" >/dev/null 2>&1 || continue
        if [ "${smi}" = "nvidia-smi" ]; then
            count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        else
            count=$("${smi}" 2>/dev/null | grep -cE '^(GPU|Card series|Device)')
        fi
        if [ -n "${count}" ]; then
            echo "${count}"
            return 0
        fi
    done

    echo ""
}

# 卡数不足时跳过整档，而不是让用例以一堆 NCCL 错误的形式失败。
# 探测不出来时必须报错：静默跳过会让 CI 在没有真正测试的情况下显示通过。
require_gpus() {
    local needed=$1
    local have
    have=$(visible_gpu_count)
    if [ -z "${have}" ]; then
        echo "[error] 无法确定本机可见卡数（HIP_VISIBLE_DEVICES、torch、smi 均未给出结果）" >&2
        echo "        拒绝继续，因为无法判断 ${needed} 卡的用例能否真正运行。" >&2
        exit 1
    fi
    if [ "${have}" -lt "${needed}" ]; then
        echo "[skip] 需要 ${needed} 张卡，本机可见 ${have} 张，跳过本档"
        exit 0
    fi
}

# 记录一次失败但继续执行——整机跑完再统一判定退出码，避免第一个失败就中断后面所有用例。
record() {
    local -a cmd=("$@")
    # 每次 pytest 调用都要单独一个 XML 文件，否则后一次会覆盖前一次。
    if [ -n "${REPORTS_DIR}" ]; then
        local word
        for word in "$@"; do
            if [ "${word}" = "pytest" ]; then
                mkdir -p "${REPORTS_DIR}"
                cmd+=(--junitxml="${REPORTS_DIR}/${REPORT_TAG}-$(printf '%03d' "${RECORD_SEQ}").xml")
                break
            fi
        done
    fi
    RECORD_SEQ=$((RECORD_SEQ + 1))

    local rc=0
    "${cmd[@]}" || rc=$?
    if [ "${rc}" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
        echo ">>> FAILED (rc=${rc}): $*"
    fi
    return 0
}

exit_with_summary() {
    local label=$1
    if [ "${FAILURES:-0}" -ne 0 ]; then
        echo "[${label}] ${FAILURES} 条用例失败"
        exit 1
    fi
    echo "[${label}] 全部通过"
}

require_gpus 2

cd "${TESTS_DIR}"

FAILURES=0

record pytest --timeout=1400 ./unit/inference/test_checkpoint_sharding.py
record pytest --timeout=1400 ./torch_compile/test_deepcompile_z3_release.py
record pytest --timeout=1400 ./unit/linear/test_linear.py
record pytest --timeout=1400 ./unit/v1/autotp/test_autotp_custom_patterns.py
record pytest --timeout=1400 ./unit/v1/autotp/test_tp_plan_e2e.py
record env NCCL_P2P_DISABLE=1 pytest --timeout=1400 ./unit/v1/sequence_parallelism/test_autosp_equivalence.py
record env NCCL_P2P_DISABLE=1 pytest --timeout=1400 ./unit/v1/sequence_parallelism/test_autosp_integration.py
record pytest --timeout=1400 ./unit/v1/half_precision/test_with_autocast.py
record pytest --timeout=1400 ./unit/v1/zero/test_offload_states.py
record pytest --timeout=1400 ./unit/v1/zero/test_stage2_flatten_on_gpu.py
record pytest --timeout=1400 ./unit/v1/zero/test_zero_hook_count_regression.py
record pytest --timeout=1400 ./unit/v1/zero/test_zero_user_backward.py
record pytest --timeout=1400 ./unit/inference/quantization/test_intX_quantization.py
record pytest --timeout=1400 ./unit/elasticity/test_elastic.py
record pytest --timeout=1400 ./unit/v1/compile/test_compile_zero.py
record pytest --timeout=1400 ./unit/v1/zero/test_zero_autocast.py
record pytest --timeout=1400 ./unit/v1/zero/test_zero.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_allreduce_fetch_params.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_autocast_mixed_dtype.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_linear_direct_init.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_shared_loss_gradient.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_stale_handle_wait.py
record pytest --timeout=1400 ./unit/ulysses_alst/test_dim_zero_all_to_all.py
record pytest --timeout=1400 ./unit/v1/autotp/test_autotp_multiple_models.py
record pytest --timeout=1400 ./unit/v1/compile/test_offload_activation.py
record pytest --timeout=1400 ./unit/v1/compile/test_offload_opt_states.py
record pytest --timeout=1400 ./unit/v1/moe/test_autoep_fused_parity.py
record pytest --timeout=1400 ./unit/v1/ops/muon/test_muon.py
record pytest --timeout=1400 ./unit/v1/ops/muon/test_muon_checkpoint.py
record pytest --timeout=1400 ./unit/v1/ops/muon/test_muon_overflow.py
record pytest --timeout=1400 ./unit/v1/ops/muon/test_muon_partial_training.py

exit_with_summary "2-GPU bucket, shard 2 of 2"
