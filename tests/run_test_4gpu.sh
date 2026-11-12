# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from tests/run_test.sh by tests/gen_run_test_buckets.py.
# Do not edit by hand; edit run_test.sh and regenerate instead.

#
# Tests that need at most 4 GPU(s). Targets appear in exactly one bucket
# (their maximum requirement), so the buckets are disjoint and together cover
# every active target in run_test.sh exactly once.
#
# transformers floor for this bucket: >= 5.0.0
#   >= 5.0.0  unit/compile/test_tp_compile.py
#
# Class-gated (these skip themselves if the installed transformers lacks
# the model class they probe):
#   unit/compile/test_tp_compile.py
#
# Avoid transformers 5.8.0 .. 5.10.1: the MoE compile coverage skips
# itself on that range (BROKEN_TRANSFORMERS_MOE_VERSIONS in
# deepspeed/compile/init_tp.py).

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

require_gpus 4

cd "${TESTS_DIR}"

FAILURES=0

record pytest --timeout=1400 ./unit/v1/sequence_parallelism/test_ulysses.py
record pytest --timeout=1400 ./unit/runtime/pipe/test_topology.py
record pytest --timeout=1400 ./unit/runtime/pipe/test_pipe.py
record pytest --timeout=1400 ./unit/runtime/utils/test_partition.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_offloadpp.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zeropp.py
record pytest --timeout=1400 ./unit/v1/autotp/test_configurable_parallel_mp.py
record pytest --timeout=1400 ./unit/v1/autotp/test_configurable_parallel_pp.py
record pytest --timeout=1400 ./unit/v1/autotp/test_autotp_training.py
record pytest --timeout=1400 ./unit/v1/moe/test_moe_tp.py
record pytest --timeout=1400 ./unit/v1/moe/test_moe.py
record pytest --timeout=1400 ./unit/checkpoint/test_pipeline.py
record pytest --timeout=1400 ./unit/checkpoint/test_zero_optimizer.py
record pytest --timeout=1400 ./unit/checkpoint/test_moe_checkpoint.py
record pytest --timeout=1400 ./unit/checkpoint/test_universal_checkpoint.py
record pytest --timeout=1400 ./unit/checkpoint/test_autotp_uc_checkpoint.py
record pytest --timeout=1400 ./unit/v1/compile/test_compile_autosp.py
record pytest --timeout=1400 ./unit/v1/moe/test_autoep_grad_parity.py
record pytest --timeout=1400 ./unit/v1/zero/test_zero_coalesce_grad_reduction.py
# transformers >= 5.0.0
record pytest --timeout=1400 ./unit/compile/test_tp_compile.py
record pytest --timeout=1400 ./unit/v1/moe/test_autoep_deepep_parity.py

exit_with_summary "4-GPU bucket"
