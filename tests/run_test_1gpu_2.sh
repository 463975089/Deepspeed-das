# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from tests/run_test.sh by tests/gen_run_test_buckets.py.
# Do not edit by hand; edit run_test.sh and regenerate instead.

#
# Tests that need at most 1 GPU(s), shard 2 of 4.
# The shards of one bucket are disjoint; the CI matrix runs them in
# parallel and together they cover every target exactly once.
#
# transformers floor for this shard: >= 5.0.0
#   >= 5.0.0  unit/v1/moe/test_autoep_unit.py
#
# Class-gated (these skip themselves if the installed transformers lacks
# the model class they probe):
#   unit/v1/moe/test_autoep_unit.py

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

require_gpus 1

cd "${TESTS_DIR}"

FAILURES=0

record pytest --timeout=1400 ./unit/ops/adam/test_hybrid_adam.py
record pytest --timeout=1400 ./unit/ops/adam/test_zf_torch_adam.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_attention.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_softmax.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_moe_res_matmult.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_bias_gelu.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_layer_norm.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_bias_geglu.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_bias_relu.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_bias_add.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_residual_add.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_gelu.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_matmul.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_rms_norm.py
record pytest --timeout=1400 ./unit/ops/transformer/inference/test_rope.py
record pytest --timeout=1400 ./unit/hybrid_engine/test_he_all.py
record pytest --timeout=1400 ./unit/hybrid_engine/test_he_lora.py
record pytest --timeout=1400 ./unit/hybrid_engine/test_he_llama.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_pre_ln.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_post_ln.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_blas_linear.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_gated_activation.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_bias_activation.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/core_ops/test_rms_norm.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/ragged_ops/test_atom_builder.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/ragged_ops/test_moe_scatter.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/ragged_ops/test_logits_gather.py
record pytest --timeout=1400 ./unit/inference/v2/kernels/ragged_ops/test_ragged_embed.py
record python ./hybrid_engine/hybrid_engine_test.py
record python ./perf/adagrad_test.py
record python ./perf/adam_test.py
record python ./small_model_debugging/partial_offload_test.py
record pytest --timeout=1400 ./unit/compile/test_inductor_aot_kwargs.py
record pytest --timeout=1400 ./unit/compile/test_zero3_grad_dtype.py
record pytest --timeout=1400 ./unit/linear/test_quant_param.py
record pytest --timeout=1400 ./unit/module_inject/test_tp_plan_converter.py
record pytest --timeout=1400 ./unit/v1/nvme/test_fast_file_writer_fd_close.py
record pytest --timeout=1400 ./unit/ops/deepspeed4science/test_evoformer_attn_builder.py
record pytest --timeout=1400 ./unit/ops/test_op_builder.py
record pytest --timeout=1400 ./unit/runtime/half_precision/test_fp8.py
record pytest --timeout=1400 ./unit/runtime/tensor_parallel/test_autotp_universal_checkpoint.py
record pytest --timeout=1400 ./unit/runtime/test_precision_config_loss_scale.py
record pytest --timeout=1400 ./unit/runtime/test_tp_plan_extraction.py
record pytest --timeout=1400 ./unit/runtime/zero/test_zero_late_module_attach.py
record pytest --timeout=1400 ./unit/v1/compile/test_compile_fx.py
record pytest --timeout=1400 ./unit/v1/compile/test_selective_gather.py
record pytest --timeout=1400 ./unit/v1/half_precision/test_mixed_precision_dtype.py
# transformers >= 5.0.0
record pytest --timeout=1400 ./unit/v1/moe/test_autoep_unit.py
record pytest --timeout=1400 ./unit/v1/moe/test_fused_expert_layout.py

exit_with_summary "1-GPU bucket, shard 2 of 4"
