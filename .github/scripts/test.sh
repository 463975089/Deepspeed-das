#!/bin/bash

# Default to the repo root so CI checkouts work; override WORK_SPACE for a custom layout.
WORK_SPACE="${WORK_SPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# CI sets REPORTS_DIR so all logs land in one place for artifact upload.
REPORTS_DIR="${REPORTS_DIR:-${WORK_SPACE}/tests}"
TEST_LOG="${TEST_LOG:-${REPORTS_DIR}/deepspeed_test.log}"
RUN_TEST_SH="${RUN_TEST_SH:-${WORK_SPACE}/tests/run_test.sh}"
FAILED_FILE="${FAILED_FILE:-${REPORTS_DIR}/faild_test.txt}"
LOG_DIR=${REPORTS_DIR}/unit_test_logs
mkdir -p ${LOG_DIR}
TIMEOUT="${TIMEOUT:-10800}"
mkdir -p "$(dirname "${TEST_LOG}")"
: > "${TEST_LOG}"
pip install transformers==5.18.0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/hcusmi:$LD_LIBRARY_PATH
source /opt/dtk/env.sh
status=0

detect_parallelism() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        n=$(nvidia-smi -L 2>/dev/null | wc -l)
    elif command -v rocm-smi >/dev/null 2>&1; then
        n=$(rocm-smi --showid 2>/dev/null | grep -c '^HCU')
    else
        n=0
    fi
    if [ "$n" -gt 0 ] 2>/dev/null; then
        echo "$n"
    else
        echo 4
    fi
}
GPU_COUNT="${GPU_COUNT:-3}"
export GPU_COUNT
rm -rf "${LOG_DIR:?}"/*
rm -f "${FAILED_FILE}"*
record_failure() {
    local line="$1"
    (
        flock 9
        printf '%s\n' "$line" >> "$FAILED_FILE"
    ) 9>"${FAILED_FILE}.lock"
}
if [ ! -e /tmp/cifar10-data/cifar-10-python.tar.gz ];then
       mkdir -p /tmp/cifar10-data/
       wget http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz -P /tmp/cifar10-data/
fi
run_tests() {
    local test_status=0
    local rc=0

    run_one() {
        echo
        echo "===== RUN: $* ====="
        "$@"
        rc=$?
        echo "===== EXIT ${rc}: $* ====="
        test_status=$((test_status || rc))
        # Workers only report through FAILED_FILE, so record here too to give CI one combined list.
        if [ "${rc}" -eq 5 ]; then
            echo "SKIP (no tests collected): $*"
        elif [ "${rc}" -ne 0 ]; then
            echo "FAIL (rc=${rc}): $*"
            record_failure "$*"
        fi
    }

    cd ${WORK_SPACE}/tests || return 1
    cd ./accelerator || return 1
    run_one pytest ./ > ${LOG_DIR}/accelerator.log 2>&1 
    cd .. || return 1

    #pytest --timeout=1400 ./model/BingBertSquad/test_e2e_squad.py #todo
    run_one pytest -vs --timeout=1400 ./model/Megatron_GPT2/test_common.py > ${LOG_DIR}/Megatron_GPT2_test_common.log 2>&1
    run_one pytest -vs --timeout=1400 ./lightning/test_simple.py > ${LOG_DIR}/lightning_test_simple.log 2>&1
    cd torch_compile
    python  ./test_compile.py > ${LOG_DIR}/torch_compile_test_compile.log 2>&1
    cd ..
    # The bulk of the suite lives under ./unit and is already xdist-aware:
    # tests/unit/common.py offsets master ports and rotates the visible
    # accelerator list per PYTEST_XDIST_WORKER, matching what
    # .github/workflows/{amd-mi200,nv-torch-latest-v100}.yml already run in CI.
    sed -i 's/reuse_dist_env = True/reuse_dist_env = False/' \
        "${WORK_SPACE}/tests/unit/runtime/zero/test_zero_tensor_fragment.py"
    run_one pytest -vs --timeout=1400 ./onebit/test_mpi_perf.py > ${LOG_DIR}/onebit_test_mpi_perf.log 2>&1
    python ./onebit/test_nccl_backend.py > ${LOG_DIR}/onebit_test_nccl_backend.log 2>&1
    run_one pytest -vs --timeout=1400 ./onebit/test_mpi_backend.py > ${LOG_DIR}/onebit_test_mpi_backend.log 2>&1
    python ./onebit/test_nccl_perf.py > ${LOG_DIR}/onebit_test_nccl_perf.log 2>&1
    python ./onebit/test_compressed_backend.py > ${LOG_DIR}/onebit_test_compressed_backend.log 2>&1
    python ./onebit/test_compressed_perf.py > ${LOG_DIR}/onebit_test_compressed_perf.log 2>&1
    run_one pytest -vs --timeout=1400 ./small_model_debugging/test_model.py > ${LOG_DIR}/small_model_debugging_test_model.log 2>&1
    run_one pytest -vs --timeout=1400 ./small_model_debugging/test_mics_config.py > ${LOG_DIR}/small_model_debugging_test_mics_config.log 2>&1
    python ./small_model_debugging/stage3_test.py > ${LOG_DIR}/small_model_debugging_stage3_test.log 2>&1
    return ${test_status}
}

mapfile -t TESTS < <(
    grep -E '^pytest[[:space:]]' "$RUN_TEST_SH" \
        | grep -E '\./unit/' \
        | sed -E 's/^pytest[[:space:]]+//' \
        | sed -E 's/^--timeout=[0-9]+[[:space:]]+//'
    )

TOTAL=${#TESTS[@]}

worker() {
    local gpu_id=$1
    local idx=$2

    while [[ $idx -lt $TOTAL ]]; do
        local cmd="${TESTS[$idx]}"
        # 由原始 pytest 参数还原测试名（用于日志文件）
        local test_path
        test_path="$(printf '%s\n' "$cmd" | awk '{print $NF}')"   # 取最后一个参数作为路径
        local test_name
        test_name="$(echo "$test_path" | sed -E 's|^\./||; s|/|_|g; s|\.py$||')"
        local log_file="${LOG_DIR}/${test_name}.log"
        cd ${WORK_SPACE}/tests
        echo "[GPU $gpu_id][$((idx+1))/$TOTAL] RUN: $cmd"
        # 通过 CUDA_VISIBLE_DEVICES 把该测试绑定到固定 GPU
        if HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout "${TIMEOUT}s" \
               pytest -vs --timeout="$TIMEOUT" $cmd >"$log_file" 2>&1; then
            echo "[GPU $gpu_id][$((idx+1))/$TOTAL] PASS: $cmd"
        else
            local rc=$?
            # pytest 返回 5 表示没有收集到测试（被 pytest.ini 的 -m 排除或模块级 skip），不算失败
            if [ "$rc" -eq 5 ]; then
                echo "[GPU $gpu_id][$((idx+1))/$TOTAL] SKIP (no tests collected): $cmd"
            else
                echo "[GPU $gpu_id][$((idx+1))/$TOTAL] FAIL (rc=$rc): $cmd"
                record_failure "$cmd"
            fi
        fi

        idx=$((idx + GPU_COUNT))
    done
}

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT="https://hf-mirror.com"

run_tests >"${TEST_LOG}" 2>&1 &
test_pid=$!
tail -n +1 --pid="${test_pid}" -F "${TEST_LOG}" &
tail_pid=$!
wait "${test_pid}"
status=$?
wait "${tail_pid}" 2>/dev/null || true
echo "Test log is ready for upload: ${TEST_LOG}"
pids=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    worker "$gpu" "$gpu" &
    pids+=($!)
done

# 等待所有 worker 完成
wait "${pids[@]}"
wait_status=$?

# Worker exit codes are always 0, so the failure list is the only reliable signal for CI.
if [ "${status}" -ne 0 ] || [ -s "${FAILED_FILE}" ]; then
    echo "Failed tests:"
    cat "${FAILED_FILE}" 2>/dev/null
    exit 1
fi
