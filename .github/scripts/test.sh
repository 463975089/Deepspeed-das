#!/bin/bash
source /opt/dtk/env.sh

RUN_TEST_SH="${WORK_SPACE}/tests/run_test.sh"
TIMEOUT="${TIMEOUT:-3000}"
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/hcusmi:$LD_LIBRARY_PATH
if [ ! -e /tmp/cifar10-data/cifar-10-python.tar.gz ];then
       mkdir -p /tmp/cifar10-data/
       wget http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz -P /tmp/cifar10-data/
fi
# 统计失败的测试命令数。任何一项失败都要让整个脚本以非零退出码结束，
# 否则 GitLab 任务会一直显示为“成功”（假绿）。
FAILURES=0
pip install transformers==4.46.3
# 执行一条测试命令并统计失败（失败后继续执行剩余测试，最终统一判定退出码）。
record() {
    local rc=0
    "$@" || rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILURES=$((FAILURES + 1))
        echo ">>> FAILED (rc=${rc}): $*"
    fi
}

# 执行一条输出重定向到日志文件的测试命令并统计失败。
record_log() {
    local log=$1
    local rc=0
    shift
    "$@" > "$log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILURES=$((FAILURES + 1))
        echo ">>> FAILED (rc=${rc}): $* (log: ${log})"
    fi
}

run_tests() {
    cd ${WORK_SPACE}/tests
    cd ./accelerator
    record pytest -v -s ./ --junitxml=${LOG}/accelerator.xml
    cd ..

    #pytest --timeout=1400 ./model/BingBertSquad/test_e2e_squad.py #todo
    record pytest -vs --timeout=1400 ./model/Megatron_GPT2/test_common.py --junitxml=${LOG}/Megatron_GPT2_test_common.xml
    record pytest -vs --timeout=1400 ./lightning/test_simple.py --junitxml=${LOG}/lightning_test_simple.xml
    cd torch_compile
    record_log ${LOG}/torch_compile_test_compile.log python ./test_compile.py
    cd ..
    # The bulk of the suite lives under ./unit and is already xdist-aware:
    # tests/unit/common.py offsets master ports and rotates the visible
    # accelerator list per PYTEST_XDIST_WORKER, matching what
    # .github/workflows/{amd-mi200,nv-torch-latest-v100}.yml already run in CI.
    sed -i 's/reuse_dist_env = True/reuse_dist_env = False/' \
        "${WORK_SPACE}/tests/unit/runtime/zero/test_zero_tensor_fragment.py"
    record pytest -vs --timeout=1400 ./onebit/test_mpi_perf.py --junitxml=${LOG}/onebit_test_mpi_perf.xml
    record_log ${LOG}/onebit_test_nccl_backend.log python ./onebit/test_nccl_backend.py
    record pytest -vs --timeout=1400 ./onebit/test_mpi_backend.py --junitxml=${LOG}/onebit_test_mpi_backend.xml
    record_log ${LOG}/onebit_test_nccl_perf.log python ./onebit/test_nccl_perf.py
    record_log ${LOG}/onebit_test_compressed_backend.log python ./onebit/test_compressed_backend.py
    record_log ${LOG}/onebit_test_compressed_perf.log python ./onebit/test_compressed_perf.py
    record pytest -vs --timeout=1400 ./small_model_debugging/test_model.py --junitxml=${LOG}/small_model_debugging_test_model.xml
    record pytest -vs --timeout=1400 ./small_model_debugging/test_mics_config.py --junitxml=${LOG}/small_model_debugging_test_mics_config.xml
    record_log ${LOG}/small_model_debugging_stage3_test.log python ./small_model_debugging/stage3_test.py
    # 以下测试不满足上面 ./unit/ 的 pytest 过滤规则（python 脚本、非 unit 目录或带环境变量前缀），在这里单独执行
    record_log ${LOG}/hybrid_engine_hybrid_engine_test.log python ./hybrid_engine/hybrid_engine_test.py
    record_log ${LOG}/perf_adagrad_test.log python ./perf/adagrad_test.py
    record_log ${LOG}/perf_adam_test.log python ./perf/adam_test.py
    record_log ${LOG}/small_model_debugging_partial_offload_test.log python ./small_model_debugging/partial_offload_test.py
    record pytest -vs --timeout=1400 ./torch_compile/test_deepcompile_z3_release.py --junitxml=${LOG}/torch_compile_test_deepcompile_z3_release.xml
    record_log ${LOG}/torch_compile_test_deepcompile_skipped_frame.log python ./torch_compile/test_deepcompile_skipped_frame.py
    record env NCCL_P2P_DISABLE=1 pytest -vs --timeout=1400 ./unit/v1/sequence_parallelism/test_autosp_equivalence.py --junitxml=${LOG}/unit_v1_sequence_parallelism_test_autosp_equivalence.xml
    record env NCCL_P2P_DISABLE=1 pytest -vs --timeout=1400 ./unit/v1/sequence_parallelism/test_autosp_integration.py --junitxml=${LOG}/unit_v1_sequence_parallelism_test_autosp_integration.xml
}

mapfile -t TESTS < <(
    grep -E '^pytest[[:space:]]' "$RUN_TEST_SH" \
        | grep -E '\./unit/' \
        | sed -E 's/^pytest[[:space:]]+//' \
        | sed -E 's/^--timeout=[0-9]+[[:space:]]+//'
    )

TOTAL=${#TESTS[@]}
GPU_COUNT=3
export GPU_COUNT
worker() {
    local gpu_id=$1
    local idx=$2
    local fail=0

    while [[ $idx -lt $TOTAL ]]; do
         local cmd="${TESTS[$idx]}"
         # 由原始 pytest 参数还原测试名（用于日志文件）
         local test_path
         test_path="$(printf '%s\n' "$cmd" | awk '{print $NF}')"   # 取最后一个参数作为路径
         local test_name
         test_name="$(echo "$test_path" | sed -E 's|^\./||; s|/|_|g; s|\.py$||')"
         local log_file="${LOG}/${test_name}.xml"
         if HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout "${TIMEOUT}s" \
               pytest -vs --timeout="$TIMEOUT" $cmd --junitxml="$log_file" 2>&1; then
            echo "[GPU $gpu_id][$((idx+1))/$TOTAL] PASS: $cmd"
         else
            local rc=$?
            fail=$((fail + 1))
            echo "[GPU $gpu_id][$((idx+1))/$TOTAL] FAIL (rc=$rc): $cmd"
         fi
         idx=$((idx + GPU_COUNT))

    done
    echo "[GPU $gpu_id] finished: ${fail} failure(s)"
    return $((fail > 0))
}

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT="https://hf-mirror.com"

run_tests
pids=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    worker "$gpu" "$gpu" &
    pids+=($!)
done

# 汇总各 worker 的失败情况（worker 内部已统计，这里按 worker 维度再累加）。
for pid in "${pids[@]}"; do
    wait "$pid" || FAILURES=$((FAILURES + 1))
done

if [[ $FAILURES -gt 0 ]]; then
    echo "TEST SUMMARY: ${FAILURES} test command(s) FAILED"
    exit 1
fi
echo "TEST SUMMARY: all tests passed"
exit 0
