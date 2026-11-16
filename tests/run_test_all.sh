# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from tests/run_test.sh by tests/gen_run_test_buckets.py.
# Do not edit by hand; edit run_test.sh and regenerate instead.

#
# Runs every generated shard script, smallest card count first.
#
#   ./run_test_all.sh          run every script
#   ./run_test_all.sh 4        start at the 4-GPU bucket
set -uo pipefail

TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

START=${1:-1}
FAILURES=0

if [ "1" -ge "$START" ]; then
    echo "[run_test_all] === run_test_1gpu_1.sh ==="
    if ! "${TESTS_DIR}/run_test_1gpu_1.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_1gpu_1.sh FAILED"
    fi
fi

if [ "1" -ge "$START" ]; then
    echo "[run_test_all] === run_test_1gpu_2.sh ==="
    if ! "${TESTS_DIR}/run_test_1gpu_2.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_1gpu_2.sh FAILED"
    fi
fi

if [ "1" -ge "$START" ]; then
    echo "[run_test_all] === run_test_1gpu_3.sh ==="
    if ! "${TESTS_DIR}/run_test_1gpu_3.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_1gpu_3.sh FAILED"
    fi
fi

if [ "1" -ge "$START" ]; then
    echo "[run_test_all] === run_test_1gpu_4.sh ==="
    if ! "${TESTS_DIR}/run_test_1gpu_4.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_1gpu_4.sh FAILED"
    fi
fi

if [ "2" -ge "$START" ]; then
    echo "[run_test_all] === run_test_2gpu_1.sh ==="
    if ! "${TESTS_DIR}/run_test_2gpu_1.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_2gpu_1.sh FAILED"
    fi
fi

if [ "2" -ge "$START" ]; then
    echo "[run_test_all] === run_test_2gpu_2.sh ==="
    if ! "${TESTS_DIR}/run_test_2gpu_2.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_2gpu_2.sh FAILED"
    fi
fi

if [ "3" -ge "$START" ]; then
    echo "[run_test_all] === run_test_3gpu.sh ==="
    if ! "${TESTS_DIR}/run_test_3gpu.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_3gpu.sh FAILED"
    fi
fi

if [ "4" -ge "$START" ]; then
    echo "[run_test_all] === run_test_4gpu.sh ==="
    if ! "${TESTS_DIR}/run_test_4gpu.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_4gpu.sh FAILED"
    fi
fi

if [ "8" -ge "$START" ]; then
    echo "[run_test_all] === run_test_8gpu.sh ==="
    if ! "${TESTS_DIR}/run_test_8gpu.sh"; then
        FAILURES=$((FAILURES + 1))
        echo "[run_test_all] run_test_8gpu.sh FAILED"
    fi
fi

if [ "$FAILURES" -ne 0 ]; then
    echo "[run_test_all] ${FAILURES} script(s) failed"
    exit 1
fi
echo "[run_test_all] all scripts passed"
