#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Split tests/run_test.sh into per-GPU-count runner scripts.

Every runnable target in run_test.sh is assigned to exactly one bucket file,
run_test_<N>gpu[_<K>.sh], where N is the maximum world_size that target needs
and K numbers the parallel shards of big buckets (see SPLIT_COUNTS). The
scripts are disjoint: running them all runs the suite exactly once. The CI
matrix in .github/workflows/pr-test-hcu.yml runs one script per job.

world_size is derived from the DistributedTest contract in tests/unit/common.py:
the class attribute ``world_size`` (default 2) maxed against any per-method
``@pytest.mark.world_size(N)`` override. Targets that never start worker
processes run in a single process and land in the 1-GPU bucket.

Only version floors that the test sources actually assert are recorded in the
script headers; targets that merely probe for a model class are marked
class-gated and skip themselves when the installed transformers lacks it. The
scripts do not manage the transformers install themselves — the CI pins the
validated release before running them.

Usage:
    python tests/gen_run_test_buckets.py
"""

import ast
import os
import re
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_TEST = os.path.join(TESTS_DIR, "run_test.sh")

DEFAULT_WORLD_SIZE = 2  # DistributedTest's documented default

# Highest transformers version the suite is known to install and pass with,
# matching .github/scripts/test.sh.
PINNED_TRANSFORMERS = "5.18.0"

# Targets whose Mixtral coverage needs the static-shape experts implementation
# that first shipped in transformers 5.0.
MIXTRAL_5X_TARGETS = {"unit/compile/test_tp_compile.py"}

# transformers releases the suite knows to be broken, mirroring
# deepspeed.compile.init_tp.BROKEN_TRANSFORMERS_MOE_VERSIONS. A test cannot pass
# on these, so a bucket that needs one of them refuses to install into the gap.
BROKEN_TRANSFORMERS_RANGES = [("5.8.0", "5.10.1")]

# How many parallel shard files each big bucket is split into. The CI matrix runs
# one shard per job; keep every count at or below the hosting pool's maxRunners
# (1card 8, 2card 2, 4card 2, 8card 1) or the extra shards would only queue.
# Buckets absent from this map stay in a single file.
SPLIT_COUNTS = {1: 4, 2: 2}

# Targets that need a transformers release older than the suite's pin because they
# use an API that transformers 5.x removed. They stay in run_test.sh (the single
# source of truth) but are excluded from the generated bucket scripts; the CI
# workflow runs them in a dedicated step after installing the version they need.
LEGACY_TRANSFORMERS_TARGETS = {
    "unit/inference/test_model_profiling.py",
}


def in_broken_range(version):
    if version is None:
        return None
    for first_broken, first_fixed in BROKEN_TRANSFORMERS_RANGES:
        if version_key(first_broken) <= version_key(version) < version_key(first_fixed):
            return (first_broken, first_fixed)
    return None


def version_key(v):
    return tuple(int(p) for p in re.findall(r"\d+", v))


def higher(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if version_key(a) >= version_key(b) else b


def parse_run_test(path):
    """Return every runnable target, plus the disabled lines kept for parity."""
    targets = []
    disabled = []
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1
        if not stripped:
            continue

        commented = stripped.startswith("#")
        body = stripped.lstrip("#").strip()

        # A leading `VAR=value` assignment is not a command, so `record VAR=1 pytest ...`
        # would try to execute "VAR=1". Route it through `env` instead.
        env_prefix = []
        while True:
            m_env = re.match(r"^([A-Za-z_][A-Za-z0-9_]*=\S*)\s+(.*)$", body)
            if not m_env:
                break
            env_prefix.append(m_env.group(1))
            body = m_env.group(2)

        m = re.search(r"(?:^|\s)(pytest|python)\s+(.+)$", body)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2)

        if env_prefix:
            body = "env %s %s" % (" ".join(env_prefix), body)

        target = None
        for tok in rest.split():
            if tok.startswith("./") or tok.endswith(".py"):
                target = tok
        if target is None:
            continue

        entry = dict(line=i,
                     raw=stripped,
                     kind=kind,
                     rel=target.lstrip("./"),
                     full_cmd=body,
                     cwd=None)

        # `cd ./accelerator` / `pytest ./` / `cd ..` is one target that must run
        # from inside the accelerator directory.
        if target.lstrip("./") == "" and kind == "pytest":
            entry["cwd"] = "accelerator"
            entry["rel"] = "accelerator"

        if commented:
            disabled.append(entry)
        else:
            targets.append(entry)
    return targets, disabled


def _int_of(node):
    """Literal int, tolerating ``2``, ``[2]`` and ``[2, 4]``."""
    try:
        value = ast.literal_eval(node)
    except Exception:
        try:
            value = ast.unparse(node)
        except Exception:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        ints = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
        return max(ints) if ints else None
    return None


def world_size_of(path):
    """Return (max_world_size, starts_worker_processes).

    Inheritance is resolved transitively: a class that derives from a
    DistributedTest subclass through an intermediate base still launches
    workers, and its world_size is whatever the nearest ancestor that sets it
    declares, otherwise the DistributedTest default.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return 1, False

    classes = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [ast.unparse(b) for b in node.bases]
        own_ws = None
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "world_size":
                        own_ws = _int_of(stmt.value)
        method_ws = []
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in stmt.decorator_list:
                if isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("mark.world_size"):
                    if dec.args:
                        value = _int_of(dec.args[0])
                        if value:
                            method_ws.append(value)
        classes[node.name] = dict(bases=bases, own_ws=own_ws, method_ws=method_ws)

    ROOTS = ("DistributedTest", "DistributedFixture")

    def base_name(base):
        # Strip subscripts/generics so `DistributedTest` and `Foo[T]` both resolve.
        return base.split("[")[0].strip()

    def resolved_ws(name, seen=None):
        """(is_distributed, effective_world_size) for a class in this file."""
        seen = seen or set()
        if name in seen or name not in classes:
            return False, None
        seen.add(name)

        info = classes[name]
        for base in info["bases"]:
            if base_name(base) in ROOTS:
                return True, info["own_ws"] if info["own_ws"] else DEFAULT_WORLD_SIZE
        inherited = None
        for base in info["bases"]:
            is_dist, ws = resolved_ws(base_name(base), seen)
            if is_dist:
                inherited = ws
        if inherited is None:
            return False, None
        return True, info["own_ws"] if info["own_ws"] else inherited

    max_ws = 0
    spawned = False
    for name, info in classes.items():
        is_dist, ws = resolved_ws(name)
        if not is_dist:
            continue
        spawned = True
        candidates = [ws] + info["method_ws"]
        max_ws = max(max_ws, max(candidates))

    if not spawned:
        return 1, False
    return max_ws, True


def transformers_floor(source):
    """Highest numeric transformers floor the file asserts, or None."""
    floor = None
    for m in re.finditer(r'importorskip\(\s*["\']transformers["\']\s*,\s*minversion\s*=\s*["\']([\d.]+)["\']',
                         source):
        floor = higher(floor, m.group(1))
    for m in re.finditer(r'min_version\s*=\s*["\']([\d.]+)["\']', source):
        floor = higher(floor, m.group(1))
    return floor


def is_class_gated(source):
    """True when the file skips itself if the installed transformers lacks a class."""
    patterns = [
        r'getattr\(\s*transformers\s*,\s*["\'][A-Z]',
        r'hasattr\(\s*transformers\s*,\s*["\'][A-Z]',
        r'importorskip\(\s*["\']transformers["\']\s*\)',
    ]
    return any(re.search(p, source) for p in patterns)


def assign(targets):
    buckets = {}
    meta = {}
    for t in targets:
        path = os.path.join(TESTS_DIR, t["rel"].replace("/", os.sep))
        if not os.path.exists(path) and t["cwd"] is None:
            raise SystemExit("line %d: missing file %s" % (t["line"], t["rel"]))
        if t["rel"] == "accelerator" and not os.path.isdir(os.path.join(TESTS_DIR, "accelerator")):
            raise SystemExit("line %d: missing accelerator directory" % t["line"])

        if t["rel"].endswith(".py"):
            ws, spawned = world_size_of(path)
        else:
            ws, spawned = 1, False

        # Bare `python` targets that never init a process group run in one
        # process; the ones that do need an external launcher.
        if t["kind"] == "python":
            ws = 2 if spawned else 1

        source = open(path, encoding="utf-8", errors="replace").read() if path.endswith(".py") else ""
        floor = transformers_floor(source)
        broken = None
        if t["rel"] in MIXTRAL_5X_TARGETS:
            floor = higher(floor, "5.0.0")
            # The Mixtral branch skips itself on this range; record it so the bucket
            # can refuse to run into a release where the coverage silently vanishes.
            broken = BROKEN_TRANSFORMERS_RANGES[0]
        meta[t["rel"]] = dict(floor=floor, gated=is_class_gated(source), broken=broken)

        buckets.setdefault(ws, []).append(t)
    return buckets, meta


LICENSE = """# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from tests/run_test.sh by tests/gen_run_test_buckets.py.
# Do not edit by hand; edit run_test.sh and regenerate instead.
"""


# Helpers inlined into every generated script so each one is self-contained —
# no shared source file to keep in sync. The CI pins the validated transformers
# release before running, so version management lives in the workflow, not here.
COMMON_HELPERS = '''\
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
'''


def bucket_filename(ws, shard_idx, shard_count):
    """run_test_3gpu.sh, or run_test_1gpu_2.sh for the shard files of a big bucket."""
    if shard_count > 1:
        return "run_test_%dgpu_%d.sh" % (ws, shard_idx + 1)
    return "run_test_%dgpu.sh" % ws


def bucket_header(ws, targets, meta, shard_idx=0, shard_count=1):
    required = sorted({meta[t["rel"]]["floor"] for t in targets if meta[t["rel"]]["floor"]},
                      key=version_key)
    top = required[-1] if required else None
    lines = [LICENSE]
    lines.append("#")
    if shard_count > 1:
        lines.append("# Tests that need at most %d GPU(s), shard %d of %d." % (ws, shard_idx + 1, shard_count))
        lines.append("# The shards of one bucket are disjoint; the CI matrix runs them in")
        lines.append("# parallel and together they cover every target exactly once.")
    else:
        lines.append("# Tests that need at most %d GPU(s). Targets appear in exactly one bucket" % ws)
        lines.append("# (their maximum requirement), so the buckets are disjoint and together cover")
        lines.append("# every active target in run_test.sh exactly once.")
    lines.append("#")
    if top:
        scope = "shard" if shard_count > 1 else "bucket"
        lines.append("# transformers floor for this %s: >= %s" % (scope, top))
        for t in targets:
            floor = meta[t["rel"]]["floor"]
            if floor:
                lines.append("#   >= %-6s %s" % (floor, t["rel"]))
    else:
        lines.append("# transformers: no target here asserts a floor; any release that satisfies")
        lines.append("# requirements/requirements-dev.txt (>= 4.51.3) is enough.")
    gated = [t["rel"] for t in targets if meta[t["rel"]]["gated"]]
    if gated:
        lines.append("#")
        lines.append("# Class-gated (these skip themselves if the installed transformers lacks")
        lines.append("# the model class they probe):")
        for rel in gated:
            lines.append("#   %s" % rel)

    broken = sorted({meta[t["rel"]]["broken"] for t in targets if meta[t["rel"]]["broken"]})
    if broken:
        lines.append("#")
        for first_broken, first_fixed in broken:
            lines.append("# Avoid transformers %s .. %s: the MoE compile coverage skips" % (first_broken, first_fixed))
            lines.append("# itself on that range (BROKEN_TRANSFORMERS_MOE_VERSIONS in")
            lines.append("# deepspeed/compile/init_tp.py).")
    return "\n".join(lines) + "\n"


def write_bucket(ws, targets, meta, shard_idx, shard_count):
    if shard_count > 1:
        # Contiguous near-equal chunks; boundaries stay stable across regenerations
        # as long as run_test.sh keeps its target order.
        size = len(targets)
        chunk = targets[size * shard_idx // shard_count:size * (shard_idx + 1) // shard_count]
    else:
        chunk = targets

    out = [bucket_header(ws, chunk, meta, shard_idx, shard_count)]
    out.append(COMMON_HELPERS)
    out.append("require_gpus %d\n" % ws)
    out.append('cd "${TESTS_DIR}"\n')
    out.append("FAILURES=0\n")

    for t in chunk:
        floor = meta[t["rel"]]["floor"]
        if floor:
            out.append("# transformers >= %s" % floor)
        if t["cwd"]:
            # A subshell would swallow the FAILURES increment, so cd and return.
            out.append('pushd "${TESTS_DIR}/%s" >/dev/null' % t["cwd"])
            out.append("record %s" % t["full_cmd"])
            out.append("popd >/dev/null")
        else:
            out.append("record %s" % t["full_cmd"])

    out.append("")
    if shard_count > 1:
        out.append('exit_with_summary "%d-GPU bucket, shard %d of %d"' % (ws, shard_idx + 1, shard_count))
    else:
        out.append('exit_with_summary "%d-GPU bucket"' % ws)

    path = os.path.join(TESTS_DIR, bucket_filename(ws, shard_idx, shard_count))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    os.chmod(path, 0o755)
    return len(chunk)


def write_all(scripts):
    """Run every generated script, smallest card count first.

    scripts: [(ws, filename), ...] in run order. Each script checks the visible
    card count itself: it skips when the host has too few cards and refuses when
    the count cannot be detected, so no extra gating is needed here.
    """
    out = [LICENSE]
    out.append("#")
    out.append("# Runs every generated shard script, smallest card count first.")
    out.append("#")
    out.append("#   ./run_test_all.sh          run every script")
    out.append("#   ./run_test_all.sh 4        start at the 4-GPU bucket")
    out.append("set -uo pipefail\n")
    out.append('TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\n')
    out.append("START=${1:-1}")
    out.append("FAILURES=0")
    for ws, name in scripts:
        out.append("")
        out.append('if [ "%d" -ge "$START" ]; then' % ws)
        out.append('    echo "[run_test_all] === %s ==="' % name)
        out.append('    if ! "${TESTS_DIR}/%s"; then' % name)
        out.append("        FAILURES=$((FAILURES + 1))")
        out.append('        echo "[run_test_all] %s FAILED"' % name)
        out.append("    fi")
        out.append("fi")
    out.append("")
    out.append('if [ "$FAILURES" -ne 0 ]; then')
    out.append('    echo "[run_test_all] ${FAILURES} script(s) failed"')
    out.append("    exit 1")
    out.append("fi")
    out.append('echo "[run_test_all] all scripts passed"')

    path = os.path.join(TESTS_DIR, "run_test_all.sh")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    os.chmod(path, 0o755)


def main():
    targets, disabled = parse_run_test(RUN_TEST)
    legacy = [t["rel"] for t in targets if t["rel"] in LEGACY_TRANSFORMERS_TARGETS]
    targets = [t for t in targets if t["rel"] not in LEGACY_TRANSFORMERS_TARGETS]
    buckets, meta = assign(targets)

    scripts = []
    counts = {}
    for ws in sorted(buckets):
        shard_count = SPLIT_COUNTS.get(ws, 1)
        for idx in range(shard_count):
            name = bucket_filename(ws, idx, shard_count)
            counts[name] = write_bucket(ws, buckets[ws], meta, idx, shard_count)
            scripts.append((ws, name))
    write_all(scripts)

    total = sum(len(v) for v in buckets.values())
    print("active targets: %d (plus %d disabled kept for parity)" % (total, len(disabled)))
    if legacy:
        print("  (%d target(s) needing an older transformers release run in a dedicated CI step)" % len(legacy))
    for _, name in scripts:
        print("  %-22s %d targets" % (name, counts[name]))
    print("  run_test_all.sh")
    print("pinned transformers: %s" % PINNED_TRANSFORMERS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
