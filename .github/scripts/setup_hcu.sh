#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Sets up the HCU environment inside the job container: merge the PR under test
# (manual runs only), configure pip, install system libraries, optionally the
# compiler, then the PyTorch stack. Every job that runs repository code needs
# this, because each ARC job gets a fresh container with no persisted state.
#
# This is a plain script instead of a composite action on purpose: ARC's
# kubernetes container mode resolves local composite actions in the runner pod,
# which cannot read the files that actions/checkout wrote into the job pod, so
# "uses: ./.github/actions/..." fails with "Can't find 'action.yml'". Plain
# "run:" steps execute inside the job pod and never hit that path.
#
# Inputs arrive through the environment:
#   PR_NUMBER         PR to merge into the checked-out base branch; empty tests
#                     the branch as-is
#   TORCH_VERSION     torch version to install
#   DEVPI_URL         devpi index that serves the torch wheels
#   PIP_TRUSTED_HOST  trusted host for the devpi index
#   INSTALL_COMPILER  'true' only for the build job; the test jobs skip it to
#                     keep their startup short
#   GH_TOKEN          github token, needed only when PR_NUMBER is set

set -euo pipefail

: "${TORCH_VERSION:?TORCH_VERSION must be set}"
: "${DEVPI_URL:?DEVPI_URL must be set}"
: "${PIP_TRUSTED_HOST:?PIP_TRUSTED_HOST must be set}"

# Manual runs may ask to validate an open PR; PR runs already checked out the
# base branch and must merge the head on top of it.
if [ -n "${PR_NUMBER:-}" ]; then
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    # The repo is private, so the fetch needs a token. Pass it only to this
    # command so it never lands in .git/config.
    auth=$(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 -w0)
    git -c http.extraheader="AUTHORIZATION: basic ${auth}" \
        fetch "https://github.com/${GITHUB_REPOSITORY}.git" "pull/${PR_NUMBER}/head:pr"
    git merge --no-ff --no-edit pr
fi

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
    index-url = ${DEVPI_URL}/nightly/dtk2610/+simple/
    trusted-host = ${PIP_TRUSTED_HOST}
EOF

# Required at runtime by the compiled operator libraries, not only at build time,
# so every job installs it.
yum install -y libaio-devel libaio

if [ "${INSTALL_COMPILER:-false}" = "true" ]; then
    bash .github/scripts/hcu_llvm_installer.sh
fi

source /opt/dtk/env.sh
pip install torch==${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh torchvision ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh triton ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh apex ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh flash_attn ${TORCH_VERSION}

# ${{ github.workspace }} resolves to the host path inside a container job,
# so derive the reports directory from the container's own filesystem.
echo "REPORTS_DIR=${GITHUB_WORKSPACE}/reports" >> "$GITHUB_ENV"
