---
name: hcu-build
description: 在 HCU（海光 DCU）环境中编译并安装 DeepSpeed，流程与 .github/workflows/pr-test-hcu.yml 保持一致。当用户要求在 HCU/DCU 上编译、打 whl 包、安装 DeepSpeed，或复现 CI 的编译步骤时使用。
---

# 在 HCU 上编译 DeepSpeed

`.github/workflows/pr-test-hcu.yml` 和 `.github/scripts/compile.sh` 是唯一的依据。DTK、torch、AICC 和 pip 源的版本经常变化，所以本文件不写具体版本，每次都从 workflow 里读取。

所有命令都在仓库根目录用 bash 执行。每一步确认成功后再进行下一步；失败时先报告错误，不要跳过。
## 0. 前提条件
编译deepspeed需要dtk、aicc、torch和一些依赖包。请确保这些依赖已经安装在HCU环境中，并且可以通过环境变量访问。
dtk的安装方式如下:
```bash
wget http://42.228.13.241:18000/dtk-pkg/dtk26.10/DTK-26.10-Rocky8.6-x86_64.tar.gz
tar -xzf DTK-26.10-Rocky8.6-x86_64.tar.gz -C /opt/
ln -s /opt/dtk-26.10-Rocky8.6-x86_64 /opt/dtk
source /opt/dtk/env.sh
```
安装torch的方式如下:
```bash
cat > ~/.pip/pip.conf << EOF
[global]
    index-url = http://10.16.1.201:9929/nightly/dtk2610/+simple/
    trusted-host = 10.16.1.201
EOF
pip install torch==${TORCH_VERSION}
```
aicc的安装方式如下:
```bash
bash .github/scripts/hcu_llvm_installer.sh
```
# 用法:
#   ./{script}                                             # 默认装最新版本(版本号最高, 该版本内日期最新)
#   ./{script} --major 2.0.0                               # 指定大版本号, 装该版本中日期最新的包
#   ./{script} --version 2.0.0+dtk26042.2609071657.ga995da # 指定完整版本, 精确匹配该包
#   ./{script} --commit g16beee                            # 指定 commit, 装对应包(可省略前缀 g, 支持短 hash 前缀匹配)
#   ./{script} --dtk-dir /opt/dtk                          # 指定 dtk 安装目录, 默认 /opt/dtk
#   ./{script} --dry-run                                   # 只打印将要下载安装的包, 不实际下载安装
#   ./{script} --keep                                      # 安装成功后保留下载的 run 包(默认装完自动删除)
#
# 安装方式: yes | bash <包> --dtk_dir <目录>



## 1. 安装依赖库

deepspeed需要libaio-devel和libaio库，请确保这些库已经安装在HCU环境中。可以使用以下命令安装：
```bash
yum install -y libaio-devel libaio
```
deepspeed测试时还需要一些对应的Python包，请确保这些包已经安装在HCU环境中。可以使用以下命令安装：
```bash
source /opt/dtk/env.sh
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
    index-url = http://10.16.1.201:9929/nightly/dtk2610/+simple/
    trusted-host = 10.16.1.201
EOF
bash .github/scripts/install_with_torch.sh torchvision ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh triton ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh apex ${TORCH_VERSION}
bash .github/scripts/install_with_torch.sh flash_attn ${TORCH_VERSION}
```
install_with_torch.sh 脚本会根据当前安装的 torch 版本，自动选择合适的包版本进行安装。
# DeepSpeed Team
# install_with_torch.sh - install a PyTorch ecosystem package matching a given torch version
#
# Usage:
#   ./install_with_torch.sh <package> <torch_version> [package_version] [--dry-run] [-h|--help]
#
# Examples:
#   ./install_with_torch.sh torchaudio 2.10.0
#   ./install_with_torch.sh torchvision "2.10"
#   ./install_with_torch.sh triton 2.10 3.6.0
#   ./install_with_torch.sh torchaudio 2100 --dry-run
#   ./install_with_torch.sh triton 2.10 3.6.0 --dry-run
#
# Accepted torch version formats:
#   2.10.0  -> torch2100
#   2.10    -> torch2100 (preferred), then torch210
#   2.1     -> torch2100 (preferred), then torch210, then torch21
#   2100    -> torch2100
#
# Accepted package version: the leading segment of a wheel version, e.g.
#   "3.6.0" matches 3.6.0+dtk2604.torch2100.<date>.g<hash>
#   "3.6"   matches 3.6.x (prefix match)
#
# Exit codes:
#   0  success
#   1  generic failure
#   2  invalid arguments
#   3  pip index query failed
#   4  no matching version found
#   5  pip install failed

## 2. 执行编译脚本

.github/scripts/compile.sh 是 deepspeed的编译脚本，编译时会首先判断是否安装了dtk,是否安装了torch,是否安装了AICC，如果没有则退出，否则会继续编译deepspeed。请确保在HCU环境中执行以下命令：
```bash
bash .github/scripts/compile.sh
```
编译成功后会在dist目录里面生成whl包，用户可以使用pip install dist/*.whl安装deepspeed。



## 3. 验证

在仓库外执行，确保导入的是安装好的包，而不是源码目录：

```bash
source /opt/dtk/env.sh
(cd /tmp && python -c "import deepspeed; print(deepspeed.__version__, deepspeed.__file__)")
ds_report
```

向用户报告：

- whl 的文件名
- `deepspeed.__version__`
- `ds_report` 里各个算子的编译状态，以及 torch 和加速器的信息

只有这一步通过，才能说编译成功。

## 4. 执行单元测试

单元测试可执行测试脚本，会并行测试加快测试进度
```bash
bash .github/scripts/test.sh
```


## 排错

| 现象 | 处理 |
| --- | --- |
| `import torch` 失败 | `compile.sh` 把这条命令的 stderr 丢掉了一次，要单独运行 `python -c "import torch"` 看完整报错 |
| 导入 deepspeed 或运行时缺少 Python 包 | 读 `compile.sh` 的 `install_dependencies`，它不一定被调用。向用户说明后，按其中的命令安装 |
| 找不到 `hipcc` | 确认 `/opt/dtk/env.sh` 存在，且 DTK 解压完整 |
| 找不到 `libaio` 头文件或库 | 确认第 4 步的 `yum install` 已成功 |
| `install_with_torch.sh` 退出码为 4 | 内部源没有匹配当前 torch 版本的包。把脚本列出的可用 torch tag 告诉用户，由用户决定版本 |
| 无法访问内部 pip 源或 `/ArchivedFile` | 当前环境不在内网或没有挂载，告诉用户。不要改用公共源，否则装到的不是 HCU 版 torch |

## 约束

- 不要提交或推送任何改动，除非用户明确要求。
- 不要修改 workflow 或 `compile.sh` 来绕过问题，除非用户要求。发现它们有问题时告诉用户。
- 不要运行测试，除非用户要求。需要时，全量测试用 `bash .github/scripts/test.sh`（约 6 小时），冒烟测试用 `bash tests/run_test_smoke.sh`。
