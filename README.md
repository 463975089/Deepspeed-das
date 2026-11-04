# DeepSpeed for Hygon

DeepSpeed for Hygon 基于 DeepSpeed 官方版本开发，并包含海光针对海光软硬件环境维护的适配修改。

## 上游来源

- **上游项目：** [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- **上游版本：** [v0.19.7](https://github.com/deepspeedai/DeepSpeed/tree/v0.19.7)
- **上游提交：** [`f5af15c2b4da10411e340850c00ba0cb5b14fd87`](https://github.com/deepspeedai/DeepSpeed/commit/f5af15c2b4da10411e340850c00ba0cb5b14fd87)
- **上游文档：** [DeepSpeed v0.19.7 文档](https://github.com/deepspeedai/DeepSpeed/tree/v0.19.7/docs)
- **上游 README 备份：** [README_ORIGIN.md](README_ORIGIN.md)
- **开源许可证：** Apache License 2.0

本仓库保留 `README_ORIGIN.md`，用于查阅和对比上游内容。该文件沿用上游原文，不作为本项目的版本发布说明。

本仓库在 [LICENSE](LICENSE) 中保留 DeepSpeed 原有的版权声明、许可条款和免责声明。适用的第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 海光适配范围

海光信息技术股份有限公司维护本仓库中面向海光软硬件环境的适配修改，主要包括：

- 海光后端集成；
- 构建与打包调整；
- 算子和内核适配；
- 相关文档与元数据更新。

除非源文件或第三方声明另有说明，源自上游 DeepSpeed 的文件继续遵循相应的上游许可条款。

## 构建与安装
### 预依赖
使用前请确认已准备好基于海光环境包括驱动，dtk等组件
### 构建说明
请按照对应海光 DeepSpeed 发行版本提供的发布说明或打包文档完成构建和安装。平台构建说明见 [docs/build.md](docs/build.md)。如需了解 DeepSpeed 的通用构建方法，请参阅[上游文档](https://github.com/deepspeedai/DeepSpeed/tree/v0.19.7/docs)或 [README_ORIGIN.md](README_ORIGIN.md)。
### 验证
安装完成后，可运行以下命令检查软件包是否能够正常导入，并查看版本号：

```bash
python -c "import deepspeed; print(deepspeed.__version__)"
```

## 参考资料

- [DeepSpeed 官方仓库](https://github.com/deepspeedai/DeepSpeed)
- [DeepSpeed v0.19.7 源代码](https://github.com/deepspeedai/DeepSpeed/tree/v0.19.7)
- [上游 README 备份](README_ORIGIN.md)

## License
本仓库许可证以为 [Apache License 2.0](LICENSE)，保留 DeepSpeed 原有的版权声明、许可条款和免责声明。适用的第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
Hygon 的版权声明仅适用于其新增或修改内容，不替代 ROCm、NVIDIA 或其他上游作者的版权声明。
