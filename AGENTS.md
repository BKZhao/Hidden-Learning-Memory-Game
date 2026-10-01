# hidden-memory

本项目实现 `hidden_learning_memory_protocol_v1.docx` 的公共物品博弈协议。
行动编码固定为 C=0、D=1、P=2，主计算使用 float64；1 MCS=N 次有放回抽样。

- 源码：`src/hidden_memory/`；测试：`tests/`；配置：`configs/`。
- 文档入口：[docs/README.md](docs/README.md)。原始 Word 是研究协议的权威来源。
- 运行产物：`artifacts/`；临时文件：`tmp/`；不得覆盖既有检查点。
- 测试：`uv run python -m unittest discover -s tests -v`。
- 阶段 A：`uv run python -m hidden_memory audit --output artifacts/stage-a`。
- 阶段B：`campaign E0/E1`；E2：`scripts/run_confirmatory.py`，完整命令见 [docs/experiments.md](docs/experiments.md)。
- 明示协议修订、网络/历史层级与确认性缺失规则见 [docs/plans/confirmatory-v1.1.md](docs/plans/confirmatory-v1.1.md)。
- 已启动的批量实验使用固定源码，运行期间不修改其源码；源归档与种子存于运行目录。
- 未通过不变性验收不得启动科学实验；未锁定参数不得启动确认性实验。
- 共享事件带的每个事件始终占四个随机槽位；冲击使用独立随机流。
- Q 冻结仅禁用 Q 更新，不能冻结公开策略；sham 必须逐位保留 Q。
- 原始文档与运行产物不得静默删除。没有明确授权不 commit、push 或 merge。
