# Hidden Learning Memory

公共物品博弈中隐藏学习记忆与合作韧性的可审计参考实现及 Numba 加速后端。
研究设计见 `hidden_learning_memory_protocol_v1.docx`；当前已完成E0/E1、v1.1工作点验证与E2确认性主实验。

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m hidden_memory audit --output artifacts/my-audit
uv run python -m hidden_memory benchmark --output artifacts/my-benchmark --events 1000000
uv run python -m hidden_memory benchmark --output artifacts/my-fast-benchmark --events 1000000 --backend numba
```

输出目录必须尚不存在，既有检查点与审计结果不会被覆盖。`audit` 默认使用
N=20、训练 200 MCS、响应 100 MCS，仅用于工程核验，不能作为 E0 复现或 E1 选点证据。

产物包含检查点、完整四槽事件带、冲击节点、六分支轨迹、冻结学习对照、事件日志和哈希。
`benchmark` 在 N=1000 上测量参考实现吞吐率，按实际微更新数换算预算。

`--backend numba` 编译执行异步事件循环，保留 float64、更新次序与每 MCS 快照。
库接口 `run_trajectory(..., backend='numba')` 和 `run_paired(..., backend='numba')`
支持加速运行；逐事件日志使用默认 `reference` 后端。
实测 N=1000 下 1 亿次更新约 16.44 秒，详细预算与验证见验收报告。

[模型与实现约定](docs/model.md) · [验收证据](docs/verification.md) · [阶段计划](docs/plans/stage-a.md)

[E0/E1 运行与进度](docs/experiments.md) · [论文格式参照](docs/paper-format.md)
GitHub 远端：<https://github.com/BKZhao/Hidden-Learning-Memory-Game>。

确认性配置在 `configs/confirmatory.json`，已按独立sham验证和方差预实验锁定。
主冲击由v1.0的delta=.10明示修订为v1.1的.30，原协议及旧结果保留。
E2入口：`python scripts/run_confirmatory.py configs/confirmatory.json --output artifacts/my-e2`。
设计见 [E2固定设计](docs/plans/confirmatory-v1.1.md)，结果及基线解释见 [E2结果](docs/e2-results.md)。
