# 实验运行与结果入口

Owner：项目维护者。输入：协议 v1.0 与 E0/E1 固定网格；输出：可追溯训练轨迹、检查点、筛选与失败记录。
范围：阶段 B。相关：[实施计划](plans/stage-b.md)、[模型](model.md)、[程序验收](verification.md)。

本轮 E1 已完成，见 [E1 结果](e1-results.md)。
E0 已完成60/60，原文图3(a–c)对照见 [E0结果](e0-results.md)。
E1 完成状态见 [status.json](../artifacts/e1-20261001/status.json)。

## 启动

```bash
uv run python -m unittest discover -s tests -v
uv run python -m hidden_memory campaign E0 --output artifacts/e0-20261001 --workers 4 --seed 2026100100
uv run python -m hidden_memory campaign E1 --output artifacts/e1-20261001 --workers 2 --seed 2026100101
```

两个命令各自运行完整网格。输出目录必须不存在；不覆盖旧结果。
独立任务使用 spawn 进程并行，每条异步序列内部保持顺序；事件带按块生成。
正式启动前通过全套测试，保留本轮测试日志。E0 的原文图形对照与运行成功分开记录。
E1 只进行 sham 筛选，可以与 E0 计算并行；未进入大规模因果干预或 E2。

## 输出 contract

| 文件 | 内容与 owner |
|---|---|
| `manifest.json`、`source.tar.gz` | 固定配置、源码与协议快照、依赖版本和哈希；`experiment_io` |
| `status.json`、`results.jsonl` | 总进度及每条成功/失败记录；`campaign` |
| `jobs/<id>/config.json` | 参数和独立随机流种子；`experiment_specs` |
| `trajectory.npy`、`trajectory_columns.json` | t=0..T 每 MCS 的原始快照和列含义；`training` |
| `checkpoints/mcs-*.npz` | 预设时间完整状态、流状态、来源与校验和；`checkpoint` |
| `diagnostics.json`、`training_summary.json` | 相邻窗口稳定性、Q 差距、行最优集合、访问率与年龄；`training` |
| `controls.json` | 每条训练检查点上的收益交叉核验、冻结和 sham 逐事件负对照；`audit` |
| `pilot/` | 达标 E1 的无冲击和四个强度的 sham 配对轨迹、节点集合、终态与恢复指标；`pilot` |
| `progress.json`、`failure.json` | 当前任务阶段/事件数，或完整异常原因；不静默丢弃 |

训练检查点时刻为 0、100、1000、5000、10000、15000、20000、100000、200000、500000，
取不超过任务时长者并包含最后时刻。阶段 B 不据中间结果提前停止协议规定的训练。
恢复指标输入为 1..T，所有原始 `trajectory` 文件保留 t=0；不把 t=0 纳入主要损失均值。

## 固定筛选规则

E1 先看末5000 MCS平均 q≥0.80，末两段5000 MCS的 q/fP 均值差均≤0.02。
通过者继续无冲击5000 MCS；其均值相对末训练窗口 q/fP 均值差均≤0.02，标记为操作性候选。
随后保留 delta={0.05,0.10,0.20,0.30} 全部 sham 响应，不按响应方向丢弃样本。
该续跑阈值是协议“无明显持续漂移”的预先操作化约定，记录于阶段 B 计划。
程序不自动锁定主工作点；尚需看稳定覆盖率、冲击响应范围并在 N=1000 再验证。

32 个共同事件的冻结/sham 核验是每条任务的工程负对照；控制冲击仅取最多一个贡献节点，
与科学响应的指定 delta 区分。所有 E1 科学冲击使用 floor(delta*N)，不足时记录非法而非截断。

## 进度查看

读取 `status.json` 得到完成/失败计数；长程任务可读取 `jobs/*/progress.json` 检查 MCS 增长。
每个结果都有源码归档、协议哈希、独立种子与网络哈希。当前分支未提交时 commit 为 null，
归档与源码哈希仍可精确标识执行版本。批量任务启动后不得修改其源码。

当前 Git 远端：`https://github.com/BKZhao/Hidden-Learning-Memory-Game.git`。
关联远端与本地运行不代表已经发布文件；commit/push 仍遵守用户明确授权规则。

汇总入口：`uv run python scripts/summarize_campaign.py artifacts/e1-20261001`。
脚本仅读取批量结果，不修改运行中的模型源码；输出 `analysis/points.csv`（含全部点）和
`analysis/summary.json`（状态及输入/脚本哈希）。汇总目录同样拒绝覆盖。
缺失结果保留 planned/completed 区分，不把正在运行的重复当作成功，也不自动选择工作点。

## N=1000 补充筛选与分析

设计见 [规模复验](plans/scale-validation.md)。完整36点×3重复，训练100000MCS，独立root seed；
108/108已完成，0失败，结果见 [规模复验结果](e1-scale-results.md)。
完整运行状态保留于 `artifacts/e1-scale-20261001/status.json`。
独立启动器及配置副本保存于运行目录，补足旧归档器未纳入scripts的范围。

```bash
uv run python scripts/run_scale_validation.py configs/e1-scale-validation.json --output artifacts/e1-scale-20261001
UV_PROJECT_ENVIRONMENT=.venv-analysis uv sync --group analysis --locked
.venv-analysis/bin/python scripts/diagnose_pilot.py artifacts/e1-scale-20261001
```

分析脚本只读取已完成批次，输出 `diagnostics/`，包括候选前50MCS原始响应、合法P行覆盖率、
所有任务的隐藏状态检查点差异和图形。图中的stable candidates仅指公开比例的操作性筛选，
不等于隐藏Q分布已经收敛。分析环境独立于模拟环境，新增绘图依赖不改变模拟数值版本。

## v1.1确认性E2

sham验证与方差预实验已完成，见[工作点结果](shock-validation-results.md)和[样本规划](power-pilot-results.md)。
E2已完成40/40历史、400套配对未来，0失败；见 [结果与基线分解](e2-results.md)。
完成状态保留于 `artifacts/e2-20261001/status.json`。

```bash
.venv/bin/python scripts/run_confirmatory.py configs/confirmatory.json --output artifacts/e2-20261001
.venv/bin/python scripts/analyze_confirmatory.py artifacts/e2-20261001
.venv-analysis/bin/python scripts/plot_confirmatory.py artifacts/e2-20261001
```

输出目录必须全新，分析等待整批完成；固定H20/B2/K10，主delta=.30，详见[固定设计](plans/confirmatory-v1.1.md)。
`hierarchy.json`保存真实network/history映射，`histories/network-XX-history-b/`保存训练和响应。
复用响应器的内部`response/network-NN`中NN是唯一history任务号，不是独立网络计数；
统计分组始终使用顶层network字段。`analysis/networks.csv`才是网络层独立结果表。

## E3机制补充

入口：[机制设计](plans/mechanism-v1.1.md)、[结果](e3-results.md)。运行脚本必须使用新的输出目录。

```bash
.venv/bin/python scripts/audit_gap_support.py --source artifacts/e2-20261001 --reference artifacts/e1-shock-20261001 --output artifacts/e3-gap-audit-new
.venv/bin/python scripts/run_matched_controls.py configs/matched-controls.json --output artifacts/e3-matched-new
.venv-analysis/bin/python scripts/analyze_matched_controls.py artifacts/e3-matched-new
.venv/bin/python scripts/replay_divergence.py --source artifacts/e2-20261001 --output artifacts/e3-divergence-new
```

`matched-controls.json`锁定既有匹配审计、执行器与当前源码哈希；变更输入或实现需新配置与新归档。
`decompose_e2.py`在E2运行目录中创建`decomposition/`且拒绝覆盖；已生成产物直接复核。

## E4/E4b自然历史

设计与结果入口：[E4](e4-results.md)、[E4b](e4b-results.md)。运行任务分为历史/输入支持审计和响应两阶段；
支持不合法也保存完整状态与null效应。两个阶段使用独立配置和输出目录，不覆盖旧结果。

- E4：`scripts/run_natural_history.py`、`configs/natural-history.json`，结果`artifacts/e4-natural-history-20261001`。
- E4b：`scripts/run_paired_cost_history.py`、`configs/paired-cost-history.json`，结果`artifacts/e4b-paired-cost-history-20261001`。
- 两者响应：`scripts/run_natural_response.py`；对应`natural-response.json`或`paired-cost-response.json`。
- 分析：`scripts/analyze_natural_history.py --history <历史目录> --response <响应目录>`，使用`.venv-analysis/bin/python`。
- E4b成本阶段图：`scripts/plot_cost_phase.py <E4b历史目录>`。分析创建新子目录，拒绝覆盖。

源算子owner是`natural_intervention.py`；连续洗脱与支持审计owner是`natural_washout.py`；脚本负责协调、锁定和并行任务。
原E4执行后提取了共享洗脱模块，其旧锁定配置须配合运行目录的`source.tar.gz`与归档启动器复现；
不能直接将旧哈希配置与当前源码混用。E4b记录新的源码哈希。历史运行产物均保留。
