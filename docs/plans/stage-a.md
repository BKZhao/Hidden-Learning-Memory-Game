# 阶段 A 实施计划

用途：将协议第十一节阶段 A 落地。Owner：项目维护者。
输入：协议第 2–6、10–12 节；输出：可审查实现、T01–T12 和小规模审计产物。
相关：[模型](../model.md)、[验收](../verification.md)。

## 设计与范围

采用 Python + NumPy 参考实现，标准库 unittest 做协议要求的行为测试。
先建立可核验的顺序实现，再根据实测吞吐率决定生产加速方案。
不先采用编译内核或分布式执行，以便直接检查事件语义；不运行尚未锁定的 E2。
现目录不在 Git 仓库中，没有已有 branch/worktree 可隔离；直接新增项目文件，保留原始文档。
用户已授权按现有协议开始；不重复设置设计审批或生成 commit。

## 任务与验收

- [x] Explore：完整读取 Word 正文及 OMML 数学公式，核实目录、环境和现有规则。
- [x] Design：确定模型公式、模块 owner、离散端点与恢复删失约定。
- [x] Implement 1：先写 `tests/test_model.py`，再实现 `parameters.py`、`hypergraph.py`、`payoff.py`、`state.py`、`learner.py`、`events.py`。验收 T01–T06。
- [x] Implement 2：先写 `tests/test_branches.py`，再实现 `intervention.py`、`shock.py`、`checkpoint.py`、`runner.py`。验收 T07–T10、T12。
- [x] Implement 3：先写 `tests/test_metrics.py`，再实现 `metrics.py` 与未锁定配置门槛。验收 T11、有符号配对损失、非法配置拒绝。
- [x] Verify：24 个测试通过，恢复阈值浮点消减问题已修复并覆盖默认窗口边界。
- [x] Verify：完成小图学习、六分支、冻结与 sham 审计；保存事件日志、轨迹、检查点、配置与哈希。
- [x] Review：对照协议检查公式、更新顺序、分支独立性和输出可追溯性；实测百万次更新及资源用量。
- [x] Cleanup：可复现产物与缓存的保留清单见验收报告，未删除用户文件。

## 后续门槛

阶段 B：E0 的 3 个 r 切片、每点 20 次、500000 MCS；E1 的 36 点×3 次、20000 MCS。
E0 缺少原文 PDF/原始曲线数值时，不宣称完成与原文的数值复现。
E1 仅按 sham 稳定性与冲击响应选点；N=1000 再验证后锁定。
阶段 C 需要独立预实验支持的工作点、样本量、预算和代码版本，不能填入臆测默认值。
阶段 D/E 的自然历史、特异性、泛化和预测保留为后续阶段，不由阶段 A 测试替代。
