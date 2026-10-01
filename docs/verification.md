# 阶段 A 验证报告

用途：记录当前参考实现的可检查证据。范围：协议阶段 A，不是 E0–E5 科学结果。
Owner：项目维护者。输入：`tests/`、源码与小规模审计配置；输出：验收状态和下一阶段门槛。
日期：2026-10-01。相关：[模型约定](model.md)、[阶段计划](plans/stage-a.md)。

## 结论

阶段 A 的参考模型与 T01–T12 验收已完成。包含加速后端的 29 个自动化测试通过，
默认小规模六分支审计通过，冻结学习的配对主对比严格为 0。
没有运行原文规模复现、独立参数筛选或确认性实验；尚无论文结论。

## 命令与证据

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
uv run python -m hidden_memory audit --output artifacts/stage-a
uv run python -m hidden_memory benchmark --output artifacts/benchmark-reference --events 1000000
uv run python -m hidden_memory benchmark --output artifacts/benchmark-numba-long --events 100000000 --backend numba
uv run python -m hidden_memory validate-confirmatory configs/confirmatory.json
```

测试、编译、审计与两个后端的基准成功；最后一项按设计以非零状态退出，原因是 `awaiting_pilot_lock`，没有启动实验。
审计和 benchmark 输出目录不可复用；重跑请换新目录名。
没有预先声明的 formatter/linter，因此未擅自引入或声称通过此类检查。

| 协议编号 | 证据 |
|---|---|
| T01–T02 | 全 C/D/P、混合群体及 g=5 合法组成枚举，福利恒等式 |
| T03–T04 | 新行动计数、单条目更新、`s=a` 时旧 Q 目标 |
| T05 | 全零、双并列、单最优三组 30000 次抽样与理论概率比较 |
| T06 | 小图所有节点/候选行动的双收益实现核验 |
| T07 | 所有状态行策略不变，共同越界排除，分支内存独立 |
| T08–T09 | 冻结 Q 与 sham 的逐事件行为/Q 核验；审计覆盖有/无冲击两组 |
| T10 | 同一冲击节点集合，贡献下降 m/N，Q 和访问记录不受冲击改写 |
| T11 | 阈值相等、持续期、完整后向窗口、右删失、负损失保留 |
| T12 | 保存/重载续跑、拒绝覆盖、数组篡改检测、事件带分块一致性 |
| 端到端 | CLI 生成六分支/日志/检查点，检查 UUID 关联与目录拒绝覆盖 |

测试日志：[tests.log](../artifacts/stage-a/tests.log)。
包含加速后端的当前测试日志：[accelerated-tests.log](../artifacts/accelerated-tests.log)。
集成审计：[audit.json](../artifacts/stage-a/audit.json)，
配置：[config.json](../artifacts/stage-a/config.json)，
事件日志：[events.jsonl](../artifacts/stage-a/events.jsonl)。
NPZ 产物使用 `numpy.load(..., allow_pickle=False)` 读取，字段名见 `audit.py`。

## 默认小规模审计

- N=20、g=5、平均超度 8；训练 200 MCS=4000 次更新，未来每分支 100 MCS=2000 次更新。
- 六分支全部保存，额外运行六条冻结分支及逐事件 sham 复制核验。
- 最大投影奖励误差 `6.661338147750939e-16`，低于 `1e-10` 门槛。
- 可干预行比例为 `0.16666666666666666`，已标记低于 20% 的低覆盖率。
- `frozen_theta=0.0`。普通学习分支的数值只作为真实工程样本保留在产物中，
  不进行显著性解释，不用来选工作点；自然支持参照仍为 `null`。
- 本样例记录到全状态策略概率分歧和后续实际行动分歧，证明事件诊断路径可工作；
  这不构成机制结论。训练不足 5000 MCS，未声称获得达标高合作检查点。

## 吞吐与资源：参考后端

在 N=1000、g=5、平均超度 8 下，实际运行 1000000 次更新，耗时 **45.436 秒**，
约每秒 22009 次更新。包括每 MCS 的快照核算，不包括正式批量调度和完整结果 I/O。
检查点写入耗时 0.0179 秒，压缩后 108793 字节；事件带按 10000 条分块，单块 320000 字节。
另用 100000 次更新的资源测量得到最大常驻内存 38912 KiB（约 38 MiB）。

证据：[benchmark.json](../artifacts/benchmark-reference/benchmark.json)、
[资源测量](../artifacts/benchmark-memory.resources.log)。
测量环境为 Python 3.12.3、NumPy 2.5.3；完整源码哈希与软件信息保存在产物中。

仅按慢速参考后端作单进程线性外推（不作为当前生产预算）：

| 工作负载 | 微更新总量 | 参考实现估算 |
|---|---:|---:|
| E1：36 点×3 次×N=300×20000 MCS | 648000000 | 8.18 小时 |
| E2：全部训练达到上限，加六分支响应 | 32000000000 | 16.83 天 |

这是硬件与实现相关的粗略外推；N=300 的真实 E1 吞吐可能不同。
不包含 E0、机制、自然历史、调度、额外 I/O；不是承诺完成时间。

## 加速后端与当前预算

性能剖析显示主要开销是每次事件的数组复制、收益函数分派和记录对象创建。
Numba 后端把相同的异步循环编译执行，保持参数、float64、随机事件、收益求和次序和采样频率。
编译缓存采用非递归的归约实现；两个独立进程的缓存加载测试覆盖了首次实现暴露的崩溃问题。
环境依赖已锁定：Numba 0.68.0、llvmlite 0.50.0，NumPy 仍为 2.5.3。

| 测量 | 实际微更新数 | 计算耗时 |
|---|---:|---:|
| N=1000，Numba | 1000000 | 0.170 秒 |
| N=1000，Numba 长程 | 100000000 | 16.437 秒 |
| N=300，一条 E1 等长训练 | 6000000 | 1.436 秒 |

百万次样例相对参考后端约加速 267 倍。该样例最终策略、Q、群体计数、访问次数和更新时间
与原参考检查点逐位一致；这项有限样例核验不等于对所有未来运行的数学证明。
测试另覆盖三种种子、冻结学习、分块续跑、不同超度求和与非法后端。

证据：[百万次基准](../artifacts/benchmark-numba-v2/benchmark.json)、
[全状态对照](../artifacts/benchmark-numba-v2/reference_comparison.json)、
[一亿次基准](../artifacts/benchmark-numba-long/benchmark.json)、
[N=300 基准](../artifacts/benchmark-numba-n300/benchmark.json)。

据各自规模的实测值，当前单进程纯计算预算为：

- E1 的 108 条主训练：约 **2.6 分钟**，另加网络生成、续跑筛选与结果保存。
- 初始E2预算曾按320亿次估计串行耗时约 **1.46小时**；v1.1加入完整冻结对照后的固定预算见 [确认性设计](plans/confirmatory-v1.1.md)。
- E0三切片×20次×500000MCS曾估计串行约 **1.37小时**；实际4 workers约15分钟完成，见 [E0结果](e0-results.md)。

计时含事件带生成、状态校验和每 MCS 快照，不含 JIT 首次编译、建图和计时结束后的检查点写入。
预热采用独立状态与随机流，不改变计时序列；当前缓存预热约 0.12 秒，不代表冷编译耗时。
不同参数、机器负载和输出策略会影响真实总时长；以上是预算，不是已完成的科学实验结果。

## Review 与保留清单

- 源码按收益、学习、干预、冲击、持久化、指标和执行拆分，无文件超过 200 行。
- 更新序列不并行化；事件输入显式传入，分支深拷贝。
- 原始Word文档保留；本地Git已关联用户指定远端，未创建commit、push或重置用户状态。
- `artifacts/stage-a/` 为审计产物；`artifacts/benchmark-reference/` 与
  `artifacts/benchmark-memory/` 及同名前缀日志为资源测量产物，全部保留。
- 加速基准、测试日志、`artifacts/profile-reference/` 与 `tmp/reference.prof` 保留为性能证据。
  `artifacts/benchmark-numba/` 是缓存崩溃时创建的空目录，不作为成功产物；未静默删除。
- `.venv/`、`__pycache__/`、构建生成的 egg-info 为本地环境/缓存，已配置忽略；
  本轮未删除用户文件或运行产物。

## 后续验证与当前边界

训练流式输出、检查点、失败记录、E0原文核对、E1目标规模验证及v1.1工作点锁定均已完成。
确认性流程新增同图双历史、首个合格检查点选择、完整冻结对照和网络聚类分析。
[36项测试日志](../artifacts/e2-preflight-tests.log)与[真实双历史smoke](../artifacts/e2-smoke/verification.json)均通过。
[最终预检](../artifacts/e2-final-preflight.json)记录配置哈希与重复ID拒绝检查；1020个E2流与全部1896个既有流无交集。
自然历史、匹配C/D扰动、固定时刻敏感性与多工作点泛化仍未由这次工程验收证明。
