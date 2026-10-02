# 论文初稿与图表设计

Owner：论文作者；图表代码 owner：`scripts/paper_figures/`。输入：两篇指定 PDF 和 E0–E4c 已完成数据。
输出：`paper/` 可编辑英文稿、`artifacts/paper-draft/` PDF/图/溯源文件。
范围：真实现有结果的研究初稿，不启动新实验，不宣称投稿就绪。
相关：[格式约定](../paper-format.md)、[待补证据](paper-readiness.md)、[结论](../findings.md)。

## 结构与边界

沿用 Zou–Huang 四主节：Introduction、Model、Simulation results and analysis、Conclusion。
借鉴 Guo 等共同学习状态分叉、绝对路径与配对差并列、直接机制对比的写法；不借用其环境反馈结论。
文内作者—年份引用，Elsevier `elsarticle` 双栏草稿；不复制出版社卷期、DOI或署名。
Markdown 为正文单一编辑源，构建时转换到运行目录内的 LaTeX 分节，避免维护两份正文。

## 主张—证据对应

| 主张 | 数据 | 正文角色与限制 |
|---|---|---|
| 初始全状态 policy 相同不保证继续学习的路径相同 | E2 六臂、全 Q 冻结 | 主结果；仅锁定工作点，效应低于预设2个百分点 |
| 无冲击瞬态贡献较大 | E2 全量分解 | 同时展示受冲击绝对合作率，避免倒置解释 |
| C 值下降可启动局部排序分化 | E3 预定8个回放前缀 | 局部机制示例，不推断总体中介比例 |
| P 特异性尚未建立 | E3 匹配 P/D 及直接差 | C 支持不足不填零；探索区间跨零 |
| 自然值支持与非零自然效应是不同问题 | E4/E4b失败与E4c修订验证 | 同时保留支持失败和效应不确定，不声称等效 |

## 图表 owner 与验证

| 图 | Owner | 输入 |
|---|---|---|
| 1 六臂设计示意 | `paper/figures/design.tex` | 干预公式与协议，不伪造数值 |
| 2 参考模型三条件复现 | `scripts/paper_figures/reproduction.py` | E0 已归档曲线及运行清单 |
| 3 参数相图 | `scripts/paper_figures/phase_map.py` | E5密集189点网格；q、fP及有稳定支持的sham冲击损失 |
| 4 绝对路径、基线差与网络推断 | `scripts/paper_figures/main_result.py` | E2 网络聚合及主要统计，断言主效应重构一致 |
| 5 Q交叉、8前缀及匹配对照 | `scripts/paper_figures/mechanism.py` | E3 真实事件前后值及匹配统计 |
| 6 长成本阶段与短脉冲验证 | `scripts/paper_figures/natural_history.py` | E4b成本路径、E4c输入门槛与响应 |
| A1 惩罚者比例与福利 | `scripts/paper_figures/secondary.py` | E2 网络聚合 |

图形保持统一处理配色，时间对数轴不平滑；区间方法按实际网络bootstrap标注。
E0阴影为独立运行SD，不能误写为CI。每图保存PDF、300 dpi PNG及输入/脚本SHA-256。
验证包括数据断言、编译日志、所有页面与图的渲染审阅及独立只读审查。
E5密集相图只补充宏观相态背景，不等同于第二工作点六臂验证。第二工作点、扩展分歧统计与完整查新仍是后续工作，不在本稿写成已完成实验。
