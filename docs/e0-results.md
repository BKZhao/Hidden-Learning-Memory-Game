# E0 原文图3(a–c)对照

Owner：项目维护者。输入：Zou–Huang完整PDF和 `artifacts/e0-20261001/`；
输出：20重复统计、轨迹图和有边界的复现判断。相关：[原文核对](source-paper-check.md)。

60/60任务完成，0失败。2026-10-01 16:27:57–16:43:05（Asia/Shanghai），约15分8秒。
每条 N=1000，训练500000 MCS，总计300亿次训练微更新；4个独立worker。

下列是末5000MCS时间均值在20次独立重复间的均值±样本标准差，不是置信区间。

| r | fC | fD | fP | q=fC+fP |
|---|---|---|---|---|
| 1.0 | .54890±.00348 | .43947±.00352 | .01164±.00013 | .56053 |
| 1.1 | .84107±.00259 | .07043±.01051 | .08850±.01108 | .92957 |
| 1.2 | .79080±.00542 | .01552±.00301 | .19368±.00685 | .98448 |

[轨迹图PNG](../artifacts/e0-20261001/figures/e0-timeseries.png) ·
[矢量PDF](../artifacts/e0-20261001/figures/e0-timeseries.pdf) ·
[精确统计CSV](../artifacts/e0-20261001/figures/tail-statistics.csv) ·
[图与输入哈希](../artifacts/e0-20261001/figures/provenance.json)。

曲线取1400个对数间隔整数时点，无时间平滑；阴影为独立重复间SD。
主agent已检查渲染，独立只读审查也复算了三组统计并目视核对原图。
结论是图3(a–c)的定性趋势和终点量级吻合，尚无作者原始数值用于严格误差判定。
不能把这次对照写成复现全部论文，也不能把fC在r=1.1最高写成q在r=1.1最高。

自动汇总中的39个 `training_stable` 是额外套用E1的高贡献率/漂移筛选计数，
不属于E0文献复现通过率；其余任务不是程序失败。E0没有sham续跑，`continuation_stable=0`不适用。
原始 `result.json` 的 `source_figure_comparison=pending` 保留运行时事实，后续对照结论由本文承载。

复现绘图命令（已有输出不可覆盖）：

```bash
UV_PROJECT_ENVIRONMENT=.venv-analysis uv sync --group analysis --locked
.venv-analysis/bin/python scripts/plot_e0.py artifacts/e0-20261001
```
