# 独立只读审查

Owner：主agent。输入：E1产物及后续E0对照；输出：reviewer回复、解析及路由。
Reviewer：`/root/pilot_review`。使用技能：`experiment-result-to-claim`。

## 初始prompt原文（由reviewer回传）

背景：/home/bingkun_zhao/projects/hidden-memory 是惩罚Q-learning研究；用户要求继续实验。按 experiment-result-to-claim 技能需要独立结果审查。只读，不写源码/结果，不启动实验，不联系外部人。E0正在运行，不能修改模拟源码。任务：审查现有E1预实验是否支持锁定主工作点/E2，及下一步全部36点×3重复在N=1000、g=5、平均超度8、训练100000MCS、独立seed复验的合理性。证据：docs/e1-results.md、docs/model.md、docs/paper-format.md；artifacts/e1-20261001/results.jsonl、analysis/points.csv和response_diagnostics.json；必要时读取checkpoint和原始npz轨迹。已知108/108完成，88训练稳定、86续跑稳定；delta=.10的88样本恢复时间均最早可判定50MCS；我们额外读取86稳定样本原始1..50轨迹，低于q0-.05最后时刻最多1MCS，可干预P行比例.43-.689。没有测near/far因果效应，不能把sham无大损失说成无隐藏记忆效应。输出：结论、证据/涉及文件、风险等级、建议改动、已运行验证、未验证事项；并含claim_supported yes/partial/no、支持/不支持什么、missing_evidence、next_experiments、confidence。请特别检查任何阻止N1000独立复验的程序或统计问题，以及36全网格复验是否合理。报告简洁，返回给主agent，由主agent保存审查记录。

## 原文到达后的追加prompt

原文PDF现已由用户提供：docs/1-s2.0-S0957417425026065-main-2.pdf。请继续只读审查：核对原文第2–3页模型和第5页图3(a–c)，与E0完整结果（等待完成则标注当前数目）及docs/source-paper-check.md的对照是否准确。特别核对图3(c) r=1.2与正文p6 r=1.5差异、公开q与fC区别。输出结论/证据/风险/已验证/未验证。另请在回复提供你的初始审查prompt原文（若仍可见），便于保存审查trace。不要改文件。
