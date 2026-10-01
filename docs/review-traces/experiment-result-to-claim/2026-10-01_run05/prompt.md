# E2实现独立审查

Owner：主agent。Reviewer：/root/pilot_review。范围：科学计算主路径及完整性门槛，禁止读取运行中效果均值。

E2实现已落地，请只读审查启动阻碍（不必重跑已有长实验）：src/hidden_memory/training.py新增network_seed和selection_times；scripts/run_confirmatory.py；scripts/analyze_confirmatory.py；docs/plans/confirmatory-v1.1.md；复用run_power_pilot.run_network六分支。重点首个cp、共享图独立学习、H/B/K真实聚类、缺失规则、预算/失败保留、源码归档。主agent在跑36 tests及2history真实小规模smoke，稍后锁定config。指出P0/P1科学正确性bug；仅只读，可运行轻量复算，无修改。

追加：E2固定批次已运行，当前20/40 histories，未读取主效应。你指出的P1已在锁定/启动前修复：analysis校验完整唯一H×B和0..K-1集合、响应和冻结长度/finite，启动器计算budget拒绝越限并拒绝pilot hash，1020新种子对全部实际旧1896种子零重叠。configs/confirmatory.json已锁，36tests通过、2history实际smoke通过。请只读复核修复源码与configs hashes（不要改文件/启动实验），如无阻碍报告；结果审查稍后在全批完成后通知，暂不汇总或查看效果均值。
