# 规模复验独立审查prompt

Owner：主agent。输入：目标规模复验产物；输出：独立审查与后续路由。
Reviewer：/root/pilot_review；只读，不修改结果。

N1000补充筛选已完成，请按同一结果门槛做最终只读复核，范围限 artifacts/e1-scale-20261001/{results.jsonl,analysis/summary.json,analysis/response_diagnostics.json,diagnostics/summary.json}，必要时看原始npz。108/108零失败，88训练+续跑稳定，29点三重复均过；delta .1全部88 recovery50，raw前50内87从t1均过q0-.05、1个仅t1低于；delta .2有3个恢复339/387/586，delta .3有11个延迟但无collapse。独立seed，不含near/far。请判断主冲击工作点是否仍不足、是否应停止按当前主delta盲目扩样，以及可提出何种后续修订而不据near/far选点。输出简洁structured verdict+证据+风险+未验证，不改文件，不启动实验。
