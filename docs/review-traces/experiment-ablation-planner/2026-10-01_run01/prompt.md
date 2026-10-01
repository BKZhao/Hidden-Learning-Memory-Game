# 独立E3机制设计与执行审查

Owner：项目维护者。范围：E2之后机制设计、E3匹配与局部回放。
Reviewer：/root/pilot_review，仅只读检查，不修改实现，不启动大实验。
输入：E2全量结果、固定方案、源码、匹配审计和原始E3轨迹；输出：设计建议及可支持结论。

设计请求：独立提出最有辨识力的基线分解、C/D对照、局部学习与自然历史设计，不根据预期正向结果筛选。
执行审查请求：核对匹配算法与方案，复算20网络原始P/D theta、bootstrap及冻结，检查前缀日志的识别范围。
最终回放请求：审查scripts/replay_divergence.py与v2输出；独立核验首次分歧、参考/Numba全状态以及与E2轨迹逐位一致性。

原初设计另存design-review.md；匹配审查与最终回放审查分别存review-matched.md和review-replay.md。
执行者采用的固定设计见../../../plans/mechanism-v1.1.md，结果见../../../e3-results.md。
