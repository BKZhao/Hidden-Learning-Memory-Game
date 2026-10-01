# 写作与格式参照

Owner：论文作者。输入：用户指定的 Zou–Huang 论文及 Elsevier 官方模板；
输出：本项目统一的论文组织和格式约定。范围：后续英文稿件；不提前生成结果或论文结论。
相关：[实验入口](experiments.md)、[研究协议](../hidden_learning_memory_protocol_v1.docx)。

## 唯一参照论文

Zou, K., & Huang, C. (2026). Cooperation dynamics on hypergraphs with punishment and Q-learning.
Expert Systems with Applications, 296, 128989. https://doi.org/10.1016/j.eswa.2025.128989

用户已指定本项目的 paper 以该文为格式与写作组织参照。
用户现已提供[12页完整正式版](1-s2.0-S0957417425026065-main-2.pdf)。
已检查首页及模型/结果页渲染，并提取全文核对；模型与图3对照见[原文核对](source-paper-check.md)。
第二份[信任博弈预印本](2609.24493v1.pdf)作为相关工作材料，暂不替代已指定的格式基准。

## 可立即执行的格式约定

1. 英文写作，正文采用该文的四个一级部分：`1. Introduction`、`2. Model`、
   `3. Simulation results and analysis`、`4. Conclusion`。
2. 在 `Model` 内分别说明基础博弈、异步 Q-learning、策略不变干预、冲击和配对终点，
   把数值阈值与实现约定写清，不将实现约定归于原作者。
3. 在 `Simulation results and analysis` 下安排复现与筛选、核心配对结果、机制负对照、
   自然历史与泛化；各小节只能使用对应阶段已经产生的证据。
4. 正文PDF前置 Abstract 和 Keywords。出版社网页另列 Highlights，但这份正式PDF未单列
   Highlights 页；如投稿要求单独文件再提供，不把网页结构当成PDF版式。
5. 文内使用作者—年份引用，例如 `Zou and Huang (2026)` 或 `(Zou & Huang, 2026)`，
   不沿用内部协议 `[1]–[4]` 编号作为最终稿引用。
6. 使用可编辑的 LaTeX 公式、统一符号与连续公式/图/表编号。图形保留矢量版本、
   原始数据和脚本；曲线只由真实实验输出生成。
7. 作者贡献使用 CRediT 条目；资助、利益冲突与数据可用性按作者事实填入，不能编造。

LaTeX 类采用 Elsevier 官方 `elsarticle` 包和 author-year 文献样式作为起点。
官方模板入口及源文件打包说明见
[Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)。
官方模板下载到 `artifacts/literature/elsarticle-official.zip`，旁边 JSON 保存来源和 SHA-256。
该 ZIP 是通用投稿模板，不是 Zou–Huang 原文，也不代表已复制其最终出版版式。

## 已核验的版式与仍待核验事项

- 正式版为双栏正文，首页标题跨栏，关键词与摘要并列；主节编号1–4，图注位于图下。
- 原PDF页面约210×280 mm，12页；第2页可见正文主要字体为 CharisSIL，提取字号约7.97 pt。
  字体统计含PDF内其他字体层，采样值用于版面参照，不据此冒充期刊投稿字号要求。
- 图3为跨双栏3×3多面板图，面板(a)–(i)，时间轴对数刻度，策略色彩全图一致。
  本项目图保留清楚的面板标识、可读轴标、图下说明及统一配色，并同时输出矢量与位图。
- 文内作者—年份引用；连续公式编号；正文之后的作者贡献、利益冲突、数据声明按实际事实写。
- `elsarticle`是可编辑稿件的起点；出版版的卷期、DOI及出版社标识不复制到我们的未发表稿件。
- 最新投稿文件要求仍须投稿前核对[官方指南](https://www.sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors)。

先前403下载记录保留于 `artifacts/literature/` 作为获取历史；目前全文已可用，访问不再阻碍E0图形对照。
