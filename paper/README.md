# 英文论文研究初稿

Owner：论文作者。输入：用户指定两篇完整 PDF、E0–E4c 已有产物。
输出：[论文 PDF](../artifacts/paper-draft/main.pdf)、[矢量图与300 dpi PNG](../artifacts/paper-draft/figures/)。
范围：可阅读、可修改的研究初稿；作者信息、完整查新、公共数据归档与投稿声明尚未完成。

正文唯一编辑源为 `draft/*.md`；公式及浮动图表保留可编辑 LaTeX。
`main.tex` 只负责组装、格式和参考文献，`figures/design.tex` 为矢量设计示意。
构建将 Markdown 转到 `artifacts/paper-draft/sections/`，不要直接编辑生成的分节。

从仓库根目录运行：

```bash
.venv-analysis/bin/python scripts/build_paper.py --figures
```

依赖项目 analysis 环境的 NumPy/Matplotlib，以及系统 Pandoc、TeX Live、latexmk、BibTeX。
默认使用 `/usr/bin` 中可用的完整 TeX；当前 Conda 的 pdflatex 缺少格式文件。
Elsevier 类从已归档官方 ZIP 提取到 `tmp/elsarticle/`，不做全局安装。
`--figures` 只重绘既有数据，不启动模拟；省略此参数可以仅重编稿件。

六张正文图（含设计示意和189点密集参数相图）及一张附录图的 owner、数据路径和主张边界见
[写作设计](../docs/plans/paper-draft.md)。独立 PNG/PDF 共七组，包含六张数据/设计图和一张附录图。
每张数据图旁的 JSON 保存输入与脚本 SHA-256；主构建另保存正文和 PDF 哈希。

格式以 Zou–Huang 的四主节、双栏和作者—年份引用为基准；Guo 等用于共同起点和配对结果的叙事参照。
书目核验：Zou–Huang 与 Guo 等直接核对用户提供的 PDF 首页；Zou–Huang 与 Watkins–Dayan
另下载 Crossref BibTeX，保存在 `artifacts/paper-draft/bibliography/`；Watkins–Dayan 核对
[Springer 原始出版记录](https://doi.org/10.1007/BF00992698)。Guo 条目按本地提供版本记录，不虚构出版 DOI。
当前三条核心引用均在文中使用；完整相关工作拓展和新颖性检索仍需补充。

密集相图使用E5的189个实际参数点，不做插值；它用于描述宏观相态和临界脆弱性，不提供near/far隐藏Q效应的跨点泛化。
当前不能写成的结论：一般韧性改善、P行动特异性、稳定自然历史效应、多工作点泛化或正式等效。
投稿前的最小补证据清单见[投稿准备](../docs/plans/paper-readiness.md)。
本轮数值、图形与最终PDF的[独立审阅记录](../docs/review-traces/paper-draft/2026-10-01.md)已归档。
