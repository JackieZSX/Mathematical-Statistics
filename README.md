# 深度神经网络置信度校准的极大似然推断与假设检验

本项目为研究生《数理统计》课程学术论文及配套实证复现工程，针对现代深度卷积神经网络普遍存在的预测概率「过度自信」现象，系统构建了基于 Temperature Scaling 的极大似然估计（MLE）、Score 方程、Fisher 信息矩阵、严格凹性数学证明、一阶 Delta 方法、非参数 Bootstrap 以及三大经典假设检验（Wald / Score / LRT）统计推断体系。

## 项目结构

```
.
├── figures_calib/             # 实验生成的 5 组高清矢量与位图图表 (fig1-fig5, PDF/PNG)
├── paper/                     # 完整模块化 LaTeX 论文源码与编译成品
│   ├── figures/               # 封面校徽 (hfut_logo.png)
│   ├── main.tex               # LaTeX 主导言区与排版配置 (ctexart)
│   ├── cover.tex              # 仿规范封面页 (含校徽、三行题名、四栏下划线、日期)
│   ├── sec-abstract.tex       # 摘要五要素与关键词
│   ├── sec-intro.tex          # 第 1 节 引言 (背景、双脉络综述、4项贡献、结构)
│   ├── sec-theory-a.tex       # 第 2 节 理论与模型 (设定、MLE、Score/Fisher推导、凹性证明、牛顿法)
│   ├── sec-theory-b.tex       # 第 2 节 渐近理论与检验 (Delta方法、Bootstrap、Wald/Score/LRT、配对检验)
│   ├── sec-empirical.tex      # 第 3 节 实例分析 (无泄漏设计、表1-3、图1-5解读)
│   ├── sec-discussion.tex     # 第 4 节 结果与讨论 (深度效应、曲率排序、FPC效应、保序权衡、建议)
│   ├── sec-refs.tex           # 参考文献 (GB/T 7714 格式著录 12 篇核心经典文献)
│   ├── sec-appendix.tex       # 附录 (数据样例、核心极简算法函数 ~28 行、分箱与收敛数值表)
│   └── main.pdf               # XeLaTeX 编译生成的 34 页学术级课程论文 PDF
├── run_calibration_experiment.py # 完整统计校准与假设检验流水线脚本
├── plot_calibration_figures.py   # 高学术水准图表绘制脚本 (Figure 1 - 5)
├── calibration_summary.json   # 完整实验数据与检验统计量结构化结果
├── calibration_summary.csv    # 评估集与校准集核心指标汇总表
├── pyproject.toml             # Python 项目与依赖配置 (uv 管理)
├── uv.lock                    # 依赖锁定文件
├── AGENT.md                   # LaTeX 排版与学术规范指南
├── 数理统计课程论文_学术级论文各节标题与内容写作指南.md # 各节标题与内容写作指引
└── README.md                  # 项目复现与说明文档
```

## 核心理论与实证结果

1. **极大似然凹性与唯一性保证**：
   证明了观测 Fisher 信息量恒等于模型条件方差和：$I_n(\beta) = \sum_{i=1}^n \operatorname{Var}_{p_i(\beta)}(Z_i) > 0$，确立了对数似然函数的全局严格凹性，保证了极大似然估计的唯一存在性与阻尼牛顿法的局部二次收敛。
2. **渐近推断与三大假设检验**：
   在 CIFAR-10 官方测试集（$N=10{,}000$，按 $2000:8000$ 严格分层划分为校准集与独立评估集，种子固定为 20261111）上：
   - **ResNet-20**：$\hat T = 1.6245$ (95% Delta CI $[1.5147, 1.7343]$), Wald $= 328.09$, Score $= 381.69$, LRT $= 170.84$ ($p < 10^{-16}$)；
   - **ResNet-56**：$\hat T = 1.6904$ (95% Delta CI $[1.5884, 1.7925]$), Wald $= 502.47$, Score $= 727.95$, LRT $= 212.94$ ($p < 10^{-16}$)；
   - 三大检验均以压倒性置信度拒绝无需校准的原假设 $H_0: T=1$。
3. **独立评估集表现**：
   - ResNet-20：ECE 下降 $66.1\%$（$3.88\% \to 1.31\%$），NLL 下降 $17.7\%$，Brier 分数下降 $5.8\%$（Wilcoxon $p < 10^{-16}$）；
   - ResNet-56：ECE 下降 $68.6\%$（$3.80\% \to 1.19\%$），NLL 下降 $24.7\%$，Brier 分数下降 $8.0\%$（Wilcoxon $p < 10^{-16}$）；
   - McNemar 决策检验统计量为 $0.0$（$p = 1.000$），证实单参数缩放严格保证分类精度无损失。
4. **相合速率验证**：
   $\log(\mathrm{SE})$ 对 $\log(n_{\mathrm{cal}})$ 双对数回归斜率为 $-0.501$（ResNet-20）与 $-0.506$（ResNet-56），严格验证了中心极限定理预测的 $O(n^{-1/2})$ 渐近相合速率。

## 运行与编译指南

### 1. 运行校准实验与绘图
```bash
python run_calibration_experiment.py
python plot_calibration_figures.py
```

### 2. 编译 LaTeX 论文
在 `paper` 目录下使用 XeLaTeX 执行编译：
```bash
cd paper
xelatex -interaction=nonstopmode -enable-installer main.tex
xelatex -interaction=nonstopmode -enable-installer main.tex
```
生成的 `main.pdf` 即为定稿论文。
