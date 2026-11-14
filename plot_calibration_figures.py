#!/usr/bin/env python3
"""
plot_calibration_figures.py
绘制 5 张高信息密度、线条与坐标轴丰富的学术统计图表 (PNG + PDF):
  Fig 1: 原始模型「过度自信」探索性分析 (2x2: KDE+CDF、正确/错误小提琴+箱线、逐类哑铃折线、ECE 对分箱数 M 多曲线)
  Fig 2: 极大似然估计 (MLE) 剖面曲线、Score 函数切线几何与三大检验统计量对比 (1x3)
  Fig 3: 校准前后可靠性图 (Reliability Diagram)、LOWESS 平滑校准曲线与分位点损失改善轨迹 (2x2)
  Fig 4: 渐近正态性与 B=2000 Bootstrap 验证 (直方图+KDE+理论正态密度叠加、Q-Q 图含 95% 包络带) (1x3)
  Fig 5: 校准样本量 n_cal 收敛轨迹、log-log -1/2 斜率验证与评估集多维指标下降对比 (1x3)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, PercentFormatter
from scipy import stats

# 字体与学术排版配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["figure.dpi"] = 150

OUT_DIR = "figures_calib"
os.makedirs(OUT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

C_R20 = "#1f77b4"     # 经典蓝 (ResNet-20)
C_R56 = "#d62728"     # 砖红 (ResNet-56)
C_TS = "#2ca02c"      # 翠绿 (Temperature Scaling)
C_PLATT = "#9467bd"   # 紫罗兰 (Platt Scaling)
C_UNCAL = "#e6550d"   # 橙红 (Uncalibrated)


def style_axes(ax, x_minor=2, y_minor=2, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    if grid:
        ax.grid(True, which="major", alpha=0.30, linestyle="--", linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":", linewidth=0.5)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存 {name}.png / .pdf")


def lowess_binned(x, y, grid_x, bandwidth=0.06):
    """高斯核局部加权回归 + 逐点二项标准误带"""
    mu = np.empty_like(grid_x)
    se = np.empty_like(grid_x)
    for idx, gx in enumerate(grid_x):
        w = np.exp(-0.5 * ((x - gx) / bandwidth) ** 2)
        w_sum = np.sum(w)
        if w_sum > 1e-8:
            w_norm = w / w_sum
            p_loc = float(np.sum(w_norm * y))
            n_eff = 1.0 / np.sum(w_norm ** 2)
            mu[idx] = p_loc
            se[idx] = np.sqrt(max(p_loc * (1.0 - p_loc), 1e-6) / max(n_eff, 1.0))
        else:
            mu[idx], se[idx] = np.nan, np.nan
    return mu, se


def plot_fig1(res):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    r20 = res["resnet20"]
    r56 = res["resnet56"]

    # (a) 置信度 p_max KDE + ECDF 双轴曲线
    ax = axes[0, 0]
    style_axes(ax)
    x_grid = np.linspace(0.25, 1.0, 300)
    for r, col, lbl in [(r20, C_R20, "ResNet-20"), (r56, C_R56, "ResNet-56")]:
        c = np.array(r["eval_raw_arrays"]["conf_uncal"])
        kde = stats.gaussian_kde(c, bw_method=0.08)
        ax.plot(x_grid, kde(x_grid), color=col, lw=2.2, label=f"{lbl} KDE (均值={c.mean():.3f})")
        ax.fill_between(x_grid, 0, kde(x_grid), color=col, alpha=0.12)
    ax.set_xlim(0.25, 1.01)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("未校准最大预测置信度 $p_{\\max}$", fontsize=10.5)
    ax.set_ylabel("核密度估计 (KDE)", fontsize=10.5)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    for r, col, lbl in [(r20, C_R20, "ResNet-20"), (r56, C_R56, "ResNet-56")]:
        c = np.sort(np.array(r["eval_raw_arrays"]["conf_uncal"]))
        ecdf = np.arange(1, len(c) + 1) / len(c)
        step = max(1, len(c) // 250)
        ax2.plot(c[::step], ecdf[::step], color=col, ls="--", lw=1.5, alpha=0.85, label=f"{lbl} ECDF")
    ax2.set_ylabel("经验累积分布 (ECDF)", fontsize=10.5)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.3, frameon=True)
    ax.set_title("(a) 预测置信度分布密度 (KDE) 与累积分布 (ECDF)", fontsize=11.5, fontweight="bold")

    # (b) 正确 vs 错误样本置信度小提琴图 + 箱线图 + 均值连线
    ax = axes[0, 1]
    style_axes(ax, x_minor=0)
    groups, positions, colors_g = [], [1, 2, 3.4, 4.4], [C_R20, C_UNCAL, C_R56, C_R56]
    labels_g = ["R20\n预测正确", "R20\n预测错误", "R56\n预测正确", "R56\n预测错误"]
    for r in (r20, r56):
        c = np.array(r["eval_raw_arrays"]["conf_uncal"])
        corr = np.array(r["eval_raw_arrays"]["correct"]) == 1
        groups.append(c[corr])
        groups.append(c[~corr])
    vp = ax.violinplot(groups, positions=positions, widths=0.65, showmeans=False, showextrema=False)
    for body, col in zip(vp["bodies"], colors_g):
        body.set_facecolor(col)
        body.set_alpha(0.28)
        body.set_edgecolor(col)
    bp = ax.boxplot(groups, positions=positions, widths=0.22, patch_artist=True,
                    boxprops=dict(linewidth=1.1), medianprops=dict(color="black", lw=1.6),
                    flierprops=dict(marker=".", markersize=2.5, alpha=0.3))
    for patch, col in zip(bp["boxes"], colors_g):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    means = [np.mean(g) for g in groups]
    ax.plot(positions[:2], means[:2], "o-", color=C_R20, lw=1.8, ms=6)
    ax.plot(positions[2:], means[2:], "s-", color=C_R56, lw=1.8, ms=6)
    for p, m in zip(positions, means):
        ax.text(p, m + 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88), zorder=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_g, fontsize=9.2)
    ax.set_ylim(0.15, 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("未校准置信度 $p_{\\max}$", fontsize=10.5)
    ax.set_title("(b) 预测正确 vs 错误样本的置信度分布对比", fontsize=11.5, fontweight="bold")

    # (c) 逐类别 (10 类) 平均置信度 vs 实际准确率 哑铃折线图
    ax = axes[1, 0]
    style_axes(ax, x_minor=0)
    x_cls = np.arange(10)
    for r, col, marker, offset, lbl in [
        (r20, C_R20, "o", -0.14, "ResNet-20"),
        (r56, C_R56, "s", 0.14, "ResNet-56"),
    ]:
        accs = np.array([cs["acc"] for cs in r["class_stats"]])
        confs = np.array([cs["conf_uncal"] for cs in r["class_stats"]])
        confs_ts = np.array([cs["conf_ts"] for cs in r["class_stats"]])
        ax.plot(x_cls + offset, confs, marker=marker, ls="-", color=col, lw=1.6, ms=5.5,
                label=f"{lbl} 未校准置信度")
        ax.plot(x_cls + offset, accs, marker=marker, ls="--", color=col, lw=1.6, ms=5.5, alpha=0.85,
                mfc="white", label=f"{lbl} 真实准确率")
        ax.plot(x_cls + offset, confs_ts, ls=":", color=C_TS if r is r20 else "#17becf", lw=1.4,
                label=f"{lbl} TS 校准置信度")
        ax.fill_between(x_cls + offset, accs, confs, color=col, alpha=0.10)
    ax.set_xticks(x_cls)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0.80, 1.005)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("概率 / 准确率", fontsize=10.5)
    ax.legend(loc="lower left", ncol=2, fontsize=7.6, frameon=True)
    ax.set_title("(c) CIFAR-10 逐类别置信度与真实准确率差距 (Gap)", fontsize=11.5, fontweight="bold")

    # (d) ECE 随分箱数 M 变化多曲线敏感性
    ax = axes[1, 1]
    style_axes(ax)
    M_vals = np.array(r20["ece_by_M"]["M"])
    ax.plot(M_vals, np.array(r20["ece_by_M"]["uncal"]) * 100, "o-", color=C_R20, lw=2.0, ms=5.5, label="ResNet-20 未校准")
    ax.plot(M_vals, np.array(r56["ece_by_M"]["uncal"]) * 100, "s-", color=C_R56, lw=2.0, ms=5.5, label="ResNet-56 未校准")
    ax.plot(M_vals, np.array(r20["ece_by_M"]["ts"]) * 100, "o--", color=C_TS, lw=1.8, ms=5.0, label="ResNet-20 TempScaling")
    ax.plot(M_vals, np.array(r56["ece_by_M"]["ts"]) * 100, "s--", color="#17becf", lw=1.8, ms=5.0, label="ResNet-56 TempScaling")
    ax.plot(M_vals, np.array(r20["ece_by_M"]["platt"]) * 100, "^:", color=C_PLATT, lw=1.6, ms=5.0, label="ResNet-20 PlattScaling")
    ax.plot(M_vals, np.array(r56["ece_by_M"]["platt"]) * 100, "d:", color="#8c564b", lw=1.6, ms=5.0, label="ResNet-56 PlattScaling")
    ax.axvline(15, color="gray", ls="-.", lw=1.1, alpha=0.7)
    ax.text(15.8, ax.get_ylim()[1] * 0.88, "默认 M=15", fontsize=8.5, color="#444444")
    ax.set_xticks(M_vals)
    ax.set_xlabel("分箱数 $M$", fontsize=10.5)
    ax.set_ylabel("期望校准误差 ECE (%)", fontsize=10.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(loc="center right", fontsize=8.0, frameon=True)
    ax.set_title("(d) 校准误差 ECE 对分箱数 $M$ 的敏感性曲线", fontsize=11.5, fontweight="bold")

    save_fig(fig, "fig1_eda_overconfidence")


def plot_fig2(res):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    plt.subplots_adjust(wspace=0.28)

    r20 = res["resnet20"]
    r56 = res["resnet56"]

    # (a) 归一化对数似然剖面 l(T) - l(T_hat)
    ax = axes[0]
    style_axes(ax)
    T_grid = np.array(r20["profiles"]["T_grid"])
    crit = -0.5 * stats.chi2.ppf(0.95, df=1)  # -1.9207
    for r, col, lbl in [(r20, C_R20, "ResNet-20"), (r56, C_R56, "ResNet-56")]:
        ll = np.array(r["profiles"]["ll_grid"])
        ll_hat = r["ts_mle"]["ll_hat"]
        T_hat = r["ts_mle"]["T_hat"]
        ci_l, ci_h = r["ts_mle"]["ci_T_inv_95"]
        diff_ll = ll - ll_hat
        ax.plot(T_grid, diff_ll, color=col, lw=2.2, label=f"{lbl} ($\\hat T={T_hat:.3f}$)")
        ax.axvline(T_hat, color=col, ls=":", lw=1.3, alpha=0.8)
        ax.plot([ci_l, ci_h], [crit, crit], "o-", color=col, ms=4.5, lw=2.5, alpha=0.85)
    ax.axhline(crit, color="black", ls="--", lw=1.3, label=f"95% LRT 临界线 ({crit:.2f})")
    ax.axvline(1.0, color="#555555", ls="-.", lw=1.3, label="$H_0: T=1$ (未校准)")
    ax.set_xlim(0.75, 2.05)
    ax.set_ylim(-52, 4)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("温度参数 $T$", fontsize=10.5)
    ax.set_ylabel("相对对数似然 $\\ell(T) - \\ell(\\hat T)$", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True)
    ax.set_title("(a) 相对对数似然剖面与 LRT 95% 区间", fontsize=11.2, fontweight="bold")

    # (b) Score 函数 U(beta) 曲线与切线斜率 (-I)
    ax = axes[1]
    style_axes(ax)
    b_grid = np.array(r20["profiles"]["beta_grid"])
    for r, col, lbl in [(r20, C_R20, "ResNet-20"), (r56, C_R56, "ResNet-56")]:
        U = np.array(r["profiles"]["U_grid"])
        b_hat = r["ts_mle"]["beta_hat"]
        I_hat = r["ts_mle"]["I_hat"]
        U_null = r["ts_mle"]["U_null"]
        ax.plot(b_grid, U, color=col, lw=2.2, label=f"{lbl} $U(\\beta)$ ($\\hat\\beta={b_hat:.3f}$)")
        # 切线
        b_tan = np.linspace(b_hat - 0.16, b_hat + 0.16, 40)
        ax.plot(b_tan, -I_hat * (b_tan - b_hat), ls="--", color=col, lw=1.3, alpha=0.8)
        ax.scatter([b_hat], [0], color=col, s=45, zorder=5)
        ax.scatter([1.0], [U_null], color=col, marker="^", s=55, zorder=5)
    ax.axhline(0, color="black", lw=1.0, ls="-")
    ax.axvline(1.0, color="#555555", ls="-.", lw=1.2, label="$H_0: \\beta=1$")
    ax.set_xlim(0.42, 1.35)
    ax.xaxis.set_major_locator(MultipleLocator(0.15))
    ax.set_xlabel("逆温度参数 $\\beta = 1/T$", fontsize=10.5)
    ax.set_ylabel("Score 函数 $U(\\beta)=\\ell'(\\beta)$", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)
    ax.set_title("(b) Score 方程与 Fisher 信息几何 ($I=-U'$)", fontsize=11.2, fontweight="bold")

    # (c) 三大检验统计量对比 + chi2_1 临界阈值
    ax = axes[2]
    style_axes(ax, x_minor=0)
    names = ["Wald 检验", "Score 检验", "似然比 (LRT)"]
    x = np.arange(3)
    w = 0.34
    s20 = [r20["ts_mle"]["tests"][k]["stat"] for k in ("Wald", "Score", "LRT")]
    s56 = [r56["ts_mle"]["tests"][k]["stat"] for k in ("Wald", "Score", "LRT")]
    b1 = ax.bar(x - w/2, s20, width=w, color=C_R20, alpha=0.88, label="ResNet-20 (浅层)")
    b2 = ax.bar(x + w/2, s56, width=w, color=C_R56, alpha=0.88, label="ResNet-56 (深层)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axhline(3.8415, color="black", ls="--", lw=1.4, label="$\\chi^2_{1,0.95}=3.84$ 拒绝阈值")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0, max(s56) * 1.22)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_ylabel("检验统计量值 ($\\sim\\chi^2_1$)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True)
    ax.set_title("(c) $H_0: T=1$ 三大检验统计量对比 ($p<10^{-8}$)", fontsize=11.2, fontweight="bold")

    save_fig(fig, "fig2_mle_profile_and_tests")


def plot_fig3(res):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    for idx, (mkey, title_m) in enumerate([("resnet20", "ResNet-20"), ("resnet56", "ResNet-56")]):
        ax = axes[0, idx]
        style_axes(ax)
        r = res[mkey]
        ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="理想校准线 ($y=x$)")
        for method_k, col, marker, lbl in [
            ("uncalibrated", C_UNCAL, "o", f"未校准 (ECE={r['eval_metrics']['uncalibrated']['ece']*100:.2f}%)"),
            ("temp_scaling", C_TS, "s", f"TempScaling (ECE={r['eval_metrics']['temp_scaling']['ece']*100:.2f}%)"),
            ("platt_scaling", C_PLATT, "^", f"PlattScaling (ECE={r['eval_metrics']['platt_scaling']['ece']*100:.2f}%)"),
        ]:
            bins = r["eval_metrics"][method_k]["bins"]
            mids = np.array([b["conf"] for b in bins if b["count"] >= 5])
            accs = np.array([b["acc"] for b in bins if b["count"] >= 5])
            lo = np.array([b["acc"] - b["cp_low"] for b in bins if b["count"] >= 5])
            hi = np.array([b["cp_high"] - b["acc"] for b in bins if b["count"] >= 5])
            ax.errorbar(mids, accs, yerr=[lo, hi], fmt=f"{marker}-", color=col, lw=1.8, ms=5.0,
                        capsize=3, alpha=0.9, label=lbl)
            if method_k == "uncalibrated":
                ax.fill_between(mids, accs, mids, color=col, alpha=0.12)
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(0.25, 1.04)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel("分箱平均预测置信度", fontsize=10.5)
        ax.set_ylabel("分箱实际准确率 (含 95% CP 误差棒)", fontsize=10.5)
        ax.legend(loc="upper left", fontsize=8.3, frameon=True)
        ax.set_title(f"({'ab'[idx]}) {title_m} 校准前后可靠性图 (Reliability Diagram)", fontsize=11.5, fontweight="bold")

    # (c) 连续高斯核平滑校准曲线 + 95% 置信带 (ResNet-56 展示最强对比 + ResNet-20)
    ax = axes[1, 0]
    style_axes(ax)
    ax.plot([0.35, 1.0], [0.35, 1.0], "k--", lw=1.4, label="理想校准线 ($y=x$)")
    gx = np.linspace(0.38, 0.99, 120)
    for mkey, col_u, col_t, lbl in [
        ("resnet20", C_R20, "#17becf", "ResNet-20"),
        ("resnet56", C_R56, C_TS, "ResNet-56"),
    ]:
        arr = res[mkey]["eval_raw_arrays"]
        corr = np.array(arr["correct"], dtype=float)
        cu = np.array(arr["conf_uncal"])
        ct = np.array(arr["conf_ts"])
        mu_u, se_u = lowess_binned(cu, corr, gx, bandwidth=0.05)
        mu_t, se_t = lowess_binned(ct, corr, gx, bandwidth=0.05)
        ax.plot(gx, mu_u, color=col_u, ls="-", lw=2.0, label=f"{lbl} 未校准平滑曲线")
        ax.fill_between(gx, mu_u - 1.96 * se_u, mu_u + 1.96 * se_u, color=col_u, alpha=0.12)
        ax.plot(gx, mu_t, color=col_t, ls="-.", lw=2.0, label=f"{lbl} TS 校准平滑曲线")
        ax.fill_between(gx, mu_t - 1.96 * se_t, mu_t + 1.96 * se_t, color=col_t, alpha=0.12)
    ax.set_xlim(0.35, 1.01)
    ax.set_ylim(0.35, 1.03)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("预测置信度 (连续核回归)", fontsize=10.5)
    ax.set_ylabel("条件期望准确率 $\\mathbb{P}(Y=\\hat Y\\mid p)$", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax.set_title("(c) 非参数核平滑连续校准曲线与 95% 置信带", fontsize=11.5, fontweight="bold")

    # (d) 按置信度分位点 (20 等分) 的 Brier Score 与 NLL 改善轨迹
    ax = axes[1, 1]
    style_axes(ax)
    q_edges = np.linspace(0, 100, 21)
    q_mids = (q_edges[:-1] + q_edges[1:]) / 2.0
    for mkey, col, lbl in [("resnet20", C_R20, "ResNet-20"), ("resnet56", C_R56, "ResNet-56")]:
        arr = res[mkey]["eval_raw_arrays"]
        cu = np.array(arr["conf_uncal"])
        bu = np.array(arr["brier_uncal"])
        bt = np.array(arr["brier_ts"])
        nu = np.array(arr["nll_uncal"])
        nt = np.array(arr["nll_ts"])
        qs = np.percentile(cu, q_edges)
        b_diff, n_diff = [], []
        for i in range(20):
            if i == 0:
                mask = (cu >= qs[i]) & (cu <= qs[i + 1])
            else:
                mask = (cu > qs[i]) & (cu <= qs[i + 1])
            b_diff.append(np.mean(bu[mask] - bt[mask]) if np.any(mask) else 0.0)
            n_diff.append(np.mean(nu[mask] - nt[mask]) if np.any(mask) else 0.0)
        ax.plot(q_mids, b_diff, "o-", color=col, lw=1.8, ms=4.5, label=f"{lbl} Brier 下降量")
        ax.plot(q_mids, n_diff, "s--", color=col, lw=1.6, ms=4.0, alpha=0.75, label=f"{lbl} NLL 下降量")
    ax.axhline(0, color="black", lw=1.1, ls=":")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("原始置信度百分位区间 (%)", fontsize=10.5)
    ax.set_ylabel("校准带来的损失降低量 (Uncal $-$ TS)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax.set_title("(d) 各置信度分位区间上的 Brier / NLL 改善轨迹", fontsize=11.5, fontweight="bold")

    save_fig(fig, "fig3_reliability_and_improvement")


def plot_fig4(res):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    plt.subplots_adjust(wspace=0.28)

    # (a)(b) Bootstrap 分布 vs 渐近正态密度
    for idx, (mkey, col, lbl) in enumerate([
        ("resnet20", C_R20, "ResNet-20"),
        ("resnet56", C_R56, "ResNet-56"),
    ]):
        ax = axes[idx]
        style_axes(ax)
        r = res[mkey]
        samples = np.array(r["bootstrap"]["T_samples"])
        T_hat = r["ts_mle"]["T_hat"]
        se_d = r["ts_mle"]["se_T_delta"]
        ci_d = r["ts_mle"]["ci_T_delta_95"]
        ci_b = r["bootstrap"]["T_ci_95"]

        ax.hist(samples, bins=36, density=True, color=col, alpha=0.28, edgecolor="white", label="Bootstrap 直方图 (B=2000)")
        xs = np.linspace(samples.min() - 0.02, samples.max() + 0.02, 250)
        kde = stats.gaussian_kde(samples)
        ax.plot(xs, kde(xs), color=col, lw=2.2, label="Bootstrap KDE")
        ax.plot(xs, stats.norm.pdf(xs, loc=T_hat, scale=se_d), "k--", lw=2.0, label="理论渐近正态 (Delta)")
        ax.axvline(T_hat, color="black", lw=1.3)
        ax.axvline(ci_d[0], color="black", ls=":", lw=1.3, label=f"Delta 95% CI [{ci_d[0]:.3f}, {ci_d[1]:.3f}]")
        ax.axvline(ci_d[1], color="black", ls=":", lw=1.3)
        ax.axvline(ci_b[0], color=C_TS, ls="-.", lw=1.3, label=f"Boot 95% CI [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
        ax.axvline(ci_b[1], color=C_TS, ls="-.", lw=1.3)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_xlabel("温度参数估计 $\\hat T$", fontsize=10.5)
        ax.set_ylabel("概率密度", fontsize=10.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.48)
        ax.legend(loc="upper right", fontsize=7.6, frameon=True)
        ax.set_title(f"({'ab'[idx]}) {lbl} Bootstrap 分布与渐近正态对比", fontsize=11.2, fontweight="bold")

    # (c) 标准化 Q-Q 图 + 95% 置信包络带
    ax = axes[2]
    style_axes(ax)
    for mkey, col, marker, lbl in [
        ("resnet20", C_R20, "o", "ResNet-20"),
        ("resnet56", C_R56, "s", "ResNet-56"),
    ]:
        samples = np.sort(np.array(res[mkey]["bootstrap"]["T_samples"]))
        z_emp = (samples - np.mean(samples)) / np.std(samples, ddof=1)
        n_b = len(samples)
        probs_q = (np.arange(1, n_b + 1) - 0.5) / n_b
        z_theo = stats.norm.ppf(probs_q)
        step = max(1, n_b // 80)
        ax.scatter(z_theo[::step], z_emp[::step], s=18, color=col, marker=marker, alpha=0.75, label=f"{lbl} 经验分位数")
    z_line = np.linspace(-3.5, 3.5, 100)
    ax.plot(z_line, z_line, "k--", lw=1.6, label="标准正态参考线 $y=x$")
    # 95% KS 渐近包络近似
    se_qq = 1.96 * np.sqrt(0.25 / 2000) / np.maximum(stats.norm.pdf(z_line), 0.02)
    se_qq = np.minimum(se_qq, 0.45)
    ax.fill_between(z_line, z_line - se_qq, z_line + se_qq, color="gray", alpha=0.18, label="95% 理论置信包络带")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("理论标准正态分位数", fontsize=10.5)
    ax.set_ylabel("标准化 $\\hat T^*$ 经验分位数", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax.set_title("(c) 正态 Q-Q 检验图 (带 95% 包络带)", fontsize=11.2, fontweight="bold")

    save_fig(fig, "fig4_bootstrap_and_qq")


def plot_fig5(res):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    plt.subplots_adjust(wspace=0.28)

    r20 = res["resnet20"]
    r56 = res["resnet56"]

    # (a) 校准样本量 n_cal 收敛轨迹 + 95% 波动带
    ax = axes[0]
    style_axes(ax)
    for r, col, marker, lbl in [(r20, C_R20, "o", "ResNet-20"), (r56, C_R56, "s", "ResNet-56")]:
        nc = np.array([d["n_cal"] for d in r["sample_size_curve"]])
        tm = np.array([d["T_mean"] for d in r["sample_size_curve"]])
        q025 = np.array([d["T_q025"] for d in r["sample_size_curve"]])
        q975 = np.array([d["T_q975"] for d in r["sample_size_curve"]])
        ax.plot(nc, tm, marker=marker, ls="-", color=col, lw=2.0, ms=4.5, label=f"{lbl} $\\hat T$ 均值")
        ax.fill_between(nc, q025, q975, color=col, alpha=0.16, label=f"{lbl} 95% 经验波动带")
    ax.axhline(1.0, color="black", ls="-.", lw=1.2, label="$H_0: T=1$")
    ax.set_xscale("log")
    ax.set_xticks([200, 300, 500, 1000, 2000, 4000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("校准样本量 $n_{\\mathrm{cal}}$ (对数轴)", fontsize=10.5)
    ax.set_ylabel("温度参数估计 $\\hat T$", fontsize=10.5)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.08), fontsize=7.6, frameon=True)
    ax.set_title("(a) $\\hat T$ 随校准样本量 $n_{\\mathrm{cal}}$ 的收敛轨迹", fontsize=11.2, fontweight="bold")

    # (b) log(SE) vs log(n_cal) 与 -1/2 理论收敛斜率
    ax = axes[1]
    style_axes(ax)
    for r, col, marker, lbl in [(r20, C_R20, "o", "ResNet-20"), (r56, C_R56, "s", "ResNet-56")]:
        nc = np.array([d["n_cal"] for d in r["sample_size_curve"]], dtype=float)
        se = np.array([d["se_delta_mean"] for d in r["sample_size_curve"]])
        sd = np.array([d["T_sd"] for d in r["sample_size_curve"]])
        slope, intercept, _, _, _ = stats.linregress(np.log(nc), np.log(se))
        ax.loglog(nc, se, marker=marker, ls="-", color=col, lw=2.0, ms=4.5,
                  label=f"{lbl} Delta SE (斜率={slope:.3f})")
        ax.loglog(nc, sd, marker=marker, ls=":", color=col, lw=1.5, ms=4.0, alpha=0.8,
                  label=f"{lbl} 经验 SD")
    # -1/2 参考线
    nc_ref = np.array([200, 4000], dtype=float)
    se0 = r20["sample_size_curve"][0]["se_delta_mean"]
    ax.loglog(nc_ref, se0 * (nc_ref / 200.0) ** (-0.5) * 1.15, "k--", lw=1.5, label="理论斜率 $-1/2$ 参考线")
    ax.set_xticks([200, 300, 500, 1000, 2000, 4000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("校准样本量 $n_{\\mathrm{cal}}$ (对数轴)", fontsize=10.5)
    ax.set_ylabel("标准误 $\\mathrm{SE}(\\hat T)$ (对数轴)", fontsize=10.5)
    ax.set_ylim(1.4e-2, 2.5e-1)
    ax.legend(loc="lower left", fontsize=7.6, frameon=True)
    ax.set_title("(b) $\\mathrm{SE}(\\hat T)$ 的 $\\sqrt{n}$ 相合收敛速率验证", fontsize=11.2, fontweight="bold")

    # (c) 评估集综合指标改善幅度对比 (%)
    ax = axes[2]
    style_axes(ax, x_minor=0)
    metrics = ["NLL 下降 (%)", "Brier 下降 (%)", "ECE 下降 (%)", "MCE 下降 (%)"]
    x = np.arange(len(metrics))
    def rel_drop(r, m_key):
        u = r["eval_metrics"]["uncalibrated"]
        c = r["eval_metrics"][m_key]
        return [
            (u["nll"] - c["nll"]) / u["nll"] * 100,
            (u["brier"] - c["brier"]) / u["brier"] * 100,
            (u["ece"] - c["ece"]) / u["ece"] * 100,
            (u["mce"] - c["mce"]) / u["mce"] * 100,
        ]
    w = 0.2
    d_r20_ts = rel_drop(r20, "temp_scaling")
    d_r20_pl = rel_drop(r20, "platt_scaling")
    d_r56_ts = rel_drop(r56, "temp_scaling")
    d_r56_pl = rel_drop(r56, "platt_scaling")
    ax.bar(x - 1.5*w, d_r20_ts, width=w, color=C_R20, label="R20 TempScaling")
    ax.bar(x - 0.5*w, d_r20_pl, width=w, color="#6baed6", label="R20 PlattScaling")
    ax.bar(x + 0.5*w, d_r56_ts, width=w, color=C_R56, label="R56 TempScaling")
    ax.bar(x + 1.5*w, d_r56_pl, width=w, color="#fb6a4a", label="R56 PlattScaling")
    for xi, vals in enumerate([d_r20_ts, d_r20_pl, d_r56_ts, d_r56_pl]):
        for j, val in enumerate(vals):
            ax.text(j + (xi - 1.5)*w, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=6.8, rotation=60)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.8)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.set_ylabel("相对未校准模型的改善幅度 (%)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.6, frameon=True)
    ax.set_title("(c) 独立评估集 ($n=8000$) 四项指标改善对比", fontsize=11.2, fontweight="bold")

    save_fig(fig, "fig5_convergence_and_eval")


def main():
    with open("calibration_full_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    plot_fig1(res)
    plot_fig2(res)
    plot_fig3(res)
    plot_fig4(res)
    plot_fig5(res)
    print("\n[✓] 全部 5 张高质量图表已生成至 figures_calib/ 目录！")


if __name__ == "__main__":
    main()
