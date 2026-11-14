#!/usr/bin/env python3
"""
run_calibration_experiment.py
深度神经网络置信度校准的极大似然推断与假设检验实验脚本

数据源:
- CIFAR-10 官方测试集 (N=10,000, 10 类) 上的独立前向预测:
  * test_predictions_resnet20.npz (ResNet-20, Top-1 92.60%)
  * test_predictions_resnet56.npz (ResNet-56, Top-1 94.37%)
  无训练集数据泄漏; 按分层抽样划分为校准拟合集 (n_cal=2000) 与独立评估集 (n_eval=8000)。

统计方法:
1. 单参数 Temperature Scaling (MLE, 解析 Score 与 Fisher 信息, Newton-Raphson 求解)
2. Delta 方法与 B=2000 非参数 Bootstrap 置信区间
3. 三大假设检验 H0: T=1 (beta=1): Wald, Score, 似然比检验 (LRT)
4. 两参数 Platt Scaling (Logistic 校准) 对照
5. 评估集指标 (Accuracy, NLL, Brier Score, ECE, MCE) 与配对 Wilcoxon / t 检验 / McNemar 检验
6. 校准样本量 n_cal 收敛轨迹 (50..4000, 50 次重复) 与 ECE 分箱敏感性 (M=5..50)
"""

import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 20261111
Z95 = stats.norm.ppf(0.975)  # 1.959963984540054


def softmax_from_logits(logits, beta=1.0):
    z = beta * logits
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def log_likelihood_ts(beta, logits, y_true):
    """Temperature Scaling 对数似然 l(beta), beta = 1/T"""
    z = beta * logits
    z_max = np.max(z, axis=1, keepdims=True)
    log_sum_exp = z_max[:, 0] + np.log(np.sum(np.exp(z - z_max), axis=1))
    z_true = z[np.arange(len(y_true)), y_true]
    return float(np.sum(z_true - log_sum_exp))


def score_and_fisher_ts(beta, logits, y_true):
    """解析一阶导 Score U(beta) 与观测 Fisher 信息 I(beta) = -l''(beta)"""
    probs = softmax_from_logits(logits, beta=beta)
    z_true = logits[np.arange(len(y_true)), y_true]
    ez = np.sum(probs * logits, axis=1)
    ez2 = np.sum(probs * (logits ** 2), axis=1)
    var_z = np.maximum(ez2 - ez ** 2, 0.0)
    U = float(np.sum(z_true - ez))
    I = float(np.sum(var_z))
    return U, I


def fit_temperature_scaling_mle(logits, y_true, max_iter=60, tol=1e-12):
    """带回溯阻尼的 Newton-Raphson 求解严格凹对数似然的 MLE"""
    beta = 1.0
    ll_curr = log_likelihood_ts(beta, logits, y_true)
    for _ in range(max_iter):
        U, I = score_and_fisher_ts(beta, logits, y_true)
        if I <= 1e-15:
            break
        step = U / I
        # 阻尼步长防止小样本下大步长越过正域
        alpha_ls = 1.0
        while alpha_ls > 1e-6:
            cand = beta + alpha_ls * step
            if cand > 0.05 and log_likelihood_ts(cand, logits, y_true) >= ll_curr - 1e-12:
                break
            alpha_ls *= 0.5
        beta_new = max(0.05, beta + alpha_ls * step)
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
        ll_curr = log_likelihood_ts(beta, logits, y_true)

    ll_hat = log_likelihood_ts(beta, logits, y_true)
    ll_null = log_likelihood_ts(1.0, logits, y_true)
    U_hat, I_hat = score_and_fisher_ts(beta, logits, y_true)
    U_null, I_null = score_and_fisher_ts(1.0, logits, y_true)

    T_hat = 1.0 / beta
    se_beta = 1.0 / np.sqrt(I_hat)
    se_T_delta = se_beta / (beta ** 2)

    ci_beta = [beta - Z95 * se_beta, beta + Z95 * se_beta]
    ci_T_delta = [T_hat - Z95 * se_T_delta, T_hat + Z95 * se_T_delta]
    ci_T_inv = [1.0 / ci_beta[1], 1.0 / max(ci_beta[0], 1e-6)]

    # 三大检验 H0: beta = 1 (T = 1)
    wald_stat = float(((beta - 1.0) ** 2) * I_hat)
    score_stat = float((U_null ** 2) / I_null)
    lrt_stat = float(max(0.0, 2.0 * (ll_hat - ll_null)))

    return {
        "beta_hat": float(beta),
        "T_hat": float(T_hat),
        "se_beta": float(se_beta),
        "se_T_delta": float(se_T_delta),
        "ci_beta_95": [float(ci_beta[0]), float(ci_beta[1])],
        "ci_T_delta_95": [float(ci_T_delta[0]), float(ci_T_delta[1])],
        "ci_T_inv_95": [float(ci_T_inv[0]), float(ci_T_inv[1])],
        "ll_hat": float(ll_hat),
        "ll_null": float(ll_null),
        "U_null": float(U_null),
        "I_null": float(I_null),
        "I_hat": float(I_hat),
        "tests": {
            "Wald": {"stat": wald_stat, "p_val": float(1.0 - stats.chi2.cdf(wald_stat, df=1))},
            "Score": {"stat": score_stat, "p_val": float(1.0 - stats.chi2.cdf(score_stat, df=1))},
            "LRT": {"stat": lrt_stat, "p_val": float(1.0 - stats.chi2.cdf(lrt_stat, df=1))},
        },
    }


def fit_platt_scaling(probs, y_true):
    """两参数 Platt Scaling (在 top-1 对数几率上做 Logistic 回归 MLE)"""
    p_max = np.clip(np.max(probs, axis=1), 1e-7, 1.0 - 1e-7)
    x = np.log(p_max / (1.0 - p_max))
    y_bin = (np.argmax(probs, axis=1) == y_true).astype(float)

    def nll(params):
        a, b = params
        z = np.clip(a + b * x, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -np.sum(y_bin * np.log(p) + (1.0 - y_bin) * np.log(1.0 - p))

    res = minimize(nll, x0=[0.0, 1.0], method="BFGS")
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    return {"alpha_hat": a_hat, "beta_hat": b_hat, "nll": float(res.fun)}


def apply_platt_multiclass(probs, alpha, beta):
    """将 Platt 校准后的 top-1 概率按非最大类原比例缩放为多分类概率矩阵"""
    top_idx = np.argmax(probs, axis=1)
    p_max = np.clip(np.max(probs, axis=1), 1e-7, 1.0 - 1e-7)
    x = np.log(p_max / (1.0 - p_max))
    z = np.clip(alpha + beta * x, -30.0, 30.0)
    p_cal_top = 1.0 / (1.0 + np.exp(-z))

    out = probs.copy()
    row_idx = np.arange(len(probs))
    rem_orig = np.maximum(1.0 - p_max, 1e-12)
    scale = (1.0 - p_cal_top) / rem_orig
    out = out * scale[:, None]
    out[row_idx, top_idx] = p_cal_top
    return out


def compute_calibration_metrics(probs, y_true, n_bins=15):
    """计算 Accuracy, NLL, Brier Score, ECE, MCE 及逐分箱明细"""
    y_pred = np.argmax(probs, axis=1)
    correct = (y_pred == y_true).astype(float)
    confidences = np.max(probs, axis=1)

    acc = float(np.mean(correct))
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    nll_arr = -np.log(p_true)
    nll = float(np.mean(nll_arr))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier_arr = np.sum((probs - one_hot) ** 2, axis=1)
    brier = float(np.mean(brier_arr))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_detail = []
    N = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (cnt / N) * gap
            mce = max(mce, gap)
            k_c = int(np.sum(correct[mask]))
            cp_lo = float(stats.beta.ppf(0.025, k_c, cnt - k_c + 1)) if k_c > 0 else 0.0
            cp_hi = float(stats.beta.ppf(0.975, k_c + 1, cnt - k_c)) if k_c < cnt else 1.0
        else:
            bin_acc, bin_conf, gap, cp_lo, cp_hi = np.nan, (lo + hi) / 2, 0.0, np.nan, np.nan
        bins_detail.append({
            "bin_idx": b + 1,
            "bin_low": float(lo),
            "bin_high": float(hi),
            "bin_mid": float((lo + hi) / 2),
            "count": cnt,
            "frac": float(cnt / N),
            "acc": bin_acc,
            "conf": bin_conf,
            "gap": float(gap),
            "cp_low": cp_lo,
            "cp_high": cp_hi,
        })

    return {
        "accuracy": acc,
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "nll_arr": nll_arr,
        "brier_arr": brier_arr,
        "bins": bins_detail,
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """比较两种方法分类准确率是否一致的 McNemar 检验"""
    corr_a = (pred_a == y_true)
    corr_b = (pred_b == y_true)
    b = int(np.sum(corr_a & (~corr_b)))
    c = int(np.sum((~corr_a) & corr_b))
    if b + c == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p_val": 1.0}
    stat = ((abs(b - c) - 1) ** 2) / (b + c) if abs(b - c) >= 1 else 0.0
    p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "stat": float(stat), "p_val": p_val}


def run_bootstrap(logits_cal, y_cal, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_cal)
    T_boots = np.empty(B)
    beta_boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        res = fit_temperature_scaling_mle(logits_cal[idx], y_cal[idx], max_iter=25)
        T_boots[b] = res["T_hat"]
        beta_boots[b] = res["beta_hat"]
    return T_boots, beta_boots


def run_sample_size_curve(logits_cal, y_cal, n_grid, n_reps=50, seed=SEED):
    """研究校准样本量 n_cal 对 T_hat 收敛及 SE ~ n^{-1/2} 的影响"""
    rng = np.random.default_rng(seed)
    # 若网格最大超过 len(y_cal)，从全集抽样
    N = len(y_cal)
    records = []
    for n_sub in n_grid:
        T_list = []
        se_list = []
        for _ in range(n_reps):
            idx = rng.choice(N, size=min(n_sub, N), replace=False)
            res = fit_temperature_scaling_mle(logits_cal[idx], y_cal[idx], max_iter=25)
            T_list.append(res["T_hat"])
            se_list.append(res["se_T_delta"])
        records.append({
            "n_cal": int(n_sub),
            "T_mean": float(np.mean(T_list)),
            "T_sd": float(np.std(T_list, ddof=1)),
            "T_q025": float(np.percentile(T_list, 2.5)),
            "T_q975": float(np.percentile(T_list, 97.5)),
            "se_delta_mean": float(np.mean(se_list)),
        })
    return records


def analyze_model(model_name, npz_path):
    print(f"\n{'='*60}\n[*] 分析模型: {model_name} ({npz_path})\n{'='*60}")
    data = np.load(npz_path)
    y_true = data["y_true"].astype(int)
    probs = data["probs"].astype(np.float64)
    if "logits" in data.files:
        logits = data["logits"].astype(np.float64)
    else:
        logits = np.log(np.clip(probs, 1e-12, 1.0))

    # 分层抽样: 2000 校准集 + 8000 评估集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=8000, random_state=SEED)
    cal_idx, eval_idx = next(sss.split(np.zeros(len(y_true)), y_true))

    logits_cal, y_cal, probs_cal = logits[cal_idx], y_true[cal_idx], probs[cal_idx]
    logits_eval, y_eval, probs_eval = logits[eval_idx], y_true[eval_idx], probs[eval_idx]

    # 1. MLE 拟合 Temperature Scaling
    ts_res = fit_temperature_scaling_mle(logits_cal, y_cal)
    T_hat = ts_res["T_hat"]
    print(f"    T_hat = {T_hat:.4f} (SE_delta={ts_res['se_T_delta']:.4f}), "
          f"95% Delta CI = [{ts_res['ci_T_delta_95'][0]:.4f}, {ts_res['ci_T_delta_95'][1]:.4f}]")
    print(f"    三大检验 (H0: T=1): Wald={ts_res['tests']['Wald']['stat']:.2f} (p={ts_res['tests']['Wald']['p_val']:.2e}), "
          f"Score={ts_res['tests']['Score']['stat']:.2f} (p={ts_res['tests']['Score']['p_val']:.2e}), "
          f"LRT={ts_res['tests']['LRT']['stat']:.2f} (p={ts_res['tests']['LRT']['p_val']:.2e})")

    # 2. Bootstrap (B=2000)
    T_boots, beta_boots = run_bootstrap(logits_cal, y_cal, B=2000, seed=SEED)
    boot_ci = [float(np.percentile(T_boots, 2.5)), float(np.percentile(T_boots, 97.5))]
    print(f"    Bootstrap (B=2000): mean={np.mean(T_boots):.4f}, sd={np.std(T_boots, ddof=1):.4f}, "
          f"95% CI=[{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

    # 3. Platt Scaling 对照
    platt_res = fit_platt_scaling(probs_cal, y_cal)
    print(f"    Platt Scaling: alpha={platt_res['alpha_hat']:.4f}, beta={platt_res['beta_hat']:.4f}")

    # 4. 评估集 (n_eval=8000) 上对比三方法
    probs_uncal = softmax_from_logits(logits_eval, beta=1.0)
    probs_ts = softmax_from_logits(logits_eval, beta=ts_res["beta_hat"])
    probs_platt = apply_platt_multiclass(probs_uncal, platt_res["alpha_hat"], platt_res["beta_hat"])

    m_uncal = compute_calibration_metrics(probs_uncal, y_eval, n_bins=15)
    m_ts = compute_calibration_metrics(probs_ts, y_eval, n_bins=15)
    m_platt = compute_calibration_metrics(probs_platt, y_eval, n_bins=15)

    for tag, m in [("Uncalibrated", m_uncal), ("TempScaling", m_ts), ("PlattScaling", m_platt)]:
        print(f"    [{tag:12s}] Acc={m['accuracy']*100:.2f}%, NLL={m['nll']:.4f}, "
              f"Brier={m['brier']:.4f}, ECE={m['ece']*100:.2f}%, MCE={m['mce']*100:.2f}%")

    # 配对检验 (评估集上样本级 Brier & NLL)
    brier_diff_ts = m_ts["brier_arr"] - m_uncal["brier_arr"]
    nll_diff_ts = m_ts["nll_arr"] - m_uncal["nll_arr"]
    wilc_brier = stats.wilcoxon(brier_diff_ts)
    wilc_nll = stats.wilcoxon(nll_diff_ts)
    ttest_brier = stats.ttest_rel(m_ts["brier_arr"], m_uncal["brier_arr"])
    mcnemar_ts = mcnemar_test(y_eval, np.argmax(probs_uncal, axis=1), np.argmax(probs_ts, axis=1))

    # 5. 似然剖面曲线 (T in [0.6, 2.2], 200 点) 与 Score 曲线
    T_grid = np.linspace(0.6, 2.2, 200)
    ll_grid = np.array([log_likelihood_ts(1.0 / t, logits_cal, y_cal) for t in T_grid])
    beta_grid = np.linspace(0.4, 1.4, 200)
    U_grid = np.array([score_and_fisher_ts(b, logits_cal, y_cal)[0] for b in beta_grid])

    # 6. 分箱数 M 敏感性 (5..50)
    M_grid = [5, 10, 15, 20, 25, 30, 40, 50]
    ece_by_M = {
        "M": M_grid,
        "uncal": [compute_calibration_metrics(probs_uncal, y_eval, n_bins=M)["ece"] for M in M_grid],
        "ts": [compute_calibration_metrics(probs_ts, y_eval, n_bins=M)["ece"] for M in M_grid],
        "platt": [compute_calibration_metrics(probs_platt, y_eval, n_bins=M)["ece"] for M in M_grid],
    }

    # 7. 校准样本量 n_cal 收敛曲线 (从全 10,000 中重抽 50 次)
    n_grid = [200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
    n_curve = run_sample_size_curve(logits, y_true, n_grid, n_reps=50, seed=SEED)

    # 8. 逐类别 (0..9) 置信度与准确率
    class_stats = []
    conf_uncal = np.max(probs_uncal, axis=1)
    conf_ts = np.max(probs_ts, axis=1)
    pred_eval = np.argmax(probs_uncal, axis=1)
    for c in range(10):
        mask_c = (y_eval == c)
        class_stats.append({
            "class_id": c,
            "acc": float(np.mean(pred_eval[mask_c] == y_eval[mask_c])),
            "conf_uncal": float(np.mean(conf_uncal[mask_c])),
            "conf_ts": float(np.mean(conf_ts[mask_c])),
        })

    # 清理大数组再返回 JSON
    def strip_arr(m):
        return {k: v for k, v in m.items() if k not in ("nll_arr", "brier_arr")}

    return {
        "model": model_name,
        "n_total": int(len(y_true)),
        "n_cal": int(len(y_cal)),
        "n_eval": int(len(y_eval)),
        "ts_mle": ts_res,
        "bootstrap": {
            "B": 2000,
            "T_mean": float(np.mean(T_boots)),
            "T_sd": float(np.std(T_boots, ddof=1)),
            "T_ci_95": boot_ci,
            "T_samples": T_boots.tolist(),
        },
        "platt": platt_res,
        "eval_metrics": {
            "uncalibrated": strip_arr(m_uncal),
            "temp_scaling": strip_arr(m_ts),
            "platt_scaling": strip_arr(m_platt),
        },
        "eval_paired_tests": {
            "brier_diff_mean": float(np.mean(brier_diff_ts)),
            "brier_wilcoxon_stat": float(wilc_brier.statistic),
            "brier_wilcoxon_p": float(wilc_brier.pvalue),
            "brier_ttest_stat": float(ttest_brier.statistic),
            "brier_ttest_p": float(ttest_brier.pvalue),
            "nll_diff_mean": float(np.mean(nll_diff_ts)),
            "nll_wilcoxon_p": float(wilc_nll.pvalue),
            "mcnemar": mcnemar_ts,
        },
        "profiles": {
            "T_grid": T_grid.tolist(),
            "ll_grid": ll_grid.tolist(),
            "beta_grid": beta_grid.tolist(),
            "U_grid": U_grid.tolist(),
        },
        "ece_by_M": ece_by_M,
        "sample_size_curve": n_curve,
        "class_stats": class_stats,
        "eval_raw_arrays": {
            "conf_uncal": conf_uncal.tolist(),
            "conf_ts": conf_ts.tolist(),
            "correct": (pred_eval == y_eval).astype(int).tolist(),
            "brier_uncal": m_uncal["brier_arr"].tolist(),
            "brier_ts": m_ts["brier_arr"].tolist(),
            "nll_uncal": m_uncal["nll_arr"].tolist(),
            "nll_ts": m_ts["nll_arr"].tolist(),
        },
    }


def main():
    t0 = time.time()
    results = {}
    results["resnet20"] = analyze_model("ResNet-20", "test_predictions_resnet20.npz")
    results["resnet56"] = analyze_model("ResNet-56", "test_predictions_resnet56.npz")

    # 保存紧凑摘要 JSON (不含大样本数组, 方便正文引用与归档)
    summary = {}
    for k, v in results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk not in ("eval_raw_arrays", "profiles")}
        summary[k]["bootstrap"] = {bk: bv for bk, bv in v["bootstrap"].items() if bk != "T_samples"}
    with open("calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存完整结果供绘图使用
    with open("calibration_full_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    # 输出 CSV 汇总表
    rows = []
    for mkey in ["resnet20", "resnet56"]:
        r = results[mkey]
        for method_key, method_label in [
            ("uncalibrated", "未校准 (T=1)"),
            ("temp_scaling", "Temperature Scaling"),
            ("platt_scaling", "Platt Scaling"),
        ]:
            em = r["eval_metrics"][method_key]
            rows.append({
                "Model": r["model"],
                "Method": method_label,
                "T_hat": 1.0 if method_key == "uncalibrated" else (r["ts_mle"]["T_hat"] if method_key == "temp_scaling" else np.nan),
                "Acc_%": round(em["accuracy"] * 100, 2),
                "NLL": round(em["nll"], 5),
                "Brier": round(em["brier"], 5),
                "ECE_%": round(em["ece"] * 100, 3),
                "MCE_%": round(em["mce"] * 100, 2),
            })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv("calibration_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[✓] 实验完成! 总耗时 {time.time() - t0:.2f}s")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
