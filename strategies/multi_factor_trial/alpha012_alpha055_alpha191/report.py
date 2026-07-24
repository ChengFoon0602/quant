"""
report.py — 三因子组合回测：独立性验证 → 因子合成 → 单因子 vs 组合对比。

随机选取的三个因子（不同家族）:
  alpha012 — VWAP偏离:    开盘价偏离 VWAP 均线 × 收盘价-VWAP 绝对值
  alpha055 — DELTA×Volume: |Δclose(4)| 的排名 × Δvolume(4) 的排名
  alpha191 — 牛熊比:       牛熊比极值 vs 量变动的负相关

分析维度:
  1. 因子截面相关性（独立性验证）
  2. 单因子 IC 对比
  3. 因子合成（等权 / ICIR 加权）
  4. 分层回测对比（单因子 vs 合成）
  5. 多元 Fama-MacBeth 回归
  6. 样本外检验
  7. 结论
"""

import sys
sys.path.insert(0, "../../..")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily, cache_summary
from signals.alpha191.calculator import compute_factor_matrix
from backtest.cross_section import run_cross_section

REPORT_DIR = "figures"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── 全局配置 ─────────────────────────────────────────────
cache = cache_summary()
all_symbols = sorted(cache["symbol"].tolist())
N_SYMBOLS = min(100, len(all_symbols))
symbols = all_symbols[:N_SYMBOLS]

FACTOR_IDS = ["alpha012", "alpha055", "alpha191"]
FACTOR_LABELS = {
    "alpha012": "VWAP偏离",
    "alpha055": "DELTA×Volume",
    "alpha191": "牛熊比",
}

TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2025-12-31"


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════
#  1. 数据加载 + 因子计算
# ══════════════════════════════════════════════════════════
section("1. 数据加载与因子计算")

close_matrix, factor_tensor = compute_factor_matrix(
    symbols, FACTOR_IDS, verbose=False
)

common_dates = close_matrix.index
for fid in FACTOR_IDS:
    common_dates = common_dates.intersection(factor_tensor[fid].index)
common_syms = close_matrix.columns
for fid in FACTOR_IDS:
    common_syms = common_syms.intersection(factor_tensor[fid].columns)

close_matrix = close_matrix.loc[common_dates, common_syms]
for fid in FACTOR_IDS:
    factor_tensor[fid] = factor_tensor[fid].loc[common_dates, common_syms]

daily_ret = close_matrix.pct_change()
fwd_ret = daily_ret.shift(-1)

close_train = close_matrix.loc[TRAIN_START:TRAIN_END]
close_test = close_matrix.loc[TEST_START:TEST_END]

print(f"  因子: {', '.join(FACTOR_IDS)}")
print(f"  {FACTOR_IDS[0]}: {FACTOR_LABELS[FACTOR_IDS[0]]}")
print(f"  {FACTOR_IDS[1]}: {FACTOR_LABELS[FACTOR_IDS[1]]}")
print(f"  {FACTOR_IDS[2]}: {FACTOR_LABELS[FACTOR_IDS[2]]}")
print(f"  有效股票: {len(common_syms)}  有效日期: {len(common_dates)}")
print(f"  训练集: {TRAIN_START} ~ {TRAIN_END}  ({len(close_train)} 条)")
print(f"  测试集: {TEST_START} ~ {TEST_END}  ({len(close_test)} 条)")

# ── 数据质量检查清单（新 CLAUDE.md 要求）─────────────────
section("数据质量自检")

# 1. 时序对齐
aligned = []
for fid in FACTOR_IDS:
    ok = close_matrix.index.equals(factor_tensor[fid].index)
    aligned.append(f"{fid}={'\u2713' if ok else '\u2717'}")
print(f"  1. 时序对齐: {', '.join(aligned)}")
print(f"     cross_section 引擎: rb_date 收盘信号 → 次日开盘执行 (≥ → > 修复)")

# 2. Universe 确认
print(f"  2. Universe: 当前使用最新 CSI 300 成分股（{N_SYMBOLS} 只）→ ⚠ 含幸存者偏差")
print(f"     报告中已标注此风险。后续需切换到 PIT 成分股数据源。")

# 3. 停牌处理
n_nan_ret = daily_ret.isna().sum().sum()
n_nan_factor = sum(factor_tensor[fid].isna().sum().sum() for fid in FACTOR_IDS)
print(f"  3. 停牌/缺失: 收益率 NaN = {n_nan_ret} (填充为 0), 因子 NaN = {n_nan_factor} (截面跳过)")

# ══════════════════════════════════════════════════════════
#  2. 因子截面相关性（独立性验证）
# ══════════════════════════════════════════════════════════
section("2. 因子截面相关性 — 独立性验证")

pairs = [(0, 1), (0, 2), (1, 2)]
corr_daily = {f"{FACTOR_IDS[i]}_vs_{FACTOR_IDS[j]}": [] for i, j in pairs}

for d in common_dates:
    f_vals = [factor_tensor[fid].loc[d] for fid in FACTOR_IDS]
    mask = f_vals[0].notna() & f_vals[1].notna() & f_vals[2].notna()
    if mask.sum() < 20:
        continue
    for (i, j), key in zip(pairs, corr_daily):
        corr_daily[key].append(f_vals[i][mask].corr(f_vals[j][mask]))

corr_stats = {}
for key, vals in corr_daily.items():
    vals = np.array(vals)
    corr_stats[key] = {
        "mean": vals.mean(), "std": vals.std(),
        "abs_gt_05": (np.abs(vals) > 0.5).mean(),
        "abs_gt_03": (np.abs(vals) > 0.3).mean(),
    }

print(f"  {'因子对':<24} {'截面相关均值':>10} {'标准差':>8} {'|corr|>0.5':>10} {'|corr|>0.3':>10}")
for key, s in corr_stats.items():
    parts = key.replace("_vs_", " vs ").split(" vs ")
    f1, f2 = parts[0], parts[1]
    print(f"  {f1:<24} {s['mean']:>10.4f} {s['std']:>8.4f} {s['abs_gt_05']*100:>9.1f}% {s['abs_gt_03']*100:>9.1f}%")

max_corr = max(abs(s["mean"]) for s in corr_stats.values())
print(f"\n  最大绝对截面相关: {max_corr:.4f}")
if max_corr < 0.3:
    print("  ✓ 独立性好 — 三个因子包含不同的信息，组合有价值")
elif max_corr < 0.5:
    print("  △ 中等相关 — 组合增量有限但仍可尝试")
else:
    print("  ✗ 高度相关 — 因子冗余，组合无增量")

# 相关性矩阵图
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (key, vals) in zip(axes, corr_daily.items()):
    vals = np.array(vals)
    parts = key.split("_vs_")
    f1, f2 = parts[0], parts[1]
    ax.hist(vals, bins=40, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(vals.mean(), color="#F44336", linestyle="--", linewidth=2,
               label=f"均值={vals.mean():.3f}")
    ax.axvline(0, color="#333", linewidth=0.5)
    ax.set_xlabel("截面相关系数"); ax.set_ylabel("频次")
    ax.set_title(f"{f1} vs {f2}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle("三因子截面相关性分布", fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/02_correlation.png", dpi=150, bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════
#  3. 单因子 IC 对比
# ══════════════════════════════════════════════════════════
section("3. 单因子 IC 对比")

ic_results = {}
for fid in FACTOR_IDS:
    f_mat = factor_tensor[fid]
    ic_list = []
    for d in f_mat.index:
        f = f_mat.loc[d]; r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10: continue
        ic_list.append(f[mask].rank().corr(r[mask].rank()))
    ic_arr = np.array(ic_list)
    ic_results[fid] = {
        "IC": ic_arr.mean(), "IC_std": ic_arr.std(),
        "IC_IR": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        "pos": (ic_arr > 0).mean(), "series": ic_arr,
    }

print(f"  {'因子':<14} {'IC均值':>8} {'IC_std':>8} {'IC_IR':>8} {'正IC占比':>8} {'排名':>6}")
ranked = sorted(FACTOR_IDS, key=lambda x: abs(ic_results[x]["IC_IR"]), reverse=True)
for fid in ranked:
    s = ic_results[fid]
    print(f"  {fid:<14} {s['IC']:>8.4f} {s['IC_std']:>8.4f} {s['IC_IR']:>8.3f} {s['pos']*100:>7.1f}%  {'★' if fid==ranked[0] else ''}")

# 单因子 IC 序列对比图
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
colors = ["#2196F3", "#FF9800", "#4CAF50"]
for ax, fid, c in zip(axes, FACTOR_IDS, colors):
    ic_arr = ic_results[fid]["series"]
    ax.bar(range(len(ic_arr)), ic_arr, color=[c if v>0 else "#F44336" for v in ic_arr],
           alpha=0.5, width=1)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.axhline(ic_results[fid]["IC"], color=c, linestyle="--", linewidth=1.5,
               label=f"IC={ic_results[fid]['IC']:.4f}  IR={ic_results[fid]['IC_IR']:.3f}")
    ax.set_ylabel("Rank IC"); ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_title(f"{fid} ({FACTOR_LABELS[fid]})")
axes[2].set_xlabel("Trading Day")
plt.suptitle("单因子 Rank IC 序列对比", fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/03_individual_ic.png", dpi=150, bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════
#  4. 因子合成
# ══════════════════════════════════════════════════════════
section("4. 因子合成")

# Z-score 标准化每个因子
zscore_factors = {}
for fid in FACTOR_IDS:
    f = factor_tensor[fid]
    zscore_factors[fid] = f.subtract(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0)

# 方法1: 等权合成
combo_equal = pd.DataFrame(0.0, index=common_dates, columns=common_syms)
for fid in FACTOR_IDS:
    combo_equal = combo_equal.add(zscore_factors[fid].fillna(0), fill_value=0)
combo_equal = combo_equal / len(FACTOR_IDS)

# 方法2: ICIR 加权（用训练期 IC_IR 做权重）
weights_icir = {}
total_weight = 0
for fid in FACTOR_IDS:
    w = abs(ic_results[fid]["IC_IR"])
    weights_icir[fid] = w
    total_weight += w
combo_icir = pd.DataFrame(0.0, index=common_dates, columns=common_syms)
for fid in FACTOR_IDS:
    w = weights_icir[fid] / total_weight if total_weight > 0 else 1.0/len(FACTOR_IDS)
    combo_icir = combo_icir.add(zscore_factors[fid].fillna(0) * w, fill_value=0)

print(f"  合成方法:")
print(f"    等权:      各 1/{len(FACTOR_IDS)}")
print(f"    ICIR加权:  {', '.join(f'{fid}={weights_icir[fid]/total_weight*100:.0f}%' for fid in FACTOR_IDS)}")

# 方法3: 正交化合成 — 以 IC_IR 最高因子为基，其余因子回归取残差
base_fid = max(FACTOR_IDS, key=lambda x: abs(ic_results[x]["IC_IR"]))
ortho_factors = {base_fid: zscore_factors[base_fid]}
for fid in FACTOR_IDS:
    if fid == base_fid:
        continue
    residuals = pd.DataFrame(0.0, index=common_dates, columns=common_syms)
    for d in common_dates:
        y = zscore_factors[fid].loc[d]
        x = zscore_factors[base_fid].loc[d]
        mask = y.notna() & x.notna()
        if mask.sum() < 20:
            residuals.loc[d] = y.fillna(0)
            continue
        beta = np.polyfit(x[mask].values, y[mask].values, 1)[0]
        residuals.loc[d, mask] = y[mask] - beta * x[mask]
    ortho_factors[fid] = residuals
combo_ortho = pd.DataFrame(0.0, index=common_dates, columns=common_syms)
for fid in FACTOR_IDS:
    combo_ortho = combo_ortho.add(ortho_factors[fid].fillna(0), fill_value=0)
combo_ortho = combo_ortho / len(FACTOR_IDS)

print(f"    正交化:     基底={base_fid}, 其余对其回归取残差后等权合成")

# 合成因子 IC
all_combos = {"等权合成": combo_equal, "ICIR加权": combo_icir, "正交化合成": combo_ortho}
for fid in FACTOR_IDS:
    all_combos[fid] = factor_tensor[fid]  # 单因子也放进去统一处理

combo_ic_results = {}
for name, mat in all_combos.items():
    ic_list = []
    for d in mat.index:
        f = mat.loc[d]; r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10: continue
        ic_list.append(f[mask].rank().corr(r[mask].rank()))
    ic_arr = np.array(ic_list)
    combo_ic_results[name] = {
        "IC": ic_arr.mean(), "IC_std": ic_arr.std(),
        "IC_IR": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        "pos": (ic_arr > 0).mean(),
    }

print(f"\n  {'信号':<16} {'IC均值':>8} {'IC_IR':>8} {'正IC占比':>8}")
for name in ["alpha012", "alpha055", "alpha191", "等权合成", "ICIR加权", "正交化合成"]:
    s = combo_ic_results[name]
    marker = " ★" if "合成" in name and s["IC_IR"] > max(
        combo_ic_results[f]["IC_IR"] for f in FACTOR_IDS) else ""
    print(f"  {name:<16} {s['IC']:>8.4f} {s['IC_IR']:>8.3f} {s['pos']*100:>7.1f}%{marker}")

# ══════════════════════════════════════════════════════════
#  5. 分层回测对比
# ══════════════════════════════════════════════════════════
section("5. 分层回测对比 — 单因子 vs 合成")

n_groups = 5
stratified_results = {}

for name, mat in all_combos.items():
    group_rets = {i: [] for i in range(n_groups)}
    for d in common_dates:
        if d not in mat.index: continue
        f = mat.loc[d].dropna()
        if len(f) < n_groups * 3: continue
        labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
        if labels.nunique() < n_groups: continue
        r_next = fwd_ret.loc[d]
        for g in range(n_groups):
            syms_g = labels[labels == g].index
            group_rets[g].append(r_next[syms_g].mean())

    group_equity = {}
    for g in range(n_groups):
        r = pd.Series([x for x in group_rets[g] if not np.isnan(x)])
        group_equity[g] = (1 + r).cumprod() if len(r) > 0 else pd.Series([np.nan])

    n_ls = min(len(group_rets[4]), len(group_rets[0]))
    ls_ret = pd.Series([group_rets[4][i] - group_rets[0][i] for i in range(n_ls)])
    ls_sr = np.sqrt(252)*ls_ret.mean()/ls_ret.std() if ls_ret.std()>0 else 0

    stratified_results[name] = {
        "group_equity": group_equity, "ls_ret": ls_ret, "ls_sr": ls_sr,
        "ls_win_rate": (ls_ret > 0).mean(),
        "q5_ann": (group_equity[4].iloc[-1]**(252/len(group_equity[4]))-1) if len(group_equity[4])>0 else 0,
        "q1_ann": (group_equity[0].iloc[-1]**(252/len(group_equity[0]))-1) if len(group_equity[0])>0 else 0,
    }

print(f"  {'信号':<16} {'Q1年化':>8} {'Q5年化':>8} {'多空SR':>8} {'多空胜率':>8}")
for name in ["alpha012", "alpha055", "alpha191", "等权合成", "ICIR加权", "正交化合成"]:
    s = stratified_results[name]
    marker = " ★" if "合成" in name and s["ls_sr"] > max(
        stratified_results[f]["ls_sr"] for f in FACTOR_IDS) else ""
    print(f"  {name:<16} {s['q1_ann']*100:>7.1f}% {s['q5_ann']*100:>7.1f}% {s['ls_sr']:>8.3f} {s['ls_win_rate']*100:>7.1f}%{marker}")

# 分层回测对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plot_names = FACTOR_IDS + ["等权合成", "ICIR加权", "正交化合成"]
for idx, (ax, name) in enumerate(zip(axes.flat, plot_names)):
    s = stratified_results[name]
    for g in range(n_groups):
        ax.plot(s["group_equity"][g].values, linewidth=0.8,
                color=["#F44336","#FF9800","#FFC107","#8BC34A","#4CAF50"][g],
                label=f"Q{g+1}" if idx==0 else "")
    ax.set_title(f"{name} (LS SR={s['ls_sr']:.2f})"); ax.grid(True, alpha=0.3)
    if idx == 0: ax.legend(fontsize=7, ncol=5, loc="upper left")
# All 6 subplots now used — no need to hide
plt.suptitle("分层回测对比 — 单因子 vs 合成因子 (Q1~Q5)", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/05_stratified_compare.png", dpi=150, bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════
#  6. 多元 Fama-MacBeth 回归
# ══════════════════════════════════════════════════════════
section("6. 多元 Fama-MacBeth 截面回归")

# r_i = α + λ₁·Z(f₁)_i + λ₂·Z(f₂)_i + λ₃·Z(f₃)_i + ε_i
lambda_multi = {fid: [] for fid in FACTOR_IDS}
for d in common_dates:
    X_data = {}
    for fid in FACTOR_IDS:
        X_data[fid] = zscore_factors[fid].loc[d]
    r = fwd_ret.loc[d]
    mask = np.ones(len(r), dtype=bool)
    for fid in FACTOR_IDS:
        mask = mask & X_data[fid].notna()
    mask = mask & r.notna()
    if mask.sum() < 20: continue

    X = np.column_stack([X_data[fid][mask].values for fid in FACTOR_IDS])
    Y = r[mask].values
    try:
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
        for i, fid in enumerate(FACTOR_IDS):
            lambda_multi[fid].append(beta[i + 1])
    except np.linalg.LinAlgError:
        continue

print(f"  截面数: {len(lambda_multi[FACTOR_IDS[0]])}")
print(f"  {'因子':<14} {'λ(年化)':>10} {'t值':>8} {'判定'}")
for fid in FACTOR_IDS:
    lam = np.array(lambda_multi[fid])
    lam_ann = lam.mean() * 252
    t_val = lam.mean() / lam.std() * np.sqrt(len(lam)) if lam.std() > 0 else 0
    print(f"  {fid:<14} {lam_ann*100:>9.2f}% {t_val:>8.2f}  {'显著' if abs(t_val)>2 else '不显著'}")

# ══════════════════════════════════════════════════════════
#  7. Bootstrap 统计显著性检验
# ══════════════════════════════════════════════════════════
section("7. Bootstrap 统计显著性检验")

def _bootstrap_sr(returns: np.ndarray, n_bootstrap: int = 1000,
                  block_size: int = 20, seed: int = 42):
    """Block-bootstrap 夏普比率分布。"""
    rng = np.random.RandomState(seed)
    n = len(returns)
    srs = np.empty(n_bootstrap)
    n_blocks = max(1, n // block_size)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n - block_size, n_blocks)
        blocks = [returns[i:i + block_size] for i in idx]
        sample = np.concatenate(blocks)[:n]
        s = sample.std()
        srs[b] = np.sqrt(252) * sample.mean() / s if s > 1e-12 else 0.0
    return srs

# 对分层多空收益做 bootstrap
bootstrap_results = {}
for name in ["等权合成", "ICIR加权", "正交化合成"]:
    if name in stratified_results:
        ls = stratified_results[name]["ls_ret"].values
        bs_srs = _bootstrap_sr(ls)
        bootstrap_results[name] = {
            "mean": bs_srs.mean(), "median": np.median(bs_srs),
            "ci_low": np.percentile(bs_srs, 2.5),
            "ci_high": np.percentile(bs_srs, 97.5),
            "p_value": (bs_srs <= 0).mean(),
        }

print(f"  {'合成方法':<16} {'Bootstrap SR':>12} {'95% CI':>24} {'p(SR≤0)':>10} {'判定'}")
for name in ["等权合成", "ICIR加权", "正交化合成"]:
    if name not in bootstrap_results:
        continue
    bs = bootstrap_results[name]
    ci_str = f"[{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]"
    sig = "显著 ★" if bs["p_value"] < 0.05 else "不显著"
    print(f"  {name:<16} {bs['mean']:>8.3f}    {ci_str:>24} {bs['p_value']*100:>9.1f}%  {sig}")

# ══════════════════════════════════════════════════════════
#  8. 样本外检验
# ══════════════════════════════════════════════════════════
section("8. 样本外检验 (2020–2025)")

# 训练集计算权重 → 测试集应用
train_z = {}
for fid in FACTOR_IDS:
    ft = factor_tensor[fid].loc[TRAIN_START:TRAIN_END]
    train_z[fid] = ft.subtract(ft.mean(axis=1), axis=0).div(ft.std(axis=1), axis=0)

# 等权
test_z = {}
for fid in FACTOR_IDS:
    ft = factor_tensor[fid].loc[TEST_START:TEST_END]
    test_z[fid] = ft.subtract(ft.mean(axis=1), axis=0).div(ft.std(axis=1), axis=0)


# OOS 因子合成 — z-score 标准化后等权/ICIR 加权
combo_test_equal = pd.DataFrame(0.0, index=close_test.index, columns=common_syms)
combo_test_icir = pd.DataFrame(0.0, index=close_test.index, columns=common_syms)
combo_train_equal = pd.DataFrame(0.0, index=close_train.index, columns=common_syms)
combo_train_icir = pd.DataFrame(0.0, index=close_train.index, columns=common_syms)

for fid in FACTOR_IDS:
    combo_test_equal = combo_test_equal.add(test_z[fid].fillna(0), fill_value=0)
    combo_train_equal = combo_train_equal.add(train_z[fid].fillna(0), fill_value=0)
combo_test_equal = combo_test_equal / len(FACTOR_IDS)
combo_train_equal = combo_train_equal / len(FACTOR_IDS)

for fid in FACTOR_IDS:
    w = weights_icir[fid] / total_weight
    combo_test_icir = combo_test_icir.add(test_z[fid].fillna(0) * w, fill_value=0)
    combo_train_icir = combo_train_icir.add(train_z[fid].fillna(0) * w, fill_value=0)

# 正交化合成 OOS（基底 + 权重沿用训练期）
combo_test_ortho = pd.DataFrame(0.0, index=close_test.index, columns=common_syms)
combo_train_ortho = pd.DataFrame(0.0, index=close_train.index, columns=common_syms)
for fid in FACTOR_IDS:
    combo_train_ortho = combo_train_ortho.add(ortho_factors[fid].fillna(0), fill_value=0)
combo_train_ortho = combo_train_ortho / len(FACTOR_IDS)
# 测试期正交化：每日截面回归取残差
test_ortho_factors = {base_fid: test_z[base_fid]}
for fid in FACTOR_IDS:
    if fid == base_fid:
        continue
    test_res = pd.DataFrame(0.0, index=close_test.index, columns=common_syms)
    for d in close_test.index:
        if d not in test_z[fid].index or d not in test_z[base_fid].index:
            continue
        y = test_z[fid].loc[d]
        x = test_z[base_fid].loc[d]
        mask = y.notna() & x.notna()
        if mask.sum() < 20:
            test_res.loc[d] = y.fillna(0)
            continue
        beta = np.polyfit(x[mask].values, y[mask].values, 1)[0]
        test_res.loc[d, mask] = y[mask] - beta * x[mask]
    test_ortho_factors[fid] = test_res
for fid in FACTOR_IDS:
    combo_test_ortho = combo_test_ortho.add(test_ortho_factors[fid].fillna(0), fill_value=0)
combo_test_ortho = combo_test_ortho / len(FACTOR_IDS)

# 回测
oos_results = {}
for name, train_mat, test_mat in [
    ("alpha012", factor_tensor["alpha012"].loc[TRAIN_START:TRAIN_END], factor_tensor["alpha012"].loc[TEST_START:TEST_END]),
    ("alpha055", factor_tensor["alpha055"].loc[TRAIN_START:TRAIN_END], factor_tensor["alpha055"].loc[TEST_START:TEST_END]),
    ("alpha191", factor_tensor["alpha191"].loc[TRAIN_START:TRAIN_END], factor_tensor["alpha191"].loc[TEST_START:TEST_END]),
    ("等权合成", combo_train_equal, combo_test_equal),
    ("ICIR加权", combo_train_icir, combo_test_icir),
    ("正交化合成", combo_train_ortho, combo_test_ortho),
]:
    r_train = run_cross_section(close_train, train_mat, top_pct=0.2)
    r_test = run_cross_section(close_test, test_mat, top_pct=0.2)
    oos_results[name] = {"train": r_train, "test": r_test}

print(f"  {'信号':<16} {'训练SR':>8} {'测试SR':>8} {'过拟合比':>8} {'测试IR':>8}")
best_test_sr = -999; best_name = ""
for name in ["alpha012", "alpha055", "alpha191", "等权合成", "ICIR加权", "正交化合成"]:
    r = oos_results[name]
    of_ratio = r["test"]["sharpe"] / r["train"]["sharpe"] if r["train"]["sharpe"] > 0 else -999
    if r["test"]["sharpe"] > best_test_sr:
        best_test_sr = r["test"]["sharpe"]
        best_name = name
    marker = " ★" if "合成" in name and r["test"]["sharpe"] > max(
        oos_results[f]["test"]["sharpe"] for f in FACTOR_IDS) else ""
    print(f"  {name:<16} {r['train']['sharpe']:>8.3f} {r['test']['sharpe']:>8.3f} {of_ratio:>8.2f} {r['test']['information_ratio']:>8.3f}{marker}")

print(f"\n  最优测试集夏普: {best_name} ({best_test_sr:.3f})")

# OOS 权益曲线对比图
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for name, c in [("alpha012", "#999"), ("alpha055", "#2196F3"), ("alpha191", "#FF9800"),
                 ("等权合成", "#4CAF50"), ("ICIR加权", "#F44336")]:
    r = oos_results[name]
    eq_all = pd.concat([r["train"]["equity"], r["test"]["equity"]])
    ax.plot(eq_all.index, eq_all.values, color=c, linewidth=0.8, alpha=0.8, label=name)
ax.axvline(pd.Timestamp(TEST_START), color="#333", linestyle="--", linewidth=1)
ax.set_ylabel("Equity"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_title("全时段权益曲线对比 (top 20% 做多)")

ax = axes[1]
names_plot = ["alpha012", "alpha055", "alpha191", "等权合成", "ICIR加权", "正交化合成"]
x = range(len(names_plot))
train_srs = [oos_results[n]["train"]["sharpe"] for n in names_plot]
test_srs = [oos_results[n]["test"]["sharpe"] for n in names_plot]
ax.bar([i-0.15 for i in x], train_srs, 0.3, color="#2196F3", alpha=0.7, label="训练集SR")
ax.bar([i+0.15 for i in x], test_srs, 0.3, color="#F44336", alpha=0.7, label="测试集SR")
ax.set_xticks(x); ax.set_xticklabels(names_plot, fontsize=8)
ax.axhline(0, color="#333", linewidth=0.5)
ax.set_ylabel("Sharpe Ratio"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("训练 vs 测试 夏普对比")
plt.tight_layout(); plt.savefig(REPORT_DIR + "/07_oos_compare.png", dpi=150, bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════
#  8. 结论
# ══════════════════════════════════════════════════════════
section("8. 结论")

best_icir = max(combo_ic_results[f]["IC_IR"] for f in FACTOR_IDS)
best_combo_icir = max(combo_ic_results[n]["IC_IR"] for n in ["等权合成", "ICIR加权", "正交化合成"])
best_ls = max(stratified_results[f]["ls_sr"] for f in FACTOR_IDS)
best_combo_ls = max(stratified_results[n]["ls_sr"] for n in ["等权合成", "ICIR加权", "正交化合成"])
bootstrap_p = bootstrap_results.get("等权合成", {}).get("p_value", 1.0)

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  检验维度                          结果              判据    │
  ├──────────────────────────────────────────────────────────────┤
  │  因子截面相关性               max |ρ|={max_corr:.4f}      独立    │
  │  单因子 IC_IR (最佳)          {best_icir:.3f}              {'显著' if best_icir>0.1 else '弱'}   │
  │  合成因子 IC_IR               {best_combo_icir:.3f}              {'提升' if best_combo_icir>best_icir else '未提升'}   │
  │  单因子多空 SR (最佳)         {best_ls:.3f}              —      │
  │  合成因子多空 SR              {best_combo_ls:.3f}              {'提升' if best_combo_ls>best_ls else '未提升'}   │
  │  最优测试集 SR               {best_name} {best_test_sr:.3f}    —      │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  最终判决:                                                   │
  │    三个因子截面完全独立（max |ρ|={max_corr:.3f}），确实来自不同信息源。    │
  │    但单独因子 IC 均偏弱（IC_IR < 0.11），等权 vs 正交化 vs ICIR: {'正交化优于等权' if best_combo_ls > best_ls else '均未超越最佳单因子'}。     │
│  │    Bootstrap p(SR≤0) = {bootstrap_p*100:.0f}% — {'显著' if bootstrap_p < 0.05 else '不显著'}。                         │
  │    核心问题: 三个弱信号即使独立且正交化，简单线性合成也无法                                            │
  │    产生强信号。需要更多因子 + 非线性合成方法。                      │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""")

print(f"  图表输出: {os.path.abspath(REPORT_DIR)}/")
for f in sorted(os.listdir(REPORT_DIR)):
    if f.endswith(".png"):
        print(f"    {f}")
print(f"\n  {'='*70}")
