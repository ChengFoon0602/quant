"""
report.py — Alpha 055 深度单因子分析。

Alpha 055 公式:
    RANK(|ΔClose(d)|) × RANK(ΔVolume(d)) × -1

经济学含义: 收盘价变动幅度 × 成交量变动 × 反向——极端价量同步变动后
趋向反转。默认 d=4 天。

分析维度:
  1. 全量 IC 分析 — 1068 只 + PIT Universe，Rank IC / IC_IR
  2. 参数相图 — delta_days × top_pct 网格扫描
  3. 分层回测 — 5 分组等权，单调性检验
  4. Walk-Forward — 3 年训练 + 1 年测试滚动窗口
  5. CAPM 分解 — OLS 回归 α/β
  6. Block Bootstrap — 20 天块重采样，显著性检验

模型假设:
  - 日线级别信号，次日开盘执行
  - PIT Universe 动态过滤（上市>252天、非停牌、成交额前300、退市剔除）
  - 双边摩擦成本 0.1%（佣金+印花税+过户费）
"""

import sys
sys.path.insert(0, "../..")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
from itertools import product

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily, cache_summary
from data.universe import build_dynamic_universe, get_listing_info
from signals.alpha191.factors import factor_055
from backtest.cross_section import run_cross_section

REPORT_DIR = "figures"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── 全局配置 ─────────────────────────────────────────────
cache = cache_summary()
ALL_SYMBOLS = sorted(cache["symbol"].tolist())
print(f"全量缓存: {len(ALL_SYMBOLS)} 只")

TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2025-12-31"
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20  # 交易日
ALPHA = 0.05


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════
#  数据加载（一次性，后续步骤复用）
# ══════════════════════════════════════════════════════════
section("数据加载")

# 加载全量股票日线数据
all_dfs = {}
close_data = {}
volume_data = {}
amount_data = {}
n_failed = 0
for sym in ALL_SYMBOLS:
    df = load_daily(sym)
    if df is None or len(df) < 100:
        n_failed += 1
        continue
    # 筛选日期范围
    df = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
    if len(df) < 100:
        n_failed += 1
        continue
    all_dfs[sym] = df
    close_data[sym] = df["close"]
    volume_data[sym] = df["volume"]
    if "amount" in df.columns:
        amount_data[sym] = df["amount"]
    else:
        amount_data[sym] = pd.Series(0.0, index=df.index)

N_VALID = len(close_data)
print(f"有效股票: {N_VALID} 只（剔除 {n_failed} 只数据不足）")

# 构建矩阵
close_matrix = pd.DataFrame(close_data).sort_index()
volume_matrix = pd.DataFrame(volume_data).sort_index()
amount_matrix = pd.DataFrame(amount_data).sort_index()

# ══════════════════════════════════════════════════════════
#  PIT Universe 构建
# ══════════════════════════════════════════════════════════
section("PIT Universe 构建")

listing_dates, delist_dates = get_listing_info()
if not listing_dates:
    print("[INFO] 从本地缓存推断上市日期...")
    for sym in close_matrix.columns:
        df = all_dfs.get(sym)
        if df is not None and len(df) > 0:
            listing_dates[sym] = df.index[0]

universe_mask = build_dynamic_universe(
    close_matrix=close_matrix,
    amount_matrix=amount_matrix,
    volume_matrix=volume_matrix,
    listing_dates=listing_dates,
    delist_dates=delist_dates,
    n_top=300,
)

n_active = universe_mask.sum(axis=1)
print(f"每日 Universe 大小: 均值={n_active.mean():.0f}, "
      f"中位数={n_active.median():.0f}, 最小={n_active.min()}, 最大={n_active.max()}")

delisted_in_cache = [s for s in delist_dates if s in close_matrix.columns]
print(f"缓存中的退市股: {len(delisted_in_cache)} 只")

# 退市信息（用于截面回测引擎）
delist_info = {
    "dates": {s: delist_dates[s] for s in delist_dates
              if s in close_matrix.columns and pd.notna(delist_dates[s])},
    "prices": {},
}


# ── 辅助：从预加载数据计算因子矩阵 ─────────────────────────
def compute_factor_from_loaded(
    delta_days: int = 4
) -> pd.DataFrame:
    """从已加载的 all_dfs 计算 alpha055 因子矩阵。"""
    factor_rows = {}
    for sym, df in all_dfs.items():
        try:
            factor_rows[sym] = factor_055(df, delta_days=delta_days)
        except Exception:
            factor_rows[sym] = pd.Series(np.nan, index=df.index)
    mat = pd.DataFrame(factor_rows).sort_index()
    return mat


# 默认因子矩阵（delta_days=4）
section("因子计算 (delta_days=4)")
factor_default = compute_factor_from_loaded(delta_days=4)

# 对齐日期和股票
common_dates = close_matrix.index.intersection(factor_default.index)
common_syms = close_matrix.columns.intersection(factor_default.columns)
close_aligned = close_matrix.loc[common_dates, common_syms]
factor_aligned = factor_default.loc[common_dates, common_syms]

daily_ret = close_aligned.pct_change().fillna(0)
fwd_ret = daily_ret.shift(-1)

print(f"共同日期: {len(common_dates)} 天, 共同股票: {len(common_syms)} 只")

# ══════════════════════════════════════════════════════════
#  1. 全量 IC 分析
# ══════════════════════════════════════════════════════════
section("1. 全量 IC 分析")

ic_series = pd.Series(np.nan, index=common_dates)
n_stocks_ic = []

for d in common_dates:
    if d not in factor_aligned.index or d not in fwd_ret.index:
        continue
    f = factor_aligned.loc[d].dropna()
    r = fwd_ret.loc[d]
    mask = f.notna() & r.notna()
    if mask.sum() < 10:
        continue
    ic = f[mask].rank().corr(r[mask].rank())
    if pd.isna(ic):
        continue
    ic_series[d] = ic
    n_stocks_ic.append(mask.sum())

ic_series = ic_series.dropna()
ic_mean = ic_series.mean()
ic_std = ic_series.std()
ic_ir = ic_mean / ic_std if ic_std > 0 else 0
ic_pos = (ic_series > 0).mean()
ic_t = ic_mean / ic_std * np.sqrt(len(ic_series)) if ic_std > 0 else 0
cum_ic = ic_series.cumsum()

print(f"  截面平均股票数: {np.mean(n_stocks_ic):.0f}")
print(f"  Rank IC 均值:  {ic_mean:.4f}")
print(f"  Rank IC 标准差: {ic_std:.4f}")
print(f"  IC_IR:           {ic_ir:.3f}")
print(f"  t 值:            {ic_t:.2f}  {'✓ 显著' if abs(ic_t) > 2 else '✗ 不显著'}")
print(f"  正 IC 占比:      {ic_pos*100:.1f}%")

# IC 图
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
colors = ["#4CAF50" if v > 0 else "#F44336" for v in ic_series]
ax.bar(range(len(ic_series)), ic_series.values, color=colors, alpha=0.5, width=1)
ax.axhline(0, color="#333", linewidth=0.5)
ax.axhline(ic_mean, color="#2196F3", linestyle="--", linewidth=1.5,
           label=f"Mean IC={ic_mean:.4f}")
ax.legend(fontsize=9); ax.set_ylabel("Rank IC"); ax.grid(True, alpha=0.3)
ax.set_title(f"Alpha 055 — 全量 Rank IC 序列  |  IC_IR={ic_ir:.3f}  |  t={ic_t:.1f}")

ax = axes[1]
ax.plot(ic_series.index, cum_ic, color="#2196F3", linewidth=1)
ax.fill_between(ic_series.index, 0, cum_ic.values,
                where=(cum_ic.values >= 0), color="#4CAF50", alpha=0.1)
ax.fill_between(ic_series.index, 0, cum_ic.values,
                where=(cum_ic.values < 0), color="#F44336", alpha=0.1)
ax.axhline(0, color="#333", linewidth=0.5)
ax.set_ylabel("Cumulative IC"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
ax.set_title(f"累计 IC (正IC占比: {ic_pos*100:.0f}%)")
plt.tight_layout(); plt.savefig(REPORT_DIR + "/01_ic_analysis.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/01_ic_analysis.png")

# ══════════════════════════════════════════════════════════
#  2. 参数相图 — delta_days × top_pct 网格扫描
# ══════════════════════════════════════════════════════════
section("2. 参数相图")

delta_range = [2, 3, 4, 5, 6, 8, 10, 15, 20]
top_range = [0.05, 0.10, 0.15, 0.20]

scan_results = []
for delta_days in delta_range:
    print(f"  delta_days={delta_days} ...", end=" ", flush=True)
    f_mat = compute_factor_from_loaded(delta_days=delta_days)
    # 对齐
    cd = close_matrix.index.intersection(f_mat.index)
    cs = close_matrix.columns.intersection(f_mat.columns)
    c = close_matrix.loc[cd, cs]
    f = f_mat.loc[cd, cs]

    for tp in top_range:
        try:
            res = run_cross_section(
                c, f, top_pct=tp,
                universe=universe_mask,
                delist_info=delist_info,
            )
            scan_results.append({
                "delta_days": delta_days,
                "top_pct": tp,
                "sharpe": res["sharpe"],
                "ann_return": res["ann_return"],
                "max_drawdown": res["max_drawdown"],
                "information_ratio": res["information_ratio"],
            })
        except Exception as e:
            scan_results.append({
                "delta_days": delta_days, "top_pct": tp,
                "sharpe": np.nan, "ann_return": np.nan,
                "max_drawdown": np.nan, "information_ratio": np.nan,
            })
    print(f"ok")

df_scan = pd.DataFrame(scan_results)
# 打印最优参数
best_idx = df_scan["sharpe"].idxmax()
best = df_scan.loc[best_idx] if not pd.isna(best_idx) else None
if best is not None:
    print(f"\n  最优参数: delta_days={int(best['delta_days'])}, top_pct={best['top_pct']*100:.0f}%")
    print(f"  最优夏普: {best['sharpe']:.3f}")
    BEST_DELTA = int(best["delta_days"])
    BEST_TOPPCT = float(best["top_pct"])
else:
    BEST_DELTA = 4
    BEST_TOPPCT = 0.2

# 热力图
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Sharpe 热力图
ax = axes[0]
pivot_sr = df_scan.pivot_table(values="sharpe", index="delta_days", columns="top_pct")
im = ax.imshow(pivot_sr.values, aspect="auto", cmap="RdYlGn",
               vmin=pivot_sr.values.min(), vmax=pivot_sr.values.max())
ax.set_xticks(range(len(top_range))); ax.set_xticklabels([f"{t*100:.0f}%" for t in top_range])
ax.set_yticks(range(len(delta_range))); ax.set_yticklabels([str(d) for d in delta_range])
ax.set_xlabel("Top Pct"); ax.set_ylabel("delta_days")
for i in range(len(delta_range)):
    for j in range(len(top_range)):
        v = pivot_sr.values[i][j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "black")
ax.set_title(f"Alpha 055 参数相图 — Sharpe  |  最优 delta={BEST_DELTA}, top={BEST_TOPPCT*100:.0f}%")
plt.colorbar(im, ax=ax, shrink=0.8)

# Ann Return 热力图
ax = axes[1]
pivot_ar = df_scan.pivot_table(values="ann_return", index="delta_days", columns="top_pct")
im = ax.imshow(pivot_ar.values * 100, aspect="auto", cmap="RdYlGn")
ax.set_xticks(range(len(top_range))); ax.set_xticklabels([f"{t*100:.0f}%" for t in top_range])
ax.set_yticks(range(len(delta_range))); ax.set_yticklabels([str(d) for d in delta_range])
ax.set_xlabel("Top Pct"); ax.set_ylabel("delta_days")
for i in range(len(delta_range)):
    for j in range(len(top_range)):
        v = pivot_ar.values[i][j] * 100
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 5 else "black")
ax.set_title("Alpha 055 参数相图 — 年化收益率 (%)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/02_param_scan.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/02_param_scan.png")

# ══════════════════════════════════════════════════════════
#  3. 分层回测（最优参数）
# ══════════════════════════════════════════════════════════
section("3. 分层回测")

n_groups = 5
group_rets = {i: [] for i in range(n_groups)}
group_dates = []

for d in common_dates:
    if d not in factor_aligned.index:
        continue
    f = factor_aligned.loc[d]
    # 应用 universe mask
    if d in universe_mask.index:
        eligible = universe_mask.loc[d].reindex(f.index, fill_value=False)
        f = f[eligible]
    f = f.dropna()
    if len(f) < n_groups * 3:
        continue
    labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
    if labels.nunique() < n_groups:
        continue
    r_next = fwd_ret.loc[d] if d in fwd_ret.index else None
    if r_next is None or r_next.isna().all():
        continue
    group_dates.append(d)
    for g in range(n_groups):
        syms_g = labels[labels == g].index
        ret_g = r_next[syms_g].mean()
        group_rets[g].append(ret_g)

# 各组净值 + 多空
group_equity = {}
group_ann = {}
for g in range(n_groups):
    r = pd.Series([x for x in group_rets[g] if not np.isnan(x)])
    eq = (1 + r).cumprod() if len(r) > 0 else pd.Series([np.nan])
    group_equity[g] = eq
    n_y = len(r) / 252
    group_ann[g] = (eq.iloc[-1]) ** (1 / n_y) - 1 if n_y > 0 and eq.iloc[-1] > 0 else 0

n_ls = min(len(group_rets[4]), len(group_rets[0]))
long_short_ret = pd.Series(
    [group_rets[4][i] - group_rets[0][i] for i in range(n_ls)],
    index=group_dates[:n_ls])
ls_equity = (1 + long_short_ret).cumprod()
ls_sr = np.sqrt(252) * long_short_ret.mean() / long_short_ret.std() if long_short_ret.std() > 0 else 0
ls_win = (long_short_ret > 0).mean()

# 单调性检验：Q1<Q2<Q3<Q4<Q5?
ann_vals = [group_ann[i] for i in range(n_groups)]
is_monotonic = all(ann_vals[i] < ann_vals[i+1] for i in range(n_groups-1))

print(f"  {'分组':<10} {'累计收益':>10} {'年化收益':>10} {'夏普':>8} {'回撤':>8}")
for g in range(n_groups):
    eq = group_equity[g]
    r = pd.Series(group_rets[g])
    n_y = len(r) / 252
    ann = (eq.iloc[-1]) ** (1/n_y) - 1 if n_y > 0 and eq.iloc[-1] > 0 else 0
    sr = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    print(f"  Q{g+1}       {eq.iloc[-1]-1:>9.1%}  {ann:>9.1%}  {sr:>8.3f}  {dd:>7.1%}")

print(f"\n  多空 (Q5-Q1) 夏普: {ls_sr:.3f}")
print(f"  多空胜率: {ls_win*100:.1f}%")
print(f"  单调性: {'✓ 严格递增' if is_monotonic else '✗ 不单调'}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
colors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
for g in range(n_groups):
    ax.plot(group_equity[g].index, group_equity[g].values,
            color=colors[g], linewidth=0.9, label=f"Q{g+1}")
ax.legend(fontsize=8, ncol=5, loc="upper left"); ax.set_ylabel("Cumulative Return")
ax.set_title(f"Alpha 055 分层回测 (5 分组等权, PIT Universe)"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ls_equity.index, ls_equity.values, color="#2196F3", linewidth=0.9)
dd_ls = ls_equity / ls_equity.cummax() - 1
ax.fill_between(ls_equity.index, ls_equity.values.min() * 0.95,
                ls_equity.values, where=(dd_ls.values < 0),
                color="#F44336", alpha=0.08, linewidth=0)
ax.axhline(1, color="#333", linewidth=0.5)
ax.set_ylabel("Cumulative Return")
ax.set_title(f"Q5/Q1 多空组合  |  夏普={ls_sr:.3f}  |  胜率={ls_win*100:.0f}%")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/03_stratified.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/03_stratified.png")

# ══════════════════════════════════════════════════════════
#  4. Walk-Forward 滚动窗口
# ══════════════════════════════════════════════════════════
section("4. Walk-Forward 滚动窗口")

# 3 年训练窗 + 1 年测试窗，每年滚动
TRAIN_YEARS = 3
TEST_YEARS = 1
all_dates = sorted(common_dates)

# ── 预计算所有 delta_days 版本的因子矩阵 ──
print("  预计算所有 delta_days 因子矩阵...")
factor_cache = {}
for dd in delta_range:
    f_mat = compute_factor_from_loaded(delta_days=dd)
    cd = close_matrix.index.intersection(f_mat.index)
    cs = close_matrix.columns.intersection(f_mat.columns)
    factor_cache[dd] = f_mat.loc[cd, cs]
    print(f"    delta_days={dd} ✓")
c_wf = close_matrix.loc[factor_cache[4].index, factor_cache[4].columns]

wf_results = []
window_start_year = 2010

while True:
    train_start = pd.Timestamp(f"{window_start_year}-01-01")
    train_end = pd.Timestamp(f"{window_start_year + TRAIN_YEARS - 1}-12-31")
    test_start = pd.Timestamp(f"{window_start_year + TRAIN_YEARS}-01-01")
    test_end = pd.Timestamp(f"{window_start_year + TRAIN_YEARS + TEST_YEARS - 1}-12-31")

    if test_start > pd.Timestamp("2025-06-01"):
        break

    # 筛选日期
    train_dates = [d for d in all_dates if train_start <= d <= train_end]
    test_dates = [d for d in all_dates if test_start <= d <= test_end]

    if len(train_dates) < 500 or len(test_dates) < 200:
        window_start_year += 1
        continue

    # 训练窗内扫描最优 delta_days（从缓存切片）
    best_wf_sr = -999
    best_wf_delta = 4
    for dd in delta_range:
        f_train = factor_cache[dd].loc[train_dates]
        c_train_slice = c_wf.loc[train_dates]
        try:
            res = run_cross_section(
                c_train_slice, f_train,
                top_pct=0.2, universe=universe_mask, delist_info=delist_info,
            )
            if res["sharpe"] > best_wf_sr:
                best_wf_sr = res["sharpe"]
                best_wf_delta = dd
        except Exception:
            continue

    # 用最优参数在测试窗回测（从缓存切片）
    f_test = factor_cache[best_wf_delta].loc[test_dates]
    c_test_slice = c_wf.loc[test_dates]

    try:
        res_test = run_cross_section(
            c_test_slice, f_test,
            top_pct=0.2, universe=universe_mask, delist_info=delist_info,
        )
    except Exception:
        window_start_year += 1
        continue

    wf_results.append({
        "train_window": f"{train_start.year}-{train_end.year}",
        "test_window": f"{test_start.year}-{test_end.year}",
        "best_delta": best_wf_delta,
        "train_sr": best_wf_sr,
        "test_sr": res_test["sharpe"],
        "test_ann": res_test["ann_return"],
        "test_dd": res_test["max_drawdown"],
    })
    print(f"  {train_start.year}-{train_end.year} → {test_start.year}-{test_end.year}: "
          f"最优delta={best_wf_delta}, 训练SR={best_wf_sr:.2f}, 测试SR={res_test['sharpe']:.2f}")

    window_start_year += 1

df_wf = pd.DataFrame(wf_results)
if len(df_wf) > 0:
    wf_test_srs = df_wf["test_sr"].values
    wf_mean_sr = np.mean(wf_test_srs)
    wf_pos_ratio = (wf_test_srs > 0).mean()
    wf_delta_stable = df_wf["best_delta"].value_counts().index[0]
    wf_delta_stability = df_wf["best_delta"].value_counts().iloc[0] / len(df_wf)

    print(f"\n  Walk-Forward 窗口数: {len(df_wf)}")
    print(f"  样本外 SR 均值: {wf_mean_sr:.3f}")
    print(f"  样本外 SR > 0 占比: {wf_pos_ratio*100:.0f}%")
    print(f"  最频繁最优 delta: {wf_delta_stable} (稳定性 {wf_delta_stability*100:.0f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = range(len(df_wf))
    ax.bar(x, df_wf["train_sr"], width=0.35, color="#2196F3", alpha=0.7, label="训练 SR")
    ax.bar([i + 0.35 for i in x], df_wf["test_sr"], width=0.35, color="#F44336", alpha=0.7, label="测试 SR")
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(df_wf["test_window"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Walk-Forward 滚动窗口 SR  |  测试SR均值={wf_mean_sr:.2f}")

    ax = axes[1]
    delta_counts = df_wf["best_delta"].value_counts().sort_index()
    ax.bar(delta_counts.index.astype(str), delta_counts.values,
           color="#4CAF50", alpha=0.7)
    ax.set_xlabel("delta_days"); ax.set_ylabel("被选为最优的次数")
    ax.set_title(f"最优参数稳定性 — delta_days (最频={wf_delta_stable})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(); plt.savefig(REPORT_DIR + "/04_walk_forward.png", dpi=150); plt.close()
    print(f"  → {REPORT_DIR}/04_walk_forward.png")
else:
    wf_mean_sr = np.nan
    wf_pos_ratio = np.nan
    wf_delta_stable = 4
    wf_delta_stability = np.nan
    print("  [WARN] Walk-Forward 无有效窗口")

# ══════════════════════════════════════════════════════════
#  5. CAPM 分解
# ══════════════════════════════════════════════════════════
section("5. CAPM 分解")

# 全时段策略收益（默认参数 delta_days=4, top_pct=0.2）
res_full = run_cross_section(
    close_aligned, factor_aligned, top_pct=0.2,
    universe=universe_mask, delist_info=delist_info,
)
strategy_net = res_full["strategy_net"].dropna()

# 市场收益 = 等权全市场
mkt_ret = daily_ret.mean(axis=1)
mkt_ret = mkt_ret.reindex(strategy_net.index).fillna(0)

# OLS: r_strat = α + β * r_mkt + ε
X = mkt_ret.values
Y = strategy_net.values
mask = ~np.isnan(X) & ~np.isnan(Y)
X, Y = X[mask], Y[mask]

n = len(X)
X_mean = X.mean()
Y_mean = Y.mean()
cov_xy = np.cov(X, Y)[0, 1]
var_x = np.var(X)
beta = cov_xy / var_x if var_x > 0 else 0
alpha_daily = Y_mean - beta * X_mean

# α 的 t 检验
residuals = Y - (alpha_daily + beta * X)
se_alpha = np.sqrt(np.var(residuals) / n / var_x * (var_x + X_mean**2)) if var_x > 0 else np.inf
alpha_t = alpha_daily / se_alpha if se_alpha > 0 else 0
alpha_annual = alpha_daily * 252
r_squared = 1 - np.var(residuals) / np.var(Y) if np.var(Y) > 0 else 0

print(f"  回归样本: {n} 天")
print(f"  α (日):   {alpha_daily*100:.4f}%  (年化 {alpha_annual*100:.2f}%)")
print(f"  β:        {beta:.3f}")
print(f"  R²:       {r_squared:.3f}")
print(f"  α t 值:   {alpha_t:.2f}  {'✓ 显著' if abs(alpha_t) > 2 else '✗ 不显著'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(X * 100, Y * 100, s=3, alpha=0.3, color="#2196F3")
x_range = np.linspace(X.min(), X.max(), 100)
ax.plot(x_range * 100, (alpha_daily + beta * x_range) * 100,
        color="#F44336", linewidth=1.5, label=f"α={alpha_annual*100:.1f}%/yr, β={beta:.2f}")
ax.axhline(0, color="#333", linewidth=0.5); ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Market Return (%)"); ax.set_ylabel("Strategy Return (%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("CAPM 回归 — 策略收益 vs 市场收益")

ax = axes[1]
# 累计 α
cum_alpha = (alpha_daily * np.ones(n)).cumsum()
cum_mkt = (beta * X).cumsum()
ax.plot(strategy_net.index[mask], np.cumprod(1 + Y),
        color="#4CAF50", linewidth=0.8, label="策略总收益")
ax.plot(strategy_net.index[mask], np.cumprod(1 + beta * X),
        color="#F44336", linewidth=0.8, alpha=0.6, label=f"市场 β 部分 (β={beta:.2f})")
ax.plot(strategy_net.index[mask], np.cumprod(1 + alpha_daily * np.ones(n)),
        color="#2196F3", linewidth=0.8, linestyle="--", label=f"α 累积 (年化{alpha_annual*100:.1f}%)")
ax.legend(fontsize=8); ax.set_ylabel("Cumulative Return"); ax.grid(True, alpha=0.3)
ax.set_title("收益分解 — α vs β")
plt.tight_layout(); plt.savefig(REPORT_DIR + "/05_capm.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/05_capm.png")

# ══════════════════════════════════════════════════════════
#  6. Block Bootstrap 显著性检验
# ══════════════════════════════════════════════════════════
section("6. Block Bootstrap 显著性检验")

# 使用多空日收益率序列
ls_rets = long_short_ret.dropna().values
n_days = len(ls_rets)

# 构建块索引
n_blocks = int(np.ceil(n_days / BLOCK_SIZE))
block_starts = np.arange(0, n_days - BLOCK_SIZE + 1)

rng = np.random.default_rng(42)
sr_boot = np.empty(N_BOOTSTRAP)
ann_boot = np.empty(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    # 随机抽块
    chosen_blocks = rng.choice(block_starts, size=n_blocks, replace=True)
    boot_ret = []
    for start in chosen_blocks:
        boot_ret.extend(ls_rets[start:start + BLOCK_SIZE])
    boot_ret = np.array(boot_ret[:n_days])

    eq = (1 + boot_ret).cumprod()
    n_y = n_days / 252
    ann = (eq[-1]) ** (1 / n_y) - 1 if n_y > 0 and eq[-1] > 0 else 0
    sr = np.sqrt(252) * boot_ret.mean() / boot_ret.std() if boot_ret.std() > 1e-12 else 0
    sr_boot[i] = sr
    ann_boot[i] = ann

real_sr = ls_sr
boot_mean = sr_boot.mean()
boot_std = sr_boot.std()
p_val = (sr_boot >= real_sr).mean()
ci_lo, ci_hi = np.percentile(sr_boot, [2.5, 97.5])
p_pos = (sr_boot > 0).mean()

print(f"  Bootstrap: {N_BOOTSTRAP} 次, 块大小 {BLOCK_SIZE} 天")
print(f"  多空真实 SR = {real_sr:.4f}")
print(f"  Bootstrap SR 均值: {boot_mean:.4f} ± {boot_std:.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  P(SR ≥ 真实): {p_val:.4f}  {'✓ 显著' if p_val < ALPHA else '✗ 不显著'}")
print(f"  P(SR > 0): {p_pos*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(sr_boot, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(real_sr, color="#F44336", linewidth=2, linestyle="--",
           label=f"真实 SR={real_sr:.3f}")
ax.axvline(0, color="#999", linewidth=1, linestyle=":")
ax.axvline(ci_lo, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(ci_hi, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.fill_betweenx([0, ax.get_ylim()[1]], ci_lo, ci_hi, color="#4CAF50", alpha=0.05)
ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Density")
ax.set_title(f"Block Bootstrap 多空 SR 分布 (N={N_BOOTSTRAP}, block={BLOCK_SIZE}d)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(np.sort(sr_boot), np.arange(1, N_BOOTSTRAP + 1) / N_BOOTSTRAP,
        color="#2196F3", linewidth=1.5)
ax.axvline(real_sr, color="#F44336", linewidth=1.5, linestyle="--",
           label=f"真实 SR={real_sr:.3f}")
ax.axhline(0.05, color="#999", linewidth=0.8, linestyle=":")
ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("CDF")
ax.set_title(f"Bootstrap CDF  |  p={p_val:.3f}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/06_bootstrap.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/06_bootstrap.png")

# ══════════════════════════════════════════════════════════
#  结论汇总
# ══════════════════════════════════════════════════════════
section("结论汇总")

# 全时段回测总指标
total_ret = res_full["total_return"]
ann_ret = res_full["ann_return"]
full_sr = res_full["sharpe"]
full_dd = res_full["max_drawdown"]
full_ir = res_full["information_ratio"]

# 样本内外
close_train = close_aligned.loc[TRAIN_START:TRAIN_END]
factor_train = factor_aligned.loc[TRAIN_START:TRAIN_END]
close_test = close_aligned.loc[TEST_START:TEST_END]
factor_test = factor_aligned.loc[TEST_START:TEST_END]

res_train = run_cross_section(close_train, factor_train, top_pct=0.2,
                               universe=universe_mask, delist_info=delist_info)
res_test = run_cross_section(close_test, factor_test, top_pct=0.2,
                              universe=universe_mask, delist_info=delist_info)
overfit_ratio = res_test["sharpe"] / res_train["sharpe"] if res_train["sharpe"] > 0 else -999

wf_summary = f"{wf_mean_sr:.3f} (>{0}: {wf_pos_ratio*100:.0f}%)" if not np.isnan(wf_mean_sr) else "N/A"

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                   Alpha 055 深度分析 — 最终判决                    │
  ├──────────────────────────────────────────────────────────────────┤
  │  检验维度                         结果                   判据     │
  ├──────────────────────────────────────────────────────────────────┤
  │  全量 IC_IR            {ic_ir:>38.3f}       {'✓ 显著' if abs(ic_t)>2 else '✗ 弱'}     │
  │  IC t 值               {ic_t:>38.2f}       {'✓' if abs(ic_t)>2 else '✗'}     │
  │  分层单调性            {'Q1<Q2<Q3<Q4<Q5 严格递增':>35}       {'✓' if is_monotonic else '✗'}     │
  │  多空夏普              {ls_sr:>38.3f}       {'✓' if ls_sr > 0.3 else '✗'}     │
  │  参数稳定性 (WF)       {wf_delta_stable:>38}       —       │
  │  样本外 SR (WF 均值)   {wf_mean_sr:>38.3f}       {'✓' if wf_mean_sr > 0 else '✗'}     │
  │  α 年化 (CAPM)         {alpha_annual*100:>35.1f}%  t={alpha_t:.1f}  {'✓' if abs(alpha_t)>2 else '✗'}     │
  │  β (CAPM)              {beta:>38.3f}       —       │
  │  Bootstrap p 值        {p_val:>38.3f}       {'✓' if p_val<0.05 else '✗'}     │
  ├──────────────────────────────────────────────────────────────────┤
  │  训练集 SR (2010-19)   {res_train['sharpe']:>38.3f}                │
  │  测试集 SR (2020-25)   {res_test['sharpe']:>38.3f}                │
  │  过拟合比              {overfit_ratio:>38.2f}       {'可接受' if overfit_ratio>0.5 else '严重'}  │
  │  全时段 SR             {full_sr:>38.3f}                │
  │  全时段年化            {ann_ret*100:>36.1f}%                │
  │  全时段最大回撤        {full_dd*100:>36.1f}%                │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  最终判决:                                                       │
  │    Alpha 055 经全量 {N_VALID} 只 + PIT Universe 深度分析:                │
  │    · IC_IR = {ic_ir:.3f}, 多空 SR = {ls_sr:.3f}, Bootstrap p = {p_val:.3f}       │
  │    · {'各项检验通过——因子具备稳健截面预测能力' if ls_sr>0.3 and p_val<0.05 and is_monotonic else '部分检验未通过——因子信号存在但不够稳健'}│
  │    · 最优参数 delta_days = {BEST_DELTA}, Walk-Forward 稳定性 =        │
  │      {wf_delta_stability*100:.0f}% (最频繁最优={wf_delta_stable})                 │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
""")

# 权益曲线总图
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
eq = res_full["equity"]; bm = res_full["benchmark"]
ax.plot(eq.index, eq.values, color="#4CAF50", linewidth=0.8,
        label=f"Alpha055 top20%  SR={full_sr:.2f}")
ax.plot(bm.index, bm.values, color="#999", linewidth=0.6, alpha=0.7,
        label="等权基准")
ax.axvline(pd.Timestamp(TEST_START), color="#F44336", linestyle="--", linewidth=1)
ax.text(pd.Timestamp(TEST_START), eq.values.max() * 0.95,
        "← 样本外 →", color="#F44336", fontsize=9, ha="left")
ax.fill_between(eq.index, eq.values, bm.values,
                where=(eq.values >= bm.values), color="#4CAF50", alpha=0.08)
ax.fill_between(eq.index, eq.values, bm.values,
                where=(eq.values < bm.values), color="#F44336", alpha=0.08)
ax.set_ylabel("Equity"); ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_title(f"Alpha 055 全时段权益曲线 — PIT Universe ({N_VALID} stocks)")

ax = axes[1]
dd = eq / eq.cummax() - 1
ax.fill_between(eq.index, 0, dd.values * 100, color="#F44336", alpha=0.3, linewidth=0)
ax.plot(eq.index, dd.values * 100, color="#F44336", linewidth=0.6)
ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR + "/00_equity_curve.png", dpi=150); plt.close()
print(f"  → {REPORT_DIR}/00_equity_curve.png")

# ── 输出全部图表列表 ──────────────────────────────────────
print(f"\n图表输出: {os.path.abspath(REPORT_DIR)}/")
for f in sorted(os.listdir(REPORT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")

print(f"\n  === 完成 ===")
