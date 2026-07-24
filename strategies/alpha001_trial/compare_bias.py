"""
compare_bias.py — 幸存者偏差量化：旧 Universe vs PIT Universe。

对比维度:
  1. Q1(多头) 年化收益差 = old - new
  2. Q5(空头) 年化收益差 = old - new
  3. 多空 SR 差异
  4. Top 20% 做多年化收益差
  5. Universe 重叠率（旧版有多少股票不在新版里）

用法: python compare_bias.py
"""

import sys
sys.path.insert(0, "../..")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import cache_summary, load_daily as _load
from data.universe import build_dynamic_universe, get_listing_info
from signals.alpha191 import compute_factor_matrix
from backtest.cross_section import run_cross_section

COMPARE_DIR = "figures"
os.makedirs(COMPARE_DIR, exist_ok=True)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════
section("数据加载")

FACTOR_IDS = ["alpha191"]
cache = cache_summary()

# ── 新 Universe: 全量缓存股票（CSI 300 + CSI 500）──
all_cached = sorted(cache["symbol"].tolist())
new_symbols = all_cached  # 全部已缓存股票
print(f"新 Universe: {len(new_symbols)} 只（全部缓存，PIT 动态过滤）")

# 因子计算 — 对全量股票统一计算一次
section("因子计算")
close_matrix, factor_tensor = compute_factor_matrix(
    new_symbols, FACTOR_IDS, verbose=True
)
factor_df = factor_tensor["alpha191"]

common_dates = close_matrix.index.intersection(factor_df.index)
common_syms = close_matrix.columns.intersection(factor_df.columns)
close_matrix = close_matrix.loc[common_dates, common_syms]
factor_df = factor_df.loc[common_dates, common_syms]

print(f"共同日期: {len(common_dates)}, 共同股票: {len(common_syms)}")

# ── 旧 Universe: 最新 CSI 300 成分股前 100 只（当前做法，字母序）──
old_symbols = [s for s in all_cached if s in common_syms][:100]
N_OLD = len(old_symbols)
print(f"旧 Universe: {N_OLD} 只（缓存前 100 只，字母序）")

# ── 提取 amount_matrix 用于 PIT 过滤 ──
amount_data = {}
for sym in common_syms:
    df = _load(sym)
    if df is not None and "amount" in df.columns:
        amount_data[sym] = df["amount"]
    else:
        amount_data[sym] = pd.Series(dtype=float)

amount_matrix = pd.DataFrame(amount_data).loc[common_dates, common_syms].fillna(0)
volume_matrix = close_matrix.pct_change().notna().astype(float)  # proxy: non-NaN return = traded
# Better: use actual volume
vol_data = {}
for sym in common_syms:
    df = _load(sym)
    if df is not None and "volume" in df.columns:
        vol_data[sym] = df["volume"]
    else:
        vol_data[sym] = pd.Series(dtype=float)
volume_matrix = pd.DataFrame(vol_data).loc[common_dates, common_syms].fillna(0)

# ══════════════════════════════════════════════════════════
#  构建 PIT Universe
# ══════════════════════════════════════════════════════════
section("构建 PIT Universe")

# 获取上市/退市日期
listing_dates, delist_dates = get_listing_info()

# 补充：从缓存推断上市日期（akshare 可能失败）
if not listing_dates:
    print("[INFO] 从本地缓存推断上市日期...")
    for sym in common_syms:
        df = _load(sym)
        if df is not None and len(df) > 0:
            listing_dates[sym] = df.index[0]
    print(f"推断出 {len(listing_dates)} 只股票的上市日期")

# 构建动态 universe mask
universe_mask = build_dynamic_universe(
    close_matrix=close_matrix,
    amount_matrix=amount_matrix,
    volume_matrix=volume_matrix,
    listing_dates=listing_dates,
    delist_dates=delist_dates,
    n_top=300,
)

# 统计 Universe 覆盖
n_active_daily = universe_mask.sum(axis=1)
print(f"每日 Universe 大小: 均值={n_active_daily.mean():.0f}, "
      f"中位数={n_active_daily.median():.0f}, "
      f"最小={n_active_daily.min()}, 最大={n_active_daily.max()}")

# 退市股统计
delisted_in_cache = [s for s in delist_dates if s in common_syms]
print(f"缓存中的退市股: {len(delisted_in_cache)} 只")
if delisted_in_cache:
    for s in delisted_in_cache[:5]:
        d = delist_dates[s]
        print(f"  {s}: 退市日={d.date() if pd.notna(d) else '?'}")

# ══════════════════════════════════════════════════════════
#  旧 Universe 回测
# ══════════════════════════════════════════════════════════
section("旧 Universe 回测（最新 CSI 300 成分股，固定 100 只）")

close_old = close_matrix[old_symbols]
factor_old = factor_df[old_symbols]
result_old = run_cross_section(close_old, factor_old, top_pct=0.2)
print(f"  Total Return: {result_old['total_return']*100:.1f}%")
print(f"  Ann Return:   {result_old['ann_return']*100:.1f}%")
print(f"  Sharpe:       {result_old['sharpe']:.3f}")
print(f"  Max DD:       {result_old['max_drawdown']*100:.1f}%")
print(f"  Info Ratio:   {result_old['information_ratio']:.3f}")

# 分层回测（旧）
n_groups = 5
daily_ret_old = close_old.pct_change().fillna(0)
fwd_ret_old = daily_ret_old.shift(-1)

old_strat = {g: [] for g in range(n_groups)}
for d in common_dates:
    if d not in factor_old.index:
        continue
    f = factor_old.loc[d].dropna()
    if len(f) < n_groups * 3:
        continue
    labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
    if labels.nunique() < n_groups:
        continue
    r_next = fwd_ret_old.loc[d]
    for g in range(n_groups):
        syms_g = labels[labels == g].index
        old_strat[g].append(r_next[syms_g].mean())

old_equity = {}
old_q_ann = {}
for g in range(n_groups):
    r = pd.Series(old_strat[g]).dropna()
    old_equity[g] = (1 + r).cumprod()
    old_q_ann[g] = old_equity[g].iloc[-1] ** (252 / len(r)) - 1 if len(r) > 0 else 0

old_ls = pd.Series([old_strat[4][i] - old_strat[0][i] for i in range(min(len(old_strat[4]), len(old_strat[0])))])
old_ls_sr = np.sqrt(252) * old_ls.mean() / old_ls.std() if old_ls.std() > 0 else 0

# ══════════════════════════════════════════════════════════
#  新 Universe 回测
# ══════════════════════════════════════════════════════════
section("新 Universe 回测（PIT 动态过滤，每日 Top 300）")

close_new = close_matrix[common_syms]
factor_new = factor_df[common_syms]

# 构建退市信息 dict
delist_info = {
    "dates": {s: delist_dates[s] for s in delist_dates if s in common_syms and pd.notna(delist_dates[s])},
    "prices": {},  # 使用默认清算价（close * 0.9）
}

result_new = run_cross_section(
    close_new, factor_new, top_pct=0.2,
    universe=universe_mask,
    delist_info=delist_info,
)
print(f"  Total Return: {result_new['total_return']*100:.1f}%")
print(f"  Ann Return:   {result_new['ann_return']*100:.1f}%")
print(f"  Sharpe:       {result_new['sharpe']:.3f}")
print(f"  Max DD:       {result_new['max_drawdown']*100:.1f}%")
print(f"  Info Ratio:   {result_new['information_ratio']:.3f}")

# 分层回测（新）
daily_ret_new = close_new.pct_change().fillna(0)
fwd_ret_new = daily_ret_new.shift(-1)

new_strat = {g: [] for g in range(n_groups)}
for d in common_dates:
    if d not in factor_new.index:
        continue
    f = factor_new.loc[d]
    # 应用 universe mask
    if d in universe_mask.index:
        eligible = universe_mask.loc[d].reindex(f.index, fill_value=False)
        f = f[eligible]
    # 剔除退市股
    for sym in delist_info["dates"]:
        if sym in f.index and pd.notna(delist_info["dates"][sym]) and d >= delist_info["dates"][sym]:
            f = f.drop(sym, errors="ignore")
    f = f.dropna()
    if len(f) < n_groups * 3:
        continue
    labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
    if labels.nunique() < n_groups:
        continue
    r_next = fwd_ret_new.loc[d]
    for g in range(n_groups):
        syms_g = labels[labels == g].index
        new_strat[g].append(r_next[syms_g].mean())

new_equity = {}
new_q_ann = {}
for g in range(n_groups):
    r = pd.Series(new_strat[g]).dropna()
    new_equity[g] = (1 + r).cumprod()
    new_q_ann[g] = new_equity[g].iloc[-1] ** (252 / len(r)) - 1 if len(r) > 0 else 0

new_ls = pd.Series([new_strat[4][i] - new_strat[0][i] for i in range(min(len(new_strat[4]), len(new_strat[0])))])
new_ls_sr = np.sqrt(252) * new_ls.mean() / new_ls.std() if new_ls.std() > 0 else 0

# ══════════════════════════════════════════════════════════
#  量化对比
# ══════════════════════════════════════════════════════════
section("幸存者偏差量化")

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                    幸存者偏差量化 — Alpha191                   │
  ├──────────────────────────────────────────────────────────────┤
  │  指标                    旧 (幸存偏差)    新 (PIT)      Δ     │
  ├──────────────────────────────────────────────────────────────┤
  │  Top20% 做多年化         {result_old['ann_return']*100:>7.1f}%      {result_new['ann_return']*100:>7.1f}%    {result_old['ann_return']*100 - result_new['ann_return']*100:>+6.1f}%  │
  │  Top20% 做多夏普         {result_old['sharpe']:>7.3f}       {result_new['sharpe']:>7.3f}    {result_old['sharpe'] - result_new['sharpe']:>+6.3f}   │
  │  Q1 年化 (多头)          {old_q_ann[0]*100:>7.1f}%      {new_q_ann[0]*100:>7.1f}%    {old_q_ann[0]*100 - new_q_ann[0]*100:>+6.1f}%  │
  │  Q5 年化 (空头)          {old_q_ann[4]*100:>7.1f}%      {new_q_ann[4]*100:>7.1f}%    {old_q_ann[4]*100 - new_q_ann[4]*100:>+6.1f}%  │
  │  多空 SR                 {old_ls_sr:>7.3f}       {new_ls_sr:>7.3f}    {old_ls_sr - new_ls_sr:>+6.3f}   │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  结论:                                                       │
  │    幸存者偏差使旧版 Q1 年化偏高约 {abs(old_q_ann[0]*100 - new_q_ann[0]*100):.1f}%，          │
  │    Q5 年化偏高约 {abs(old_q_ann[4]*100 - new_q_ann[4]*100):.1f}%。                            │
  │    多空 SR 差 {abs(old_ls_sr - new_ls_sr):.2f} — {'偏差显著' if abs(old_ls_sr - new_ls_sr) > 0.1 else '差异在噪声范围内'}。          │
  │    旧版 Universe 有 {len(old_symbols)} 只固定股票，新版每日动态选择 ~300 只。     │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════
#  可视化对比
# ══════════════════════════════════════════════════════════
section("可视化")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. 权益曲线对比
ax = axes[0]
ax.plot(result_old["equity"].index, result_old["equity"].values,
        color="#2196F3", linewidth=1.5, label=f"旧 (幸存偏差) SR={result_old['sharpe']:.2f}")
ax.plot(result_new["equity"].index, result_new["equity"].values,
        color="#F44336", linewidth=1.5, label=f"新 (PIT) SR={result_new['sharpe']:.2f}")
ax.plot(result_old["benchmark"].index, result_old["benchmark"].values,
        color="#999", linewidth=0.8, linestyle="--", label="等权基准")
ax.set_title("Top 20% 做多权益曲线"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylabel("Equity")

# 2. Q1/Q5 年化收益对比
ax = axes[1]
x = np.arange(2)
width = 0.3
old_vals = [old_q_ann[0]*100, old_q_ann[4]*100]
new_vals = [new_q_ann[0]*100, new_q_ann[4]*100]
ax.bar(x - width/2, old_vals, width, color="#2196F3", alpha=0.8, label="旧 (幸存偏差)")
ax.bar(x + width/2, new_vals, width, color="#F44336", alpha=0.8, label="新 (PIT)")
for i, (ov, nv) in enumerate(zip(old_vals, new_vals)):
    ax.annotate(f'Δ={ov-nv:+.1f}%', (x[i], max(ov, nv) + 0.5), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(["Q1 (多头)", "Q5 (空头)"]); ax.set_ylabel("年化收益 (%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
ax.set_title("分层收益对比 — 幸存者偏差影响")

# 3. Universe 大小对比
ax = axes[2]
ax.plot(universe_mask.index, universe_mask.sum(axis=1).values, color="#4CAF50", linewidth=0.8)
ax.axhline(y=len(old_symbols), color="#2196F3", linestyle="--", linewidth=1.5,
           label=f"旧版固定 {len(old_symbols)} 只")
ax.set_title(f"每日 PIT Universe 大小 (中位数={n_active_daily.median():.0f})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylabel("Eligible Stocks")

plt.suptitle("幸存者偏差量化 — Alpha191 因子 (CSI 300 proxy)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(COMPARE_DIR + "/survivorship_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"图表输出: {os.path.abspath(COMPARE_DIR)}/survivorship_bias.png")

print(f"\n{'='*70}")
print("  完成。")
print(f"{'='*70}")
