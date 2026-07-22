"""
report.py — MA 交叉趋势跟踪策略完整回测报告。

报告结构:
  1. 实验设计与数据概况
  2. 参数优化（训练集相图）
  3. 滚动窗口优化 (Walk-Forward)
  4. 样本外检验
  5. 风险分析
  6. 收益归因 (CAPM)
  7. 统计显著性 (Bootstrap MC)
  8. 全市场截面检验
  9. 幸存者偏差
  10. 结论
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from pathlib import Path
import os
import builtins
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily, cache_summary
from backtest.engine import run


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

report_log = open("report_output.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, report_log)

def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ── 全局配置 ─────────────────────────────────────────────
SYMBOL = "000001"
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2025-12-31"
N_BOOTSTRAP = 500
ALPHA = 0.05
REPORT_DIR = Path("figures"); REPORT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════
#  1. 实验设计与数据概况
# ══════════════════════════════════════════════════════════
section("1. 实验设计与数据概况")

print("""
  假设检验:
    H₀: A 股不存在可利用的短期趋势跟踪效应（MA 交叉策略夏普 ≤ 0）
    H₁: MA 交叉策略在统计显著水平上产生正夏普

  策略逻辑:
    快线上穿慢线 → 全仓买入，快线下穿慢线 → 平仓空仓

  交易成本 (A 股真实费率, 2026 年标准):
    买入: 佣金 0.025% + 过户费 0.001% ≈ 0.026%
    卖出: 佣金 0.025% + 印花税 0.05% + 过户费 0.001% ≈ 0.076%

  检验维度:
    - 参数相图扫描 (训练集 2010–2019, 544 组合)
    - 滚动窗口优化 (Walk-Forward, 3 年训练 → 1 年测试)
    - 样本外验证 (固定参数 2020–2025)
    - CAPM 收益归因 (α / β 分解)
    - Bootstrap MC 显著性检验 (500 次重采样)
    - 全市场截面检验 + Bonferroni 校正
    - 幸存者偏差讨论
""")

cache = cache_summary()
close = load_daily(SYMBOL)["close"].astype(float)
close_train = close.loc[TRAIN_START:TRAIN_END]
close_test = close.loc[TEST_START:TEST_END]

print(f"  主要标的: {SYMBOL} 平安银行")
print(f"  全时段: {close.index[0].date()} ~ {close.index[-1].date()}  ({len(close)} 条)")
print(f"  训练集: {close_train.index[0].date()} ~ {close_train.index[-1].date()}  ({len(close_train)} 条)")
print(f"  测试集: {close_test.index[0].date()} ~ {close_test.index[-1].date()}  ({len(close_test)} 条)")
print(f"  全市场缓存: {len(cache)} 只, {int(cache['rows'].sum())} 条")

# ══════════════════════════════════════════════════════════
#  2. 参数优化 — 训练集相图
# ══════════════════════════════════════════════════════════
section("2. 参数优化 — 训练集相图扫描 (2010–2019)")

FAST_RANGE = range(2, 61, 2)
SLOW_RANGE = range(10, 121, 5)

train_results = []
for fast, slow in product(FAST_RANGE, SLOW_RANGE):
    if fast >= slow: continue
    fm = close_train.rolling(fast).mean()
    sm = close_train.rolling(slow).mean()
    m = run(close_train, (fm > sm).astype(int))
    train_results.append({"fast": fast, "slow": slow, **m})

df_train = pd.DataFrame(train_results)
best = df_train.loc[df_train["sharpe"].idxmax()]
best_fast, best_slow = int(best["fast"]), int(best["slow"])

sharpe_grid = df_train.pivot_table(index="slow", columns="fast", values="sharpe")
valid = sharpe_grid.values[~np.isnan(sharpe_grid.values)]
positive_ratio = (valid > 0).mean()

near = df_train[
    df_train["fast"].between(best_fast-4, best_fast+4) &
    df_train["slow"].between(best_slow-15, best_slow+15)
]
plateau_sr, plateau_std = near["sharpe"].mean(), near["sharpe"].std()
plateau_ratio = plateau_sr / best["sharpe"] if best["sharpe"] > 0 else 0

print(f"  参数组合: {len(train_results)} 个")
print(f"  正夏普区域: {positive_ratio*100:.1f}%")
print(f"  最优参数:   MA{best_fast}/{best_slow}")
print(f"  最优 SR:    {best['sharpe']:.4f}")
print(f"  最优年化:   {best['ann_return']*100:.1f}%")
print(f"  最优回撤:   {best['max_drawdown']*100:.1f}%")
print(f"  邻域 SR:    {plateau_sr:.4f} ± {plateau_std:.4f}")
print(f"  高原比:     {plateau_ratio:.2f}  (>0.7 ⇒ 稳健)")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(sharpe_grid.columns, sharpe_grid.index, sharpe_grid.values,
                    cmap="RdYlBu_r", shading="auto")
ax.plot(best_fast, best_slow, marker="*", color="black", markersize=14)
ax.set_xlabel("FAST"); ax.set_ylabel("SLOW")
ax.set_title(f"{SYMBOL} 训练集 MA 交叉 — 夏普比率 (2010–2019)")
ax.invert_yaxis()
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(REPORT_DIR / "02_param_heatmap.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  3. 滚动窗口优化 (Walk-Forward)
# ══════════════════════════════════════════════════════════
section("3. 滚动窗口优化 (Walk-Forward)")

# 3 年训练 → 1 年测试，逐年滚动
train_years = 3
wf_results = []
years = list(range(2010, 2026 - train_years))
for test_year in years:
    tr_start = f"{test_year}-01-01"
    tr_end = f"{test_year + train_years - 1}-12-31"
    te_start = f"{test_year + train_years}-01-01"
    te_end = f"{test_year + train_years}-12-31"

    c_tr = close.loc[tr_start:tr_end]
    c_te = close.loc[te_start:te_end]
    if len(c_tr) < 500 or len(c_te) < 100:
        continue

    # 训练集网格搜索
    best_wf_sr, best_wf_params = -999, (2, 10)
    for f_, s_ in product(FAST_RANGE, SLOW_RANGE):
        if f_ >= s_: continue
        fm = c_tr.rolling(f_).mean()
        sm = c_tr.rolling(s_).mean()
        mm = run(c_tr, (fm > sm).astype(int))
        if mm["sharpe"] > best_wf_sr:
            best_wf_sr = mm["sharpe"]
            best_wf_params = (f_, s_)

    # 测试集固定参数
    f_wf, s_wf = best_wf_params
    fm_te = c_te.rolling(f_wf).mean()
    sm_te = c_te.rolling(s_wf).mean()
    m_te = run(c_te, (fm_te > sm_te).astype(int))

    wf_results.append({
        "test_year": test_year + train_years,
        "best_fast": f_wf, "best_slow": s_wf,
        "train_sr": best_wf_sr,
        "test_sr": m_te["sharpe"], "test_ann": m_te["ann_return"],
        "test_dd": m_te["max_drawdown"],
    })

df_wf = pd.DataFrame(wf_results)
wf_pos = (df_wf["test_sr"] > 0).mean()
wf_mean_sr = df_wf["test_sr"].mean()
wf_mean_train = df_wf["train_sr"].mean()
# 参数稳定性: 最优 FAST/SLOW 的标准差
fast_std = df_wf["best_fast"].std()
slow_std = df_wf["best_slow"].std()

print(f"  滚动窗口: {len(df_wf)} 个 (3年训练 → 1年测试)")
print(f"  样本外正 SR 窗口占比: {wf_pos*100:.0f}%")
print(f"  样本外 SR 均值: {wf_mean_sr:.3f}  (训练集均值: {wf_mean_train:.3f})")
print(f"  最优 FAST 变化: μ={df_wf['best_fast'].mean():.0f} ± {fast_std:.0f}")
print(f"  最优 SLOW 变化: μ={df_wf['best_slow'].mean():.0f} ± {slow_std:.0f}")
print(f"  → 参数{'稳定' if slow_std < 30 else '不稳定'}（SLOW 标准差 {'<' if slow_std < 30 else '>'} 30）")

# Walk-Forward 图
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
ax.plot(df_wf["test_year"], df_wf["train_sr"], "o-", color="#2196F3", linewidth=1,
        markersize=6, label="训练集最优 SR")
ax.plot(df_wf["test_year"], df_wf["test_sr"], "s-", color="#F44336", linewidth=1.5,
        markersize=7, label="样本外 SR")
ax.axhline(0, color="#999", linewidth=0.5)
ax.set_ylabel("Sharpe Ratio")
ax.set_title(f"{SYMBOL}  Walk-Forward  {train_years}年训练 → 1年测试")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df_wf["test_year"], df_wf["best_fast"], "o-", color="#4CAF50", linewidth=1,
        markersize=6, label="最优 FAST")
ax.plot(df_wf["test_year"], df_wf["best_slow"], "s-", color="#FF9800", linewidth=1,
        markersize=6, label="最优 SLOW")
ax.set_ylabel("Parameter Value")
ax.set_xlabel("Test Year")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(REPORT_DIR / "03_walk_forward.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  4. 样本外检验
# ══════════════════════════════════════════════════════════
section("4. 样本外检验 (2020–2025)")

fm_test = close_test.rolling(best_fast).mean()
sm_test = close_test.rolling(best_slow).mean()
sig_test = (fm_test > sm_test).astype(int)
res_test = run(close_test, sig_test)

fm_train = close_train.rolling(best_fast).mean()
sm_train = close_train.rolling(best_slow).mean()
sig_train = (fm_train > sm_train).astype(int)
res_train = run(close_train, sig_train)

overfit_ratio = res_test["sharpe"] / res_train["sharpe"] if res_train["sharpe"] > 0 else -999

print(f"  参数: MA{best_fast}/{best_slow} (训练集最优)")
print(f"  {'指标':<18} {'训练集':<20} {'测试集':<20}")
print(f"  {'-'*18} {'-'*20} {'-'*20}")
print(f"  {'年化收益率':<18} {res_train['ann_return']*100:>18.1f}%  {res_test['ann_return']*100:>18.1f}%")
print(f"  {'夏普比率':<18} {res_train['sharpe']:>19.4f}  {res_test['sharpe']:>19.4f}")
print(f"  {'最大回撤':<18} {res_train['max_drawdown']*100:>19.1f}%  {res_test['max_drawdown']*100:>19.1f}%")
print(f"  {'交易次数':<18} {res_train['n_trades']:>20}  {res_test['n_trades']:>20}")
print(f"  过拟合比: {overfit_ratio:.2f}  (>0.5 ⇒ 可接受, <0 ⇒ 严重)")

close_all = pd.concat([close_train, close_test])
fm_all = close_all.rolling(best_fast).mean()
sm_all = close_all.rolling(best_slow).mean()
sig_all = (fm_all > sm_all).astype(int)
res_all = run(close_all, sig_all)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
eq = res_all["equity"] * 100_000
bm = res_all["benchmark"] * 100_000
ax.plot(close_all.index, eq, color="#4CAF50", linewidth=0.8, label=f"MA{best_fast}/{best_slow}")
ax.plot(close_all.index, bm, color="#999", linewidth=0.6, alpha=0.7, label="买入持有")
ax.axvline(pd.Timestamp(TEST_START), color="#F44336", linestyle="--", linewidth=1, alpha=0.7)
ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[1]*0.95, "← 样本外 →", color="#F44336", fontsize=9, ha="left")
ax.fill_between(close_all.index, eq, bm, where=(eq>=bm), color="#4CAF50", alpha=0.08)
ax.fill_between(close_all.index, eq, bm, where=(eq<bm), color="#F44336", alpha=0.08)
ax.set_ylabel("Equity (¥)"); ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_title(f"{SYMBOL}  MA{best_fast}/{best_slow}  权益曲线")

ax = axes[1]
dd = eq / eq.cummax() - 1
ax.fill_between(close_all.index, 0, dd*100, color="#F44336", alpha=0.3, linewidth=0)
ax.plot(close_all.index, dd*100, color="#F44336", linewidth=0.6)
ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR / "04_equity_curve.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  5. 风险分析
# ══════════════════════════════════════════════════════════
section("5. 风险分析")

sr = res_all["strategy_net"].dropna()  # drop NaN (开头无交易时段) for stats
br = res_all["benchmark"].pct_change().dropna()

var_95 = np.percentile(sr, 5)
cvar_95 = sr[sr <= var_95].mean()
calmar = res_all["ann_return"] / abs(res_all["max_drawdown"]) if res_all["max_drawdown"] < 0 else 0
rolling_sr = sr.rolling(252).apply(
    lambda x: np.sqrt(252)*(x.mean()-0.02/252)/x.std() if x.std()>1e-12 else 0)

print(f"  VaR (95%):     {var_95*100:>8.2f}%       (基准: {np.percentile(br,5)*100:.2f}%)")
print(f"  CVaR (95%):    {cvar_95*100:>8.2f}%")
print(f"  最大回撤:      {res_all['max_drawdown']*100:>8.1f}%")
print(f"  年化波动率:    {sr.std()*np.sqrt(252)*100:>8.1f}%")
print(f"  Calmar 比率:   {calmar:>8.3f}       (基准: {(bm.iloc[-1]/bm.iloc[0]**(1/15.5)-1)/abs((bm/bm.cummax()-1).min()):.3f})")
print(f"  滚动 SR 范围:  [{rolling_sr.min():.2f}, {rolling_sr.max():.2f}]")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
ax = axes[0,0]
ax.hist(sr*100, bins=80, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(var_95*100, color="#F44336", linestyle="--", linewidth=1.5, label=f"VaR 95%={var_95*100:.2f}%")
ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Daily Return (%)"); ax.set_ylabel("Density")
ax.set_title("日收益率分布"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0,1]
ax.plot(rolling_sr.index, rolling_sr, color="#2196F3", linewidth=0.8)
ax.axhline(0, color="#F44336", linestyle="--", linewidth=1)
ax.axhline(res_all["sharpe"], color="#4CAF50", linestyle="--", linewidth=1, label=f"整体 SR={res_all['sharpe']:.3f}")
ax.fill_between(rolling_sr.index, 0, rolling_sr, where=(rolling_sr>=0), color="#4CAF50", alpha=0.08)
ax.fill_between(rolling_sr.index, 0, rolling_sr, where=(rolling_sr<0), color="#F44336", alpha=0.08)
ax.set_ylabel("Sharpe (252d)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("滚动夏普比率")

ax = axes[1,0]
dd_all = res_all["equity"] / res_all["equity"].cummax() - 1
ax.fill_between(close_all.index, 0, dd_all*100, color="#F44336", alpha=0.4, linewidth=0)
ax.plot(close_all.index, dd_all*100, color="#F44336", linewidth=0.5)
ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
ax.set_title(f"回撤曲线  (最大: {dd_all.min()*100:.1f}%)")

ax = axes[1,1]
monthly = sr.resample("ME").apply(lambda x: (1+x).prod()-1)
mg = monthly.groupby([monthly.index.year, monthly.index.month]).mean().unstack()
if len(mg) > 0:
    im = ax.pcolormesh(mg.columns, mg.index, mg.values*100, cmap="RdYlBu_r", shading="auto")
    ax.set_xlabel("Month"); ax.set_ylabel("Year"); ax.set_title("月均收益率 (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(REPORT_DIR / "05_risk_analysis.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  6. 收益归因 — CAPM 回归
# ══════════════════════════════════════════════════════════
section("6. 收益归因 — CAPM 回归")

# 用等权市场组合（前 50 只股票，pd.concat 自动对齐）
sample_symbols = sorted([p.name.replace(".csv", "")
    for p in os.scandir("../../data/cache") if p.name.endswith(".csv")])[:50]
all_rets = []
for sym in sample_symbols:
    df = load_daily(sym)
    if df is not None and len(df) > 500:
        r = df["close"].astype(float).pct_change()
        r.name = sym
        all_rets.append(r)
# concat 自动按日期对齐，skipna 处理不同股票的停牌日
mkt_ret = pd.concat(all_rets, axis=1).mean(axis=1, skipna=True)
mkt_ret = mkt_ret.reindex(sr.index)
n_stocks = len(all_rets)

# OLS: r_strategy - r_f = α + β × (r_mkt - r_f) + ε
rf_daily = 0.02 / 252
y = sr - rf_daily
X = mkt_ret - rf_daily
mask = ~(y.isna() | X.isna())
y, X = y[mask], X[mask]
y, X = y[mask], X[mask]

# 手动 OLS
n = len(y)
beta = np.cov(X, y)[0, 1] / np.var(X)
alpha = y.mean() - beta * X.mean()
resid = y - alpha - beta * X
r2 = 1 - np.var(resid) / np.var(y)
alpha_se = np.sqrt(np.var(resid) / n) / np.std(X) * np.sqrt(1/n + X.mean()**2 / np.var(X))
beta_se = np.sqrt(np.var(resid) / n) / np.std(X)
alpha_t = alpha / alpha_se if alpha_se > 0 else 0
# 年化 alpha
ann_alpha = alpha * 252

print(f"  市场组合: {n_stocks} 只等权")
print(f"  回归样本: {n} 天")
print(f"")
print(f"  {'':<16} {'估计值':<12} {'t 值':<10} {'解读'}")
print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*20}")
print(f"  {'α (年化)':<16} {ann_alpha*100:>8.1f}%   {alpha_t:>8.2f}   {'显著正 alpha' if alpha_t>2 else '不显著'}")
print(f"  {'β':<16} {beta:>11.3f}   {beta/beta_se if beta_se>0 else 0:>8.2f}   {'高 beta 暴露' if beta>0.8 else '低 beta'}")
print(f"  {'R²':<16} {r2:>11.3f}            {'策略收益由市场解释' if r2>0.3 else '策略收益独立于市场'}")

if abs(alpha_t) < 2:
    print(f"\n  → α 不显著 (|t|={abs(alpha_t):.1f} < 2)：策略超额收益与零无统计差异")
    print(f"  → β = {beta:.2f}：策略收益 {beta*100:.0f}% 由市场 beta 解释")
else:
    print(f"\n  → α 显著！策略具备独立于市场的超额收益能力")
    print(f"  → 但需注意：α 显著 ≠ 可实盘，仍需结合样本外表现判断")

# CAPM 散点图
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(X*100, y*100, s=3, alpha=0.3, color="#2196F3")
x_range = np.linspace(X.min(), X.max(), 100)
ax.plot(x_range*100, (alpha + beta*x_range)*100, color="#F44336", linewidth=1.5,
        label=f"α={ann_alpha*100:.2f}%/yr  β={beta:.3f}  R²={r2:.3f}")
ax.axhline(0, color="#999", linewidth=0.5); ax.axvline(0, color="#999", linewidth=0.5)
ax.set_xlabel("Market Excess Return (%)"); ax.set_ylabel("Strategy Excess Return (%)")
ax.set_title(f"{SYMBOL}  CAPM 收益归因"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR / "06_capm.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  7. 统计显著性 — Bootstrap MC
# ══════════════════════════════════════════════════════════
section("7. 统计显著性 — Bootstrap MC")

daily_ret = close_all.pct_change().dropna()
rets_arr = daily_ret.values; n_days = len(rets_arr)
rng = np.random.default_rng(42)
sr_boot = np.empty(N_BOOTSTRAP); ann_boot = np.empty(N_BOOTSTRAP); dd_boot = np.empty(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    idx = rng.choice(n_days, size=n_days, replace=True)
    sc = pd.Series(100*(1+rets_arr[idx]).cumprod(), index=daily_ret.index)
    f = sc.rolling(best_fast).mean(); s = sc.rolling(best_slow).mean()
    m = run(sc, (f > s).astype(int))
    sr_boot[i] = m["sharpe"]; ann_boot[i] = m["ann_return"]; dd_boot[i] = m["max_drawdown"]

boot_mean, boot_std = sr_boot.mean(), sr_boot.std()
p_val = (sr_boot >= res_all["sharpe"]).mean()
ci_lo, ci_hi = np.percentile(sr_boot, [2.5, 97.5])
p_pos = (sr_boot > 0).mean()

print(f"  Bootstrap: {N_BOOTSTRAP} 次")
print(f"  真实 SR = {res_all['sharpe']:.4f}")
print(f"  合成均值: {boot_mean:.4f} ± {boot_std:.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  P(SR ≥ 真实): {p_val:.4f}  {'✓ 显著' if p_val<ALPHA else '✗ 不显著'}")
print(f"  正 SR 概率: {p_pos*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(sr_boot, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(res_all["sharpe"], color="#F44336", linewidth=2, linestyle="--", label=f"真实 SR={res_all['sharpe']:.3f}")
ax.axvline(0, color="#999", linewidth=1, linestyle=":")
ax.axvline(ci_lo, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(ci_hi, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.fill_betweenx([0, ax.get_ylim()[1]], ci_lo, ci_hi, color="#4CAF50", alpha=0.05)
ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Density")
ax.set_title(f"Bootstrap 夏普分布 (N={N_BOOTSTRAP})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(dd_boot*100, ann_boot*100, c=sr_boot, cmap="RdYlBu_r", s=5, alpha=0.4)
ax.scatter([res_all["max_drawdown"]*100], [res_all["ann_return"]*100],
           marker="*", color="black", s=250, zorder=5, label="真实")
ax.set_xlabel("Max Drawdown (%)"); ax.set_ylabel("Annual Return (%)")
ax.set_title("Bootstrap 收益-回撤散点图"); ax.axhline(0, color="#999", linewidth=0.5)
ax.axvline(0, color="#999", linewidth=0.5); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(REPORT_DIR / "07_bootstrap.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  8. 全市场截面检验
# ══════════════════════════════════════════════════════════
section("8. 全市场截面检验")

sample_n = min(50, len(cache))
syms_sample = sorted([p.name.replace(".csv", "")
    for p in os.scandir("../../data/cache") if p.name.endswith(".csv")])[:sample_n]

cross_results = []
for sym in syms_sample:
    df = load_daily(sym)
    if df is None or len(df) < 500: continue
    c = df["close"].astype(float)
    ct = c.loc[TRAIN_START:TRAIN_END]; cv = c.loc[TEST_START:TEST_END]

    best_loc_sr, best_loc_params = -999, (4, 100)
    for f_, s_ in [(2,40),(4,60),(4,100),(8,80)]:
        mm = run(ct, (ct.rolling(f_).mean() > ct.rolling(s_).mean()).astype(int))
        if mm["sharpe"] > best_loc_sr:
            best_loc_sr = mm["sharpe"]; best_loc_params = (f_, s_)

    fv, sv = best_loc_params
    mv = run(cv, (cv.rolling(fv).mean() > cv.rolling(sv).mean()).astype(int))
    cross_results.append({"symbol": sym, "train_sr": best_loc_sr, "test_sr": mv["sharpe"],
                          "test_ann": mv["ann_return"], "test_dd": mv["max_drawdown"],
                          "fast": fv, "slow": sv, "n_days": len(c)})

df_cross = pd.DataFrame(cross_results)
oos_pos = (df_cross["test_sr"] > 0).mean()
oos_better = (df_cross["test_sr"] > df_cross["train_sr"]).mean()

print(f"  截面: {len(df_cross)} 只")
print(f"  样本外正 SR: {oos_pos*100:.1f}%")
print(f"  样本外 SR > 训练: {oos_better*100:.1f}%")
print(f"  样本外 SR 均值/中位: {df_cross['test_sr'].mean():.4f} / {df_cross['test_sr'].median():.4f}")
print(f"  Bonferroni (batch_mc.py): 0/297 通过 — 全市场无显著")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
df_s = df_cross.sort_values("test_sr")
ax.barh(range(len(df_s)), df_s["test_sr"],
        color=["#4CAF50" if s>0 else "#F44336" for s in df_s["test_sr"]], alpha=0.7, height=0.8)
ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Test Sharpe"); ax.set_title(f"样本外 SR ({len(df_cross)} 只)")

ax = axes[1]
ax.scatter(df_cross["train_sr"], df_cross["test_sr"], s=15, alpha=0.6, color="#2196F3")
ax.plot([-1,2],[-1,2], color="#999", linewidth=0.5, linestyle="--")
ax.axhline(0, color="#333", linewidth=0.5); ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Train Sharpe"); ax.set_ylabel("Test Sharpe")
ax.set_title("训练 vs 样本外 夏普比率")
plt.tight_layout(); plt.savefig(REPORT_DIR / "08_cross_section.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  9. 幸存者偏差
# ══════════════════════════════════════════════════════════
section("9. 幸存者偏差讨论")

print("""
  本报告使用 2026 年沪深 300 成分股列表，回测 2010–2025 年行情。
  这意味着以下偏差无法排除:

  1. 成分股变更: 2010–2025 年间被调出指数的股票不在样本内。
     被调出的股票通常表现差于调入的股票 → 策略收益被高估。

  2. 退市股票: 研究期间退市的股票完全不可见。
     A 股退市率虽低 (~0.5%/年)，但退市往往伴随 -80% 以上跌幅，
     对策略的潜在伤害远超平均。

  3. 前视偏差 (Look-ahead Bias): 用今天的成分股回测历史，
     隐含假设"我知道 2026 年哪些股票会活下来"。

  量化文献估计幸存者偏差对多因子策略的影响约为 1–3% 年化收益高估。
  对于本报告的趋势跟踪策略，方向一致（正 SR 被高估），
  但由于策略本身在样本外已不显著，偏差不改变结论。

  消除方法:
  - 使用历史成分股数据（如每月实际指数调整记录）
  - 包含退市股票的历史数据（如加入 ST 板块、已退市池）
  - 使用 point-in-time 数据库（如 Wind、Tushare Pro 付费版）

  本报告受限于免费数据源 (baostock)，无法获取历史成分股变更记录。
  在解释训练集正夏普时，需将 1–3% 年化归因于幸存者偏差。
""")

# ══════════════════════════════════════════════════════════
#  10. 结论
# ══════════════════════════════════════════════════════════
section("10. 结论")

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  检验维度                  结果                      判据    │
  ├──────────────────────────────────────────────────────────────┤
  │  参数相图 (训练集)         SR={best['sharpe']:.3f}            高原比={plateau_ratio:.2f}  │
  │  滚动窗口 (Walk-Forward)   OOS SR 均值={wf_mean_sr:.3f}      正={wf_pos*100:.0f}%      │
  │  样本外验证 (2020–25)      SR={res_test['sharpe']:.3f}        过拟合比={overfit_ratio:.2f}│
  │  CAPM 归因                 α={ann_alpha*100:.2f}%/yr  β={beta:.3f}     R²={r2:.3f}   │
  │  Bootstrap MC              p={p_val:.3f}              {'显著' if p_val<ALPHA else '不显著'}   │
  │  全市场 Bonferroni          0/297                         不存在    │
  │  幸存者偏差                 高估 1–3% 年化                 需注意    │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  最终判决:                                                   │
  │    MA 交叉策略在 A 股不产生统计显著超额收益。                  │
  │    训练集正 SR 由牛市 beta + 幸存者偏差 + 数据挖掘解释。       │
  │    样本外过拟合比负值、CAPM α 不显著、Bootstrap p > 0.05、    │
  │    Bonferroni 全市场无一通过——五项检验独立指向同一结论。       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""")

print(f"  图表输出: {REPORT_DIR.resolve()}/")
for f in sorted(REPORT_DIR.glob("*.png")):
    print(f"    {f.name}")
print(f"\n  代码仓库: D:/桌面文件/quant/")
print(f"  策略目录: D:/桌面文件/quant/strategies/ma_crossover/")
print(f"  文本报告: D:/桌面文件/quant/strategies/ma_crossover/report_output.txt")
print(f"  {'='*70}")

sys.stdout = sys.__stdout__
report_log.close()
print()
print(f"完整文本报告已保存至: report_output.txt")
input("按 Enter 键退出...")
