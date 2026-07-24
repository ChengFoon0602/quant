"""
report.py — Alpha191 因子发现统一报告。

涵盖:
  1. alpha055: DELTA×Volume 反转因子深度分析 (6 步)
  2. alpha141: TSRANK(MIN(VWAP-LOW,5),5) 深度分析 (6 步)
  3. alpha001 + alpha141 方向对齐正交化合成

用法: python report.py
"""

import sys
sys.path.insert(0, "../..")

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib, warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily, cache_summary
from data.universe import build_dynamic_universe, get_listing_info
from signals.alpha191.factors import factor_055, factor_141, factor_001
from backtest.cross_section import run_cross_section

REPORT_DIR = "figures"
os.makedirs(REPORT_DIR, exist_ok=True)

N_STOCKS = 300
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2025-12-31"
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20
DELTA_RANGE = [2, 3, 4, 5, 6, 8, 10, 15, 20]
TOP_RANGE = [0.05, 0.10, 0.15, 0.20]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════
#  数据加载（一次性）
# ══════════════════════════════════════════════════════════
section("数据加载")

cache = cache_summary()
all_symbols = sorted(cache["symbol"].tolist())

# 预先获取退市列表，排除退市股（保留正常上市股票）
_, _delist_dates = get_listing_info()
live_symbols = [s for s in all_symbols if s not in _delist_dates]
print(f"正常上市: {len(live_symbols)} 只 (排除 {len(all_symbols) - len(live_symbols)} 只退市)")
symbols = live_symbols[:N_STOCKS]

all_dfs = {}
for sym in symbols:
    df = load_daily(sym)
    if df is None or len(df) < 100:
        continue
    df = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
    if len(df) < 100:
        continue
    all_dfs[sym] = df

close_data = {sym: df["close"] for sym, df in all_dfs.items()}
amount_data = {sym: df.get("amount", pd.Series(0, index=df.index)) for sym, df in all_dfs.items()}
volume_data = {sym: df["volume"] for sym, df in all_dfs.items()}

close_matrix = pd.DataFrame(close_data).sort_index()
amount_matrix = pd.DataFrame(amount_data).sort_index()
volume_matrix = pd.DataFrame(volume_data).sort_index()

N_VALID = len(close_data)
print(f"有效股票: {N_VALID} 只")

# PIT Universe
listing_dates, delist_dates = get_listing_info()
if not listing_dates:
    for sym in close_matrix.columns:
        df = all_dfs.get(sym)
        if df is not None and len(df) > 0:
            listing_dates[sym] = df.index[0]

universe_mask = build_dynamic_universe(
    close_matrix=close_matrix, amount_matrix=amount_matrix,
    volume_matrix=volume_matrix, listing_dates=listing_dates,
    delist_dates=delist_dates, n_top=300,
)
delist_info = {
    "dates": {s: delist_dates[s] for s in delist_dates
              if s in close_matrix.columns and pd.notna(delist_dates[s])},
    "prices": {},
}
# 300 只非退市股池子小，PIT 过滤非必需；跑全量 backtest 时再用
USE_UNIVERSE = None  # universe_mask
print(f"Universe: 全部 {N_VALID} 只（非退市，不含 PIT 动态过滤）")


# ── 工具函数 ─────────────────────────────────────────────
def compute_factor_matrix_custom(fn, **kwargs):
    """从预加载数据计算因子矩阵。"""
    rows = {}
    for sym, df in all_dfs.items():
        try:
            result = fn(df, **kwargs) if kwargs else fn(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            rows[sym] = result
        except Exception:
            rows[sym] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(rows).sort_index()


def align(factor_df):
    cd = close_matrix.index.intersection(factor_df.index)
    cs = close_matrix.columns.intersection(factor_df.columns)
    return close_matrix.loc[cd, cs], factor_df.loc[cd, cs]


def compute_ic_ir(factor_aligned, close_aligned):
    daily_ret = close_aligned.pct_change().fillna(0)
    fwd_ret = daily_ret.shift(-1)
    fwd_ret.iloc[-1] = 0
    ic_list = []
    for d in factor_aligned.index:
        f = factor_aligned.loc[d]; r = fwd_ret.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10: continue
        ic = f[mask].rank().corr(r[mask].rank())
        if pd.isna(ic): continue
        ic_list.append(ic)
    arr = np.array(ic_list)
    m, s = arr.mean(), arr.std()
    ir = m / s if s > 0 else 0
    t = m / s * np.sqrt(len(arr)) if s > 0 else 0
    return {"mean": m, "std": s, "IR": ir, "t": t, "pos": (arr > 0).mean(),
            "series": pd.Series(arr, index=factor_aligned.index[:len(arr)])}


def stratified_ls(factor_aligned, close_aligned):
    daily_ret = close_aligned.pct_change().fillna(0)
    fwd_ret = daily_ret.shift(-1); fwd_ret.iloc[-1] = 0
    grp = {i: [] for i in range(5)}
    for d in factor_aligned.index:
        f = factor_aligned.loc[d].dropna()
        if len(f) < 15: continue
        labels = pd.qcut(f, 5, labels=False, duplicates="drop")
        if labels.nunique() < 5: continue
        r_next = fwd_ret.loc[d]
        for g in range(5):
            syms = labels[labels == g].index
            grp[g].append(r_next[syms].mean())
    n = min(len(grp[4]), len(grp[0]))
    ls = pd.Series([grp[4][i] - grp[0][i] for i in range(n)])
    return ls.dropna(), grp


def block_bootstrap(returns, n_boot=N_BOOTSTRAP, block=BLOCK_SIZE, seed=42):
    vals = returns.values; n = len(vals)
    rng_bt = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    if n == 0:
        return srs  # all zeros
    if n <= block:
        for b in range(n_boot):
            s = rng_bt.choice(vals, size=n, replace=True)
            srs[b] = np.sqrt(252) * s.mean() / s.std() if s.std() > 1e-12 else 0.0
        return srs
    n_blk = n // block
    for b in range(n_boot):
        starts = rng_bt.integers(0, n - block, n_blk)
        sample = np.concatenate([vals[s:s+block] for s in starts])[:n]
        srs[b] = np.sqrt(252) * sample.mean() / sample.std() if sample.std() > 1e-12 else 0.0
    return srs


def capm_decompose(strategy_net, mkt_ret):
    X = mkt_ret.values; Y = strategy_net.values
    mask = ~np.isnan(X) & ~np.isnan(Y); X, Y = X[mask], Y[mask]
    n = len(X)
    cov = np.cov(X, Y)[0, 1]; var_x = np.var(X)
    beta = cov / var_x if var_x > 0 else 0
    alpha_d = Y.mean() - beta * X.mean()
    resid = Y - (alpha_d + beta * X)
    se = np.sqrt(np.var(resid) / n / var_x * (var_x + X.mean()**2)) if var_x > 0 else np.inf
    a_t = alpha_d / se if se > 0 else 0
    r2 = 1 - np.var(resid) / np.var(Y) if np.var(Y) > 0 else 0
    return {"alpha_daily": alpha_d, "alpha_annual": alpha_d * 252, "beta": beta,
            "t": a_t, "r2": r2, "n": n}


# ══════════════════════════════════════════════════════════
#  通用单因子深度分析函数
# ══════════════════════════════════════════════════════════
def run_factor_analysis(fid, name, factor_fn, prefix,
                        param_scan_fn=None, fn_kwargs=None):
    """6 步分析管道，返回指标 dict。"""
    results = {"fid": fid, "name": name}
    fn_kwargs = fn_kwargs or {}

    section(f"{name} ({fid}) — 因子计算")
    f_mat = compute_factor_matrix_custom(factor_fn, **fn_kwargs)
    c_aligned, f_aligned = align(f_mat)
    print(f"  对齐: {len(c_aligned)} 天 × {len(c_aligned.columns)} 只")

    # Step 1: IC
    section(f"{name} — IC 分析")
    ic = compute_ic_ir(f_aligned, c_aligned)
    results["ic"] = ic
    print(f"  IC_IR={ic['IR']:.3f}, t={ic['t']:.1f}, 正IC={ic['pos']*100:.0f}%")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    s = ic["series"]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in s]
    ax.bar(range(len(s)), s.values, color=colors, alpha=0.5, width=1)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.axhline(ic["mean"], color="#2196F3", linestyle="--", linewidth=1.5,
               label=f"Mean IC={ic['mean']:.4f}")
    ax.legend(fontsize=9); ax.set_ylabel("Rank IC"); ax.grid(True, alpha=0.3)
    ax.set_title(f"{name} — Rank IC  |  IC_IR={ic['IR']:.3f}  |  t={ic['t']:.1f}")
    axes[1].plot(s.index, s.cumsum(), color="#2196F3", linewidth=1)
    axes[1].axhline(0, color="#333", linewidth=0.5)
    axes[1].set_ylabel("Cumulative IC"); axes[1].set_xlabel("Date"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/{prefix}_01_ic.png", dpi=150); plt.close()

    # Step 2: 参数扫描
    section(f"{name} — 参数扫描")
    if param_scan_fn:
        scan_df = param_scan_fn(c_aligned)
        if len(scan_df) > 0:
            best = scan_df.loc[scan_df["sharpe"].idxmax()]
            results["best_param"] = best.to_dict()
            print(f"  最优: {best.to_dict()}")
            fig, ax = plt.subplots(figsize=(8, 5))
            piv = scan_df.pivot_table(values="sharpe", index=scan_df.columns[0],
                                       columns=scan_df.columns[1] if len(scan_df.columns) > 3 else None)
            if piv.ndim == 2:
                im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn")
                ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
                ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
                for i in range(piv.shape[0]):
                    for j in range(piv.shape[1]):
                        ax.text(j, i, f"{piv.values[i][j]:.2f}", ha="center", va="center", fontsize=8)
                ax.set_xlabel("top_pct"); ax.set_ylabel("delta_days" if "delta_days" in scan_df.columns else "param")
                plt.colorbar(im, ax=ax)
            else:
                ax.plot(range(len(piv)), piv.values, "o-", color="#2196F3")
                ax.set_xticks(range(len(piv))); ax.set_xticklabels(piv.index)
                ax.set_xlabel("top_pct"); ax.set_ylabel("Sharpe")
            ax.set_title(f"{name} — 参数相图 (Sharpe)")
            plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/{prefix}_02_param.png", dpi=150); plt.close()
        else:
            results["best_param"] = {}
    else:
        results["best_param"] = {}

    # Step 3: 分层回测
    section(f"{name} — 分层回测")
    ls_ret, grp = stratified_ls(f_aligned, c_aligned)
    ls_sr = np.sqrt(252) * ls_ret.mean() / ls_ret.std() if ls_ret.std() > 0 else 0
    results["ls_sr"] = ls_sr
    results["ls_win"] = (ls_ret > 0).mean()
    # 单调性
    grp_ann = []
    for g in range(5):
        r = pd.Series(grp[g])
        eq = (1 + r).cumprod()
        n_y = len(r) / 252
        grp_ann.append((eq.iloc[-1])**(1/n_y)-1 if n_y > 0 and eq.iloc[-1] > 0 else 0)
    results["group_ann"] = grp_ann
    is_mono = all(grp_ann[i] < grp_ann[i+1] for i in range(4))
    results["monotonic"] = is_mono
    print(f"  Q1-Q5 年化: {[f'{a*100:.1f}%' for a in grp_ann]}")
    print(f"  多空 SR={ls_sr:.3f}, 单调性={'✓' if is_mono else '✗'}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    qcolors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    for g in range(5):
        eq = (1 + pd.Series(grp[g])).cumprod()
        axes[0].plot(eq.index, eq.values, color=qcolors[g], linewidth=0.9, label=f"Q{g+1}")
    axes[0].legend(fontsize=8, ncol=5); axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title(f"{name} — 分层回测 (5 分组)"); axes[0].grid(True, alpha=0.3)
    ls_eq = (1 + ls_ret).cumprod()
    axes[1].plot(ls_eq.index, ls_eq.values, color="#2196F3", linewidth=0.9)
    axes[1].axhline(1, color="#333", linewidth=0.5)
    axes[1].set_title(f"Q5/Q1 多空  |  SR={ls_sr:.3f}  |  胜率={results['ls_win']*100:.0f}%")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/{prefix}_03_stratified.png", dpi=150); plt.close()

    # Step 4: 全时段回测 + 样本外
    section(f"{name} — 样本外")
    train_idx = [d for d in c_aligned.index if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)]
    test_idx = [d for d in c_aligned.index if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)]
    top_pct = results.get("best_param", {}).get("top_pct", 0.2)
    try:
        res_train = run_cross_section(c_aligned.loc[train_idx], f_aligned.loc[train_idx],
                                       top_pct=top_pct, universe=USE_UNIVERSE, delist_info=delist_info)
        res_test = run_cross_section(c_aligned.loc[test_idx], f_aligned.loc[test_idx],
                                      top_pct=top_pct, universe=USE_UNIVERSE, delist_info=delist_info)
        results["train_sr"] = res_train["sharpe"]
        results["test_sr"] = res_test["sharpe"]
        results["train_ann"] = res_train["ann_return"]
        results["test_ann"] = res_test["ann_return"]
        results["overfit_ratio"] = res_test["sharpe"] / res_train["sharpe"] if res_train["sharpe"] > 0 else -999
        print(f"  训练SR={res_train['sharpe']:.3f}, 测试SR={res_test['sharpe']:.3f}, "
              f"过拟合比={results['overfit_ratio']:.2f}")
    except Exception as e:
        print(f"  [WARN] 样本外回测失败: {e}")
        results["train_sr"] = np.nan; results["test_sr"] = np.nan

    # 全时段回测
    res_full = run_cross_section(c_aligned, f_aligned, top_pct=top_pct,
                                  universe=USE_UNIVERSE, delist_info=delist_info)
    results["full_sr"] = res_full["sharpe"]
    results["full_ann"] = res_full["ann_return"]
    results["full_dd"] = res_full["max_drawdown"]
    results["full_ir"] = res_full["information_ratio"]
    results["strategy_net"] = res_full["strategy_net"]
    results["equity"] = res_full["equity"]
    results["benchmark"] = res_full["benchmark"]
    results["close_aligned"] = c_aligned
    results["factor_aligned"] = f_aligned

    # Step 5: CAPM
    section(f"{name} — CAPM")
    mkt_ret = c_aligned.pct_change().fillna(0).mean(axis=1)
    mkt_ret = mkt_ret.reindex(res_full["strategy_net"].index).fillna(0)
    capm_r = capm_decompose(res_full["strategy_net"].dropna(), mkt_ret)
    results["capm"] = capm_r
    print(f"  α={capm_r['alpha_annual']*100:.1f}%/yr, β={capm_r['beta']:.2f}, "
          f"t={capm_r['t']:.1f}, R²={capm_r['r2']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X, Y = mkt_ret.values, res_full["strategy_net"].values
    mask = ~np.isnan(X) & ~np.isnan(Y); X, Y = X[mask], Y[mask]
    axes[0].scatter(X * 100, Y * 100, s=3, alpha=0.3, color="#2196F3")
    xr = np.linspace(X.min(), X.max(), 100)
    axes[0].plot(xr * 100, (capm_r["alpha_daily"] + capm_r["beta"] * xr) * 100,
                 color="#F44336", linewidth=1.5,
                 label=f"α={capm_r['alpha_annual']*100:.1f}%/yr, β={capm_r['beta']:.2f}")
    axes[0].axhline(0, color="#333", linewidth=0.5); axes[0].axvline(0, color="#333", linewidth=0.5)
    axes[0].set_xlabel("Market Return (%)"); axes[0].set_ylabel("Strategy Return (%)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{name} — CAPM 回归")
    eq_s = res_full["equity"]; bm_s = res_full["benchmark"]
    axes[1].plot(eq_s.index, eq_s.values, color="#4CAF50", linewidth=0.8, label=f"{name}")
    axes[1].plot(bm_s.index, bm_s.values, color="#999", linewidth=0.6, label="等权基准")
    axes[1].axvline(pd.Timestamp(TEST_START), color="#F44336", linestyle="--", linewidth=1)
    axes[1].legend(fontsize=8); axes[1].set_ylabel("Equity"); axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"{name} — 权益曲线  |  SR={res_full['sharpe']:.2f}")
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/{prefix}_04_equity.png", dpi=150); plt.close()

    # Step 6: Bootstrap
    section(f"{name} — Bootstrap")
    ls_vals = ls_ret.values
    bs = block_bootstrap(ls_ret)
    real_sr = ls_sr
    p_val = (bs >= real_sr).mean()
    p_pos = (bs > 0).mean()
    ci = np.percentile(bs, [2.5, 97.5])
    results["bootstrap"] = {"sr": real_sr, "p": p_val, "p_pos": p_pos, "ci_lo": ci[0], "ci_hi": ci[1]}
    print(f"  p={p_val:.4f}, P(SR>0)={p_pos*100:.0f}%, 95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(bs, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
    axes[0].axvline(real_sr, color="#F44336", linewidth=2, linestyle="--", label=f"SR={real_sr:.3f}")
    axes[0].axvline(0, color="#999", linewidth=1, linestyle=":")
    axes[0].axvline(ci[0], color="#4CAF50", linewidth=1, linestyle="--")
    axes[0].axvline(ci[1], color="#4CAF50", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Sharpe"); axes[0].set_ylabel("Density")
    axes[0].set_title(f"{name} — Block Bootstrap (p={p_val:.3f})")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.sort(bs), np.arange(1, N_BOOTSTRAP+1)/N_BOOTSTRAP, color="#2196F3", linewidth=1.5)
    axes[1].axvline(real_sr, color="#F44336", linewidth=1.5, linestyle="--")
    axes[1].axhline(0.05, color="#999", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Sharpe"); axes[1].set_ylabel("CDF")
    axes[1].set_title(f"CDF  |  P(SR>0)={p_pos*100:.0f}%")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/{prefix}_05_bootstrap.png", dpi=150); plt.close()

    return results


# ══════════════════════════════════════════════════════════
#  分析 1: alpha055
# ══════════════════════════════════════════════════════════
def scan_055(c_aligned):
    rows = []
    for dd in DELTA_RANGE:
        f = compute_factor_matrix_custom(factor_055, delta_days=dd)
        _, f_a = align(f)
        for tp in TOP_RANGE:
            try:
                r = run_cross_section(c_aligned, f_a, top_pct=tp,
                                       universe=USE_UNIVERSE, delist_info=delist_info)
                rows.append({"delta_days": dd, "top_pct": tp, "sharpe": r["sharpe"],
                             "ann_return": r["ann_return"]})
            except Exception:
                rows.append({"delta_days": dd, "top_pct": tp, "sharpe": np.nan, "ann_return": np.nan})
    return pd.DataFrame(rows)


a055 = run_factor_analysis("alpha055", "Alpha 055 (DELTA×Volume)", factor_055,
                            "a055", param_scan_fn=scan_055)

# ══════════════════════════════════════════════════════════
#  分析 2: alpha141
# ══════════════════════════════════════════════════════════
def scan_141(c_aligned):
    f_mat = compute_factor_matrix_custom(factor_141)
    _, f_a = align(f_mat)
    rows = []
    for tp in TOP_RANGE:
        try:
            r = run_cross_section(c_aligned, f_a, top_pct=tp,
                                   universe=USE_UNIVERSE, delist_info=delist_info)
            rows.append({"top_pct": tp, "sharpe": r["sharpe"], "ann_return": r["ann_return"]})
        except Exception:
            rows.append({"top_pct": tp, "sharpe": np.nan, "ann_return": np.nan})
    return pd.DataFrame(rows)


a141 = run_factor_analysis("alpha141", "Alpha 141 (TSRANK MIN(VWAP-LOW))", factor_141,
                            "a141", param_scan_fn=scan_141)

# ══════════════════════════════════════════════════════════
#  分析 3: alpha001 + alpha141 方向对齐正交化合成
# ══════════════════════════════════════════════════════════
section("合成: alpha001 + alpha141 方向对齐正交化")

print("  计算因子...")
f001 = compute_factor_matrix_custom(factor_001)
f141 = compute_factor_matrix_custom(factor_141)
c001, f001_a = align(f001)
c141, f141_a = align(f141)

# 对齐三者
cd = c001.index.intersection(c141.index)
cs = c001.columns.intersection(c141.columns)
c_synth = c001.loc[cd, cs]
f1 = f001_a.loc[cd, cs]
f2 = f141_a.loc[cd, cs]

# 方向对齐：用多空 SR
print("  方向判定...")
ls1, _ = stratified_ls(f1, c_synth)
ls2, _ = stratified_ls(f2, c_synth)
sr1 = np.sqrt(252) * ls1.mean() / ls1.std() if ls1.std() > 0 else 0
sr2 = np.sqrt(252) * ls2.mean() / ls2.std() if ls2.std() > 0 else 0

flip1 = ls1.mean() < 0
flip2 = ls2.mean() < 0
if flip1: f1 = -f1
if flip2: f2 = -f2
print(f"  alpha001: LS_SR={'%.3f'%sr1} {'→ FLIP' if flip1 else '→ 保持'}")
print(f"  alpha141: LS_SR={'%.3f'%sr2} {'→ FLIP' if flip2 else '→ 保持'}")

# Gram-Schmidt: regress alpha001 (lower IC_IR) on alpha141 (higher IC_IR) first
# Base = alpha141 (IC_IR=0.79), then regress alpha001 on base
ic141 = compute_ic_ir(f2, c_synth)
ic001 = compute_ic_ir(f1, c_synth)
if abs(ic141["IR"]) >= abs(ic001["IR"]):
    base_f, base_name = f2, "alpha141"
    other_f, other_name = f1, "alpha001"
else:
    base_f, base_name = f1, "alpha001"
    other_f, other_name = f2, "alpha141"
print(f"  基底: {base_name} (|IC_IR|={abs(ic141['IR'] if base_name=='alpha141' else ic001['IR']):.3f})")

# 正交化: regress other on base, take residuals
residual_rows = {}
for d in cd:
    y = other_f.loc[d]; x = base_f.loc[d]
    mask = y.notna() & x.notna()
    if mask.sum() < 10:
        residual_rows[d] = pd.Series(np.nan, index=cs)
        continue
    Yv = y[mask].values; Xv = x[mask].values
    Xm = np.column_stack([np.ones_like(Xv), Xv])
    try:
        beta = np.linalg.lstsq(Xm, Yv, rcond=None)[0]
        resid = Yv - Xm @ beta
    except np.linalg.LinAlgError:
        residual_rows[d] = pd.Series(np.nan, index=cs)
        continue
    rs = pd.Series(np.nan, index=cs)
    rs[mask] = resid
    residual_rows[d] = rs
ortho_other = pd.DataFrame(residual_rows).T.sort_index()

# 等权合成
combo = (base_f + ortho_other) / 2
combo = combo.reindex(index=cd, columns=cs)

print("\n── 合成 vs 单因子 IC_IR ──")
ic_base = compute_ic_ir(base_f, c_synth)
ic_ortho = compute_ic_ir(ortho_other, c_synth)
ic_combo = compute_ic_ir(combo, c_synth)
for label, ic_r in [("alpha001", ic001), ("alpha141", ic141),
                      (f"{other_name}(正交残差)", ic_ortho),
                      ("合成(等权)", ic_combo)]:
    print(f"  {label:<20s}: IC_IR={ic_r['IR']:+.4f}, t={ic_r['t']:+.1f}")

# 分层多空比较
print("\n── 分层多空比较 ──")
ls_base, grp_base = stratified_ls(base_f, c_synth)
ls_other, grp_other = stratified_ls(f1 if base_name == "alpha141" else f2, c_synth)
ls_combo, grp_combo = stratified_ls(combo, c_synth)

sr_base = np.sqrt(252) * ls_base.mean() / ls_base.std() if ls_base.std() > 0 else 0
sr_other = np.sqrt(252) * ls_other.mean() / ls_other.std() if ls_other.std() > 0 else 0
sr_combo = np.sqrt(252) * ls_combo.mean() / ls_combo.std() if ls_combo.std() > 0 else 0

bs_base = block_bootstrap(ls_base)
bs_other = block_bootstrap(ls_other)
bs_combo = block_bootstrap(ls_combo)

for label, bs, sr in [("alpha001", bs_other if base_name == "alpha141" else bs_base, sr_other if base_name == "alpha141" else sr_base),
                        ("alpha141", bs_base if base_name == "alpha141" else bs_other, sr_base if base_name == "alpha141" else sr_other),
                        ("合成(等权)", bs_combo, sr_combo)]:
    p = (bs >= sr).mean(); pp = (bs > 0).mean()
    print(f"  {label:<12s}: LS_SR={sr:+.3f}, p={p:.4f}, P(SR>0)={pp*100:.0f}%")

# 图: IC_IR 对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
labels = ["alpha001", "alpha141", "合成(等权)"]
irs = [ic001["IR"], ic141["IR"], ic_combo["IR"]]
colors_bar = ["#2196F3", "#4CAF50", "#FF9800"]
ax.bar(labels, irs, color=colors_bar, alpha=0.8)
ax.axhline(0, color="#333", linewidth=0.5)
ax.set_ylabel("IC_IR"); ax.set_title("IC_IR 对比"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
srs = [sr_other if base_name == "alpha141" else sr_base,
       sr_base if base_name == "alpha141" else sr_other, sr_combo]
ax.bar(labels, srs, color=colors_bar, alpha=0.8)
ax.axhline(0, color="#333", linewidth=0.5)
ax.set_ylabel("Long-Short SR"); ax.set_title("多空 SR 对比"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[2]
x = np.linspace(-2, 3, 200)
ax.plot(x, (1/(bs_base.std()*np.sqrt(2*np.pi)))*np.exp(-(x-bs_base.mean())**2/(2*bs_base.std()**2)),
        color="#2196F3", alpha=0.7, label=f"alpha001 (p={((bs_other if base_name=='alpha141' else bs_base)>=sr_other if base_name=='alpha141' else sr_base).mean():.3f})")
ax.plot(x, (1/(bs_combo.std()*np.sqrt(2*np.pi)))*np.exp(-(x-bs_combo.mean())**2/(2*bs_combo.std()**2)),
        color="#FF9800", alpha=0.7, label=f"合成 (p={(bs_combo>=sr_combo).mean():.3f})")
ax.axvline(0, color="#999", linewidth=0.5, linestyle=":")
ax.set_xlabel("Sharpe"); ax.set_ylabel("Density")
ax.set_title("Bootstrap SR 分布对比"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle("alpha001 + alpha141 方向对齐正交化合成", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/synth_01_compare.png", dpi=150); plt.close()

# 图: 分层回测对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
qcolors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
for idx, (label, grp) in enumerate([("alpha001", grp_other if base_name == "alpha141" else grp_base),
                                      ("alpha141", grp_base if base_name == "alpha141" else grp_other),
                                      ("合成(等权)", grp_combo)]):
    ax = axes[idx]
    for g in range(5):
        eq = (1 + pd.Series(grp[g])).cumprod()
        ax.plot(eq.index, eq.values, color=qcolors[g], linewidth=0.7, label=f"Q{g+1}")
    ax.legend(fontsize=7, ncol=5, loc="upper left"); ax.set_title(label); ax.grid(True, alpha=0.3)
    ax.set_ylabel("Cumulative Return")
plt.suptitle("分层回测对比 — 5 分组等权", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/synth_02_stratified.png", dpi=150); plt.close()

# 存储合成结果
synth_results = {
    "base_name": base_name, "other_name": other_name, "flipped": {"alpha001": flip1, "alpha141": flip2},
    "ic_base": ic_base, "ic_ortho": ic_ortho, "ic_combo": ic_combo,
    "sr_base": sr_base, "sr_other": sr_other, "sr_combo": sr_combo,
    "bs_base": bs_base, "bs_other": bs_other, "bs_combo": bs_combo,
    "p_base": (bs_base >= sr_base).mean(), "p_other": (bs_other >= sr_other).mean(),
    "p_combo": (bs_combo >= sr_combo).mean(),
}

# ══════════════════════════════════════════════════════════
#  汇总输出
# ══════════════════════════════════════════════════════════
section("汇总")

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │              Alpha 191 因子发现 — 统一报告汇总                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  指标                  alpha055      alpha141     合成(001+141)   │
  ├─────────────────────────────────────────────────────────────────┤
""")

def fmt_v(v, pct=False):
    if pd.isna(v) or v is None: return "     N/A"
    return f"{v*100:>6.1f}%" if pct else f"{v:>8.3f}"

# IC_IR row
print(f"  │  IC_IR                {fmt_v(a055['ic']['IR'])}     {fmt_v(a141['ic']['IR'])}     {fmt_v(synth_results['ic_combo']['IR'])}       │")
print(f"  │  IC t                 {fmt_v(a055['ic']['t'])}    {fmt_v(a141['ic']['t'])}    {fmt_v(synth_results['ic_combo']['t'])}       │")
print(f"  │  多空 LS_SR            {fmt_v(a055['ls_sr'])}     {fmt_v(a141['ls_sr'])}     {fmt_v(synth_results['sr_combo'])}       │")
print(f"  │  分层单调性            {'✓' if a055['monotonic'] else '✗':>8s}     {'✓' if a141['monotonic'] else '✗':>8s}     {'—':>8s}       │")
print(f"  │  CAPM α (年化)        {fmt_v(a055['capm']['alpha_annual'], pct=True)}   {fmt_v(a141['capm']['alpha_annual'], pct=True)}   {'—':>8s}       │")
print(f"  │  Bootstrap p          {fmt_v(a055['bootstrap']['p'])}     {fmt_v(a141['bootstrap']['p'])}     {fmt_v(synth_results['p_combo'])}       │")
print(f"  │  P(SR>0)              {fmt_v(a055['bootstrap']['p_pos']*100, pct=False)[:6]+'%':>8s}   {fmt_v(a141['bootstrap']['p_pos']*100, pct=False)[:6]+'%':>8s}   {fmt_v((synth_results['bs_combo']>0).mean()*100, pct=False)[:6]+'%':>8s}       │")
print(f"  │  训练 SR               {fmt_v(a055.get('train_sr', np.nan))}     {fmt_v(a141.get('train_sr', np.nan))}     {'—':>8s}       │")
print(f"  │  测试 SR               {fmt_v(a055.get('test_sr', np.nan))}     {fmt_v(a141.get('test_sr', np.nan))}     {'—':>8s}       │")
print(f"  │  全时段 SR             {fmt_v(a055['full_sr'])}     {fmt_v(a141['full_sr'])}     {'—':>8s}       │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │                                                                  │")

# Final verdict
def verdict(r, name):
    ic_ok = abs(r['ic']['IR']) > 0.05 and abs(r['ic']['t']) > 2
    ls_ok = abs(r['ls_sr']) > 0.3
    bs_ok = r['bootstrap']['p'] < 0.10
    mono_ok = r['monotonic']
    score = ic_ok + ls_ok + bs_ok + mono_ok
    if score >= 3: return f"{name}: 可用 (通过 {score}/4)"
    elif score >= 2: return f"{name}: 弱信号 (通过 {score}/4)"
    else: return f"{name}: 无效 (通过 {score}/4)"

print(f"  │  {verdict(a055, 'alpha055'):<62s} │")
print(f"  │  {verdict(a141, 'alpha141'):<62s} │")
print(f"  │  合成: IC_IR 提升 {synth_results['ic_combo']['IR'] - max(ic001['IR'], ic141['IR']):+.4f} vs 最佳单因子              │")
print(f"  │       LS_SR {'提升' if synth_results['sr_combo'] > max(sr_base, sr_other) else '下降'} {synth_results['sr_combo'] - max(sr_base, sr_other):+.3f} vs 最佳单因子              │")
print(f"  │       Bootstrap p {'降低' if synth_results['p_combo'] < min(synth_results['p_base'], synth_results['p_other']) else '上升'} {synth_results['p_combo'] - min(synth_results['p_base'], synth_results['p_other']):+.4f} vs 最佳单因子     │")
print(f"  │                                                                  │")
print(f"  └─────────────────────────────────────────────────────────────────┘""")

print(f"\n图表输出: {os.path.abspath(REPORT_DIR)}/")
for f in sorted(os.listdir(REPORT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")

print(f"\n=== 完成 ===")
