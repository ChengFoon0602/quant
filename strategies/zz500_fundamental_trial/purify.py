"""
purify.py — 月末截面因子提纯（方向2 核心方法论模块）。

与价量链路（zz500_pit_trial 逐日 IC）的本质差异：
  季频因子经日频前向填充后，相邻交易日因子值相同 → 逐日 IC 序列强自相关
  → IC_IR 虚高、有效样本虚增。因此提纯评估必须在月末截面降采样
  （每月最后一个交易日一个独立观测，2010-2025 ≈ 190 个月截面）。

评估函数（向量化，数学与 zz500_pit_trial.compute_ic_ir_vec 等价，仅日期采样不同）:
  - compute_ic_ir_monthly / compute_fm_monthly: 月末截面 Rank IC / Fama-MacBeth
  - compute_ic_ir_daily: 日频对照（同一 fwd=21 标签），实证 IC 自相关虚高
  - cross_sectional_effective_ratio_monthly: 月末 qcut 分散度

提纯判决（四维 AND）:
  |IC_IR|>0.15（月末 ~190 截面 ⟺ t>2）AND |IC_t|>2.0 AND |FM_t|>2.0 AND CS_eff>0.5
  报告附 Bonferroni 列（|t|>3.09，25 因子从严口径）。

用法:
    python strategies/zz500_fundamental_trial/purify.py
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, FIGURES_DIR,
    INDEX, FWD_DAYS, MIN_N,
    IC_IR_THRESHOLD, IC_T_THRESHOLD, FM_T_THRESHOLD,
    CS_EFFECTIVE_THRESHOLD, N_GROUPS,
    RANK_IC_CORR_THRESHOLD, MAX_POOL_SIZE, MUST_KEEP, BONFERRONI_T,
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from build_pit_matrix import load_pit_panel
from select_features import compute_rank_ic_corr_matrix, remove_redundant
from purify_v2 import cross_sectional_effective_ratio
from signals.fundamental.factors import FACTOR_SPECS, compute_factor_tensor


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


# ── 月末采样 ──────────────────────────────────────────────────

def month_end_dates(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个交易日的日期序列。"""
    s = pd.Series(daily_index, index=daily_index)
    grouped = s.groupby([s.dt.year, s.dt.month]).last()
    return pd.DatetimeIndex(grouped.values)


def fwd_return(close_matrix: pd.DataFrame, fwd_days: int = FWD_DAYS) -> pd.DataFrame:
    """21 日前向收益 = close(t+22)/close(t+1)-1（与 labels._compute_fwd_return 一致）。"""
    entry = close_matrix.shift(-1)
    exit_ = close_matrix.shift(-(fwd_days + 1))
    return exit_ / entry - 1


# ── 向量化月末 IC / FM ────────────────────────────────────────

def _ic_ir_vec(f: pd.DataFrame, r: pd.DataFrame, min_n: int = MIN_N) -> dict:
    """逐截面 Spearman Rank IC + IC_IR（采样后矩阵直接复用价量链路的数学）。"""
    cd = f.index.intersection(r.index)
    cs = f.columns.intersection(r.columns)
    f = f.loc[cd, cs]; r = r.loc[cd, cs]
    joint = f.notna() & r.notna()
    f = f.where(joint); r = r.where(joint)
    fr = f.rank(axis=1); rr = r.rank(axis=1)
    valid = joint.sum(axis=1) >= min_n
    fr, rr = fr[valid], rr[valid]
    if len(fr) == 0:
        return {"mean": 0.0, "std": 0.0, "IR": 0.0, "t": 0.0, "n": 0}
    fm = fr.mean(axis=1); rm = rr.mean(axis=1)
    fc = fr.sub(fm, axis=0); rc = rr.sub(rm, axis=0)
    cov = (fc * rc).sum(axis=1)
    denom = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic = (cov / denom.replace(0, np.nan)).dropna()
    if len(ic) == 0:
        return {"mean": 0.0, "std": 0.0, "IR": 0.0, "t": 0.0, "n": 0}
    m, s = ic.mean(), ic.std(ddof=0)
    ir = m / s if s > 0 else 0.0
    return {"mean": m, "std": s, "IR": ir,
            "t": ir * np.sqrt(len(ic)) if s > 0 else 0.0, "n": len(ic)}


def _fm_vec(f: pd.DataFrame, r: pd.DataFrame, min_n: int = MIN_N) -> dict:
    """逐截面 Fama-MacBeth λ（单变量回归，var=ddof=0, cov=ddof=1，与价量链路一致）。"""
    cd = f.index.intersection(r.index)
    cs = f.columns.intersection(r.columns)
    f = f.loc[cd, cs]; r = r.loc[cd, cs]
    joint = f.notna() & r.notna()
    f = f.where(joint); r = r.where(joint)
    n_day = joint.sum(axis=1)
    valid = n_day >= min_n
    f, r, n_day = f[valid], r[valid], n_day[valid]
    if len(f) == 0:
        return {"λ_annual": 0.0, "t": 0.0, "n": 0}
    fm = f.mean(axis=1); rm = r.mean(axis=1)
    fc = f.sub(fm, axis=0); rc = r.sub(rm, axis=0)
    var_x = (fc ** 2).sum(axis=1) / n_day
    cov_xy = (fc * rc).sum(axis=1) / (n_day - 1)
    lam = (cov_xy / var_x.where(var_x >= 1e-12)).dropna()
    if len(lam) == 0:
        return {"λ_annual": 0.0, "t": 0.0, "n": 0}
    m, s = lam.mean(), lam.std(ddof=0)
    return {"λ_annual": m * 252, "t": m / s * np.sqrt(len(lam)) if s > 0 else 0.0, "n": len(lam)}


def compute_ic_ir_monthly(factor_df: pd.DataFrame, fwd: pd.DataFrame,
                          med: pd.DatetimeIndex) -> dict:
    """月末截面 Rank IC + IC_IR。"""
    return _ic_ir_vec(factor_df.loc[med], fwd.loc[med])


def compute_fm_monthly(factor_df: pd.DataFrame, fwd: pd.DataFrame,
                       med: pd.DatetimeIndex) -> dict:
    """月末截面 Fama-MacBeth。"""
    return _fm_vec(factor_df.loc[med], fwd.loc[med])


def compute_ic_ir_daily(factor_df: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """日频对照（IC 自相关虚高实证）：逐日采样同一 fwd=21 标签。"""
    return _ic_ir_vec(factor_df, fwd)


# ── 提纯主流程 ────────────────────────────────────────────────

def purify_and_select(close_matrix, volume_matrix, member_daily, fields: list[str] | None = None):
    """月末四维提纯 → Rank IC 冗余剔除 → 返回
    (final_pool, purify_df, corr_mat, factor_tensor, keep_tensor)。

    fields: 要评估的因子名列表；None = 全部 25 个。C 方案（数据部分到位）
    可用 profit 子集冒烟，缺缓存因子不拉取不报错。
    """
    if fields is None:
        fields = list(FACTOR_SPECS.keys())
    section(f"阶段 1/2：{len(fields)} 因子月末截面评估（fwd=21 交易日）")
    fwd = fwd_return(close_matrix)
    med = month_end_dates(close_matrix.index)
    print(f"月末截面数: {len(med)} | 因子数: {len(fields)}")
    print(f"阈值: |IC_IR|>{IC_IR_THRESHOLD} |IC_t|>{IC_T_THRESHOLD} "
          f"|FM_t|>{FM_T_THRESHOLD} CS_eff>{CS_EFFECTIVE_THRESHOLD} "
          f"(Bonferroni |t|>{BONFERRONI_T:.2f})")

    factor_tensor = compute_factor_tensor(close_matrix, fields=fields)

    results = []
    keep_tensor: dict[str, pd.DataFrame] = {}
    for field in fields:
        spec = FACTOR_SPECS[field]
        fdf = factor_tensor[field].where(member_daily)   # PIT 成员掩码
        ic_m = compute_ic_ir_monthly(fdf, fwd, med)
        fm_m = compute_fm_monthly(fdf, fwd, med)
        ic_d = compute_ic_ir_daily(fdf, fwd)
        p_ic = abs(ic_m["IR"]) > IC_IR_THRESHOLD and abs(ic_m["t"]) > IC_T_THRESHOLD
        p_fm = abs(fm_m["t"]) > FM_T_THRESHOLD
        if p_ic and p_fm:   # CS 仅对已过 IC+FM 的算（AND 门，未过不必算）
            cs = cross_sectional_effective_ratio(fdf.loc[med], N_GROUPS)
        else:
            cs = np.nan
        p_cs = bool(cs > CS_EFFECTIVE_THRESHOLD) if not np.isnan(cs) else False
        passed = p_ic and p_fm and p_cs
        # 先验方向一致性：实测 IC_mean 符号 vs 先验 direction
        sign_ok = (spec["direction"] == "+" and ic_m["mean"] > 0) or \
                  (spec["direction"] == "-" and ic_m["mean"] < 0)
        results.append({
            "factor": field, "category": spec["category"], "prior_dir": spec["direction"],
            "IC_mean": ic_m["mean"], "IC_IR": ic_m["IR"], "IC_t": ic_m["t"],
            "IC_n_months": ic_m["n"], "IC_IR_daily": ic_d["IR"], "IC_t_daily": ic_d["t"],
            "FM_λ_annual": fm_m["λ_annual"], "FM_t": fm_m["t"],
            "cs_effective_ratio": cs, "sign_ok": sign_ok,
            "IC_pass": p_ic, "FM_pass": p_fm, "CS_pass": p_cs, "pass": passed,
        })
        if passed or field in MUST_KEEP:
            keep_tensor[field] = fdf

    purify_df = pd.DataFrame(results).sort_values("IC_IR", key=abs, ascending=False)
    purify_df.to_csv(THIS_DIR / "purify_results_monthly.csv", index=False)
    n_pass = int(purify_df["pass"].sum())
    print(f"  四维通过: {n_pass} 个")
    for _, r in purify_df.iterrows():
        mark = "✓" if r["pass"] else ("✗" if not np.isnan(r["cs_effective_ratio"]) else "·")
        print(f"    {r['factor']:<15} IC_IR={r['IC_IR']:+.3f} (日频 {r['IC_IR_daily']:+.3f}) "
              f"IC_t={r['IC_t']:+.2f} FM_t={r['FM_t']:+.2f} CS={r['cs_effective_ratio']:.2f} "
              f"方向{'一致' if r['sign_ok'] else '反'} {mark}")

    # 实证显著性通胀：日频 t 值系统性高于月末（IC 自相关 → 有效样本虚增）
    t_ratio = (purify_df["IC_t_daily"].abs() / purify_df["IC_t"].abs().replace(0, np.nan)).median()
    ac_ratio = (purify_df["IC_IR_daily"].abs() / purify_df["IC_IR"].abs().replace(0, np.nan)).median()
    print(f"\n  日频/月末 |IC_t| 中位比: {t_ratio:.1f}x | |IC_IR| 比: {ac_ratio:.1f}x → "
          f"{'实证日频 t 值通胀（月末采样必要）' if t_ratio > 1.5 else '通胀不显著'}")

    section("阶段 2/2：Rank IC 冗余剔除（|corr|>0.8 贪心）")
    ic_ir_map = {r["factor"]: r["IC_IR"] for _, r in purify_df.iterrows()}
    corr_mat = compute_rank_ic_corr_matrix(keep_tensor, fwd.loc[med])
    corr_mat.to_csv(THIS_DIR / "rank_ic_corr_monthly.csv")
    selected, redundant = remove_redundant(corr_mat, ic_ir_map,
                                           RANK_IC_CORR_THRESHOLD, MUST_KEEP)
    print(f"  候选 {len(keep_tensor)} → 冗余剔除后 {len(selected)}（剔 {len(redundant)}）")

    sel_icir = sorted(selected, key=lambda f: abs(ic_ir_map.get(f, 0)), reverse=True)
    final_pool = sel_icir[:MAX_POOL_SIZE]
    for mk in MUST_KEEP:
        if mk not in final_pool:
            final_pool.append(mk)
    print(f"  最终因子池 ({len(final_pool)}): {final_pool}")
    for i, f in enumerate(final_pool, 1):
        r = purify_df[purify_df["factor"] == f].iloc[0]
        print(f"    {i:2d}. {f:<15} IC_IR={r['IC_IR']:+.4f} FM_t={r['FM_t']:+.2f} "
              f"CS={r['cs_effective_ratio']:.3f}")

    return final_pool, purify_df, corr_mat, factor_tensor, keep_tensor


def plot_purify(purify_df):
    """图1: (a) 月末 IC_IR vs 日频 IC_IR 散点（自相关实证）(b) 因子方向一致性。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df = purify_df.sort_values("IC_IR", key=abs, ascending=False)
    ax = axes[0]
    x = np.arange(len(df))
    ax.scatter(x, df["IC_IR"].values, label="月末 IC_IR", marker="o", color="#c44e52")
    ax.scatter(x, df["IC_IR_daily"].values, label="日频 IC_IR（对照）", marker="x", color="#4c72b0")
    ax.axhline(IC_IR_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-IC_IR_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["factor"], rotation=90, fontsize=6)
    ax.set_ylabel("IC_IR")
    ax.set_title("月末 vs 日频 IC_IR（日频虚高 = IC 自相关实证）")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    cols = ["IC_t", "FM_t"]
    width = 0.38
    ax.bar(x - width / 2, df["IC_t"].values, width, label="IC t 值", color="#4c72b0")
    ax.bar(x + width / 2, df["FM_t"].values, width, label="FM t 值", color="#dd8452")
    ax.axhline(2.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-2.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(BONFERRONI_T, color="black", linestyle=":", linewidth=0.8)
    ax.axhline(-BONFERRONI_T, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["factor"], rotation=90, fontsize=6)
    ax.set_ylabel("t 值")
    ax.set_title("月末 IC / FM t 值（点线 = Bonferroni 从严口径）")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "01_purify_monthly.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {path}")


def main():
    section("方向2：PIT 基本面 × 中证500 — 月末因子提纯")
    print("[1] 加载 PIT 面板...")
    close, volume, member = load_pit_panel(INDEX)
    print(f"  close {close.shape} | member 月末 {member.resample('ME').last().sum().max():.0f} 只")

    final_pool, purify_df, corr_mat, factor_tensor, keep_tensor = purify_and_select(
        close, volume, member)
    plot_purify(purify_df)

    section("提纯完成")
    print(f"最终池 ({len(final_pool)}): {final_pool}")
    n_pass = int(purify_df["pass"].sum())
    print(f"四维通过 {n_pass}/{len(purify_df)} 个 | 详见 purify_results_monthly.csv")


if __name__ == "__main__":
    main()
