"""
purify_v2.py — 提纯管道 v2：panel 模式截面 RANK + 第三维过滤。

与 v1 (scan_alpha191_purify.py) 的差异:
  1. 使用 compute_factor_matrix() panel 模式 — RANK 是真正的截面排名
  2. 加入第三维: 截面有效比 > 0.5（>50% 交易日能成功分 5 组）
  3. 保存 purify_results.csv 供 downstream 使用

2026-07 PIT 改造:
  - 股票池 = PIT 沪深 300 历史成员（790 只，data/index_membership.py），
    废除 sorted(cache)[:300] 切片（事故 universe，见 models/report.md 修正 II）
  - IC / FM / CS_eff 全部在"当日指数成员"掩码内计算
  - 因子分批计算（每批 48 个），避免 191 × 790 面板同时驻留内存 (~5GB)

用法: python purify_v2.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from signals.alpha191 import list_factors
from signals.alpha191.calculator import compute_factor_matrix
from build_pit_matrix import load_pit_panel

# ── 配置 ─────────────────────────────────────────────────
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
IC_IR_THRESHOLD = 0.05
FM_T_THRESHOLD = 2.0
CS_EFFECTIVE_THRESHOLD = 0.5  # 截面有效比
N_GROUPS = 5
BATCH_SIZE = 48  # 因子分批计算，控制内存


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def cross_sectional_effective_ratio(factor_df: pd.DataFrame, n_groups: int = 5) -> float:
    """返回因子在多大比例的交易日能成功分成 N 组。

    因子如果缺乏截面分散度（如 alpha141 几乎所有值相同），
    qcut 会因为 duplicates 失败。该指标直接捕捉这一缺陷。
    """
    valid_days = 0
    total_days = 0
    for d in factor_df.index:
        f = factor_df.loc[d].dropna()
        if len(f) < n_groups * 3:
            continue
        total_days += 1
        try:
            labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
            if labels.nunique() == n_groups:
                valid_days += 1
        except Exception:
            continue
    if total_days == 0:
        return 0.0
    return valid_days / total_days


def compute_ic_ir(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> dict:
    """计算 Rank IC + IC_IR。"""
    ic_list = []
    cd = factor_df.index.intersection(fwd_ret.index)
    cs = factor_df.columns.intersection(fwd_ret.columns)
    factor_aligned = factor_df.loc[cd, cs]
    fwd_aligned = fwd_ret.loc[cd, cs]

    for d in cd:
        f = factor_aligned.loc[d]
        r = fwd_aligned.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if pd.isna(ic):
            continue
        ic_list.append(ic)

    ic_arr = np.array(ic_list)
    if len(ic_arr) == 0:
        return {"mean": 0.0, "std": 0.0, "IR": 0.0, "t": 0.0, "n_days": 0}
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std(ddof=0)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_t = ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0.0
    return {"mean": ic_mean, "std": ic_std, "IR": ic_ir, "t": ic_t, "n_days": len(ic_arr)}


def compute_fm(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> dict:
    """Fama-MacBeth 截面回归 λ (年化) + t 值。"""
    lam_list = []
    cd = factor_df.index.intersection(fwd_ret.index)
    cs = factor_df.columns.intersection(fwd_ret.columns)
    factor_aligned = factor_df.loc[cd, cs]
    fwd_aligned = fwd_ret.loc[cd, cs]

    for d in cd:
        f = factor_aligned.loc[d]
        r = fwd_aligned.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        X = f[mask].values
        Y = r[mask].values
        var_x = np.var(X)
        if var_x < 1e-12:
            continue
        cov = np.cov(X, Y)[0, 1]
        lam = cov / var_x
        lam_list.append(lam)

    lam_arr = np.array(lam_list)
    if len(lam_arr) == 0:
        return {"λ_annual": 0.0, "t": 0.0, "n_days": 0}
    lam_mean = lam_arr.mean()
    lam_std = lam_arr.std(ddof=0)
    lam_t = lam_mean / lam_std * np.sqrt(len(lam_arr)) if lam_std > 0 else 0.0
    return {"λ_annual": lam_mean * 252, "t": lam_t, "n_days": len(lam_arr)}


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    section("数据加载（PIT 沪深 300）")
    close_matrix, _, member_daily = load_pit_panel()
    print(f"股票池: {close_matrix.shape[1]} 只历史成员，"
          f"日均成员 {(member_daily & close_matrix.notna()).sum(axis=1).mean():.1f} 只")

    # ── 构建 fwd_ret 矩阵（成员掩码在因子侧统一施加）──
    daily_ret = close_matrix.pct_change().fillna(0)
    fwd_ret = daily_ret.shift(-1)
    fwd_ret.iloc[-1] = 0
    print(f"收益矩阵: {close_matrix.shape[0]} 天 × {close_matrix.shape[1]} 只")

    # ── 分批计算 191 因子并逐因子评估 ──
    section("因子计算 + 提纯评估（panel 模式，成员掩码内 IC/FM/CS）")
    all_fids = list_factors()
    print(f"因子总数: {len(all_fids)}，每批 {BATCH_SIZE} 个")
    print(f"提纯阈值: |IC_IR| > {IC_IR_THRESHOLD}, FM |t| > {FM_T_THRESHOLD}, "
          f"截面有效比 > {CS_EFFECTIVE_THRESHOLD}\n")

    results = []
    for b0 in range(0, len(all_fids), BATCH_SIZE):
        batch = all_fids[b0:b0 + BATCH_SIZE]
        print(f"\n── 批次 {b0 // BATCH_SIZE + 1}: {batch[0]} ~ {batch[-1]} ──")
        _, factor_tensor = compute_factor_matrix(
            list(close_matrix.columns), batch,
            start=DATE_START, end=DATE_END, verbose=False,
        )

        for fid in batch:
            # PIT: 因子值仅在当日成员内参与评估
            factor_df = factor_tensor.pop(fid).where(member_daily)

            ic = compute_ic_ir(factor_df, fwd_ret)
            fm = compute_fm(factor_df, fwd_ret)
            cs_eff = cross_sectional_effective_ratio(factor_df, N_GROUPS)

            pass_ic = abs(ic["IR"]) > IC_IR_THRESHOLD
            pass_fm = abs(fm["t"]) > FM_T_THRESHOLD
            pass_cs = cs_eff > CS_EFFECTIVE_THRESHOLD
            passed = pass_ic and pass_fm and pass_cs

            results.append({
                "factor": fid,
                "IC_IR": ic["IR"],
                "IC_mean": ic["mean"],
                "IC_t": ic["t"],
                "IC_days": ic["n_days"],
                "FM_λ_annual": fm["λ_annual"],
                "FM_t": fm["t"],
                "FM_days": fm["n_days"],
                "cs_effective_ratio": cs_eff,
                "IC_pass": pass_ic,
                "FM_pass": pass_fm,
                "CS_pass": pass_cs,
                "pass": passed,
            })

            # 进度
            idx = all_fids.index(fid) + 1
            stat_parts = []
            if pass_ic: stat_parts.append("IC")
            if pass_fm: stat_parts.append("FM")
            if pass_cs: stat_parts.append("CS")
            status = f"✓({'|'.join(stat_parts)})" if passed else (
                "✗" if not (pass_ic or pass_fm or pass_cs) else
                f"({'|'.join(stat_parts)})"
            )
            print(f"  [{idx:3d}/191] {fid}: IC_IR={ic['IR']:+.4f}, "
                  f"FM_t={fm['t']:+.2f}, CS_eff={cs_eff:.3f} → {status}")

    # ── 保存结果 ──
    section("结果保存")
    df = pd.DataFrame(results)
    df = df.sort_values("IC_IR", key=abs, ascending=False)

    output_path = Path(__file__).parent / "purify_results.csv"
    df.to_csv(output_path, index=False)
    print(f"结果已保存: {output_path}")

    # ── 汇总 ──
    n_pass = df["pass"].sum()
    n_ic = df["IC_pass"].sum()
    n_fm = df["FM_pass"].sum()
    n_cs = df["CS_pass"].sum()
    n_v1 = ((df["IC_pass"]) & (df["FM_pass"])).sum()

    print(f"\n  提纯统计:")
    print(f"    |IC_IR| > {IC_IR_THRESHOLD}:  {n_ic} 个")
    print(f"    FM |t| > {FM_T_THRESHOLD}:   {n_fm} 个")
    print(f"    CS_eff > {CS_EFFECTIVE_THRESHOLD}:        {n_cs} 个")
    print(f"    v1 管道 (IC+FM):             {n_v1} 个")
    print(f"    v2 管道 (IC+FM+CS):          {n_pass} 个")

    if n_pass > 0:
        passed_df = df[df["pass"]]
        print(f"\n  ★ 提纯因子 ({n_pass} 个):")
        for _, r in passed_df.iterrows():
            print(f"    {r['factor']:10s}  IC_IR={r['IC_IR']:+.4f}  "
                  f"FM_t={r['FM_t']:+.2f}  CS_eff={r['cs_effective_ratio']:.3f}")

    # CS_eff 分布
    print(f"\n  截面有效比分布:")
    print(f"    = 0.0:    {(df['cs_effective_ratio'] == 0).sum()} 个")
    print(f"    (0, 0.5]: {(df['cs_effective_ratio'].between(0.001, 0.5)).sum()} 个")
    print(f"    (0.5, 1]: {n_cs} 个")

    # 纸老虎: IC_IR 高但 CS_eff = 0
    paper_tigers = df[(abs(df["IC_IR"]) > IC_IR_THRESHOLD) & (df["cs_effective_ratio"] == 0)]
    if len(paper_tigers):
        print(f"\n  ⚠ 纸老虎 (|IC_IR| > {IC_IR_THRESHOLD} 但 CS_eff = 0): {len(paper_tigers)} 个")
        for _, r in paper_tigers.head(10).iterrows():
            print(f"    {r['factor']:10s}  IC_IR={r['IC_IR']:+.4f}  → 截面无分散度, 无法分组")

    print(f"\n  === 提纯完成 ===")
    return df


if __name__ == "__main__":
    main()
