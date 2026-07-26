"""
select_features.py — 冗余剔除 + 最终特征矩阵构建。

消费 purify_results.csv，执行:
  1. 剔除纸老虎 (CS_eff == 0)
  2. 计算 Rank IC 相关性矩阵，剔除冗余 (corr > 0.8)
  3. 硬性保留 alpha001 (基准) + alpha055 (避险)
  4. 构建最终特征矩阵 X + 目标 y (fwd_return)（PIT universe，复用 build_pit_matrix）
  5. 加入市场状态特征 (波动率, 换手率)

用法: python select_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from signals.alpha191.calculator import compute_factor_matrix

# ── 配置 ─────────────────────────────────────────────────
DATE_START, DATE_END = "2010-01-01", "2025-12-31"
RANK_IC_CORR_THRESHOLD = 0.8
IC_IR_THRESHOLD = 0.05
CS_EFFECTIVE_THRESHOLD = 0.5
MUST_KEEP = {"alpha001", "alpha055"}
MAX_POOL_SIZE = 15
FIGURES_DIR = Path(__file__).parent / "figures"


def compute_rank_ic_series(factor_df, fwd_ret):
    """返回因子逐日 Rank IC 序列。"""
    ic_list = []
    dates = []
    cd = factor_df.index.intersection(fwd_ret.index)
    cs = factor_df.columns.intersection(fwd_ret.columns)
    fa = factor_df.loc[cd, cs]
    fr = fwd_ret.loc[cd, cs]
    for d in cd:
        f = fa.loc[d]
        r = fr.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 10:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if pd.isna(ic):
            continue
        ic_list.append(ic)
        dates.append(d)
    return pd.Series(ic_list, index=dates, name="rank_ic")


def compute_rank_ic_corr_matrix(factor_dict, fwd_ret):
    """计算因子间 Rank IC 序列的 Pearson 相关性矩阵。"""
    ic_series = {}
    for fid, factor_df in factor_dict.items():
        ic_series[fid] = compute_rank_ic_series(factor_df, fwd_ret)
    # 对齐日期
    all_ics = pd.DataFrame(ic_series).dropna()
    return all_ics.corr()


def remove_redundant(corr_matrix, ic_ir_map, threshold=0.8, must_keep=None):
    """基于 Rank IC 相关性剔除冗余因子。

    贪心算法: 按 |IC_IR| 降序排列，逐个加入，若新因子与已选中因子
    的 Rank IC 相关性 > threshold 则跳过。must_keep 中的因子强制保留。
    """
    if must_keep is None:
        must_keep = set()
    # 按 |IC_IR| 降序，must_keep 排在前面
    fids_sorted = sorted(corr_matrix.columns, key=lambda f: abs(ic_ir_map.get(f, 0)), reverse=True)
    # must_keep 移到最前面
    for f in reversed(list(must_keep)):
        if f in fids_sorted:
            fids_sorted.remove(f)
            fids_sorted.insert(0, f)
    selected = []
    redundant_pairs = []  # (kept, removed)
    for fid in fids_sorted:
        conflict = False
        for sel in selected:
            if sel in corr_matrix.index and fid in corr_matrix.columns:
                corr_val = corr_matrix.loc[sel, fid]
                if abs(corr_val) > threshold and sel != fid:
                    conflict = True
                    redundant_pairs.append({"kept": sel, "removed": fid, "corr": corr_val})
                    break
        if not conflict:
            selected.append(fid)
    return selected, redundant_pairs


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    # ── 读取提纯结果 ──
    purify_path = Path(__file__).parent / "purify_results.csv"
    if not purify_path.exists():
        print("错误: 先运行 purify_v2.py 生成 purify_results.csv")
        return
    purify_df = pd.read_csv(purify_path)
    print(f"加载提纯结果: {len(purify_df)} 个因子")

    # ── 剔除纸老虎 (CS_eff == 0) ──
    paper_tigers = purify_df[purify_df["cs_effective_ratio"] == 0]
    print(f"纸老虎 (CS_eff=0): {len(paper_tigers)} 个 — 剔除")
    if len(paper_tigers):
        for _, r in paper_tigers.head(10).iterrows():
            print(f"  {r['factor']:10s} IC_IR={r['IC_IR']:+.4f}")

    # 只使用 v2 管道通过的因子 (IC_IR + FM + CS 三维全部通过)
    valid = purify_df[purify_df["pass"]].copy()
    # 确保 must_keep 因子不管是否通过 v2 都纳入候选池
    for mk in MUST_KEEP:
        if mk not in valid["factor"].values:
            mk_row = purify_df[purify_df["factor"] == mk]
            if len(mk_row):
                valid = pd.concat([valid, mk_row])
    print(f"v2 管道通过 + must_keep: {len(valid)} 个")
    candidate_fids = valid["factor"].tolist()

    # ── 加载数据（PIT 沪深 300，见 build_pit_matrix.py）──
    print("\n加载数据...")
    from build_pit_matrix import load_pit_panel, build_and_save
    close_matrix, volume_matrix, member_daily = load_pit_panel()
    fwd_ret = close_matrix.pct_change().shift(-1).fillna(0)
    print(f"数据矩阵: {close_matrix.shape}，"
          f"日均成员 {(member_daily & close_matrix.notna()).sum(axis=1).mean():.1f} 只")

    # ── 计算候选因子的因子矩阵（成员掩码内评估）──
    print(f"\n计算 {len(candidate_fids)} 个候选因子...")
    _, factor_tensor = compute_factor_matrix(
        list(close_matrix.columns),
        candidate_fids,
        start=DATE_START, end=DATE_END,
        verbose=True,
    )
    for fid in list(factor_tensor):
        factor_tensor[fid] = factor_tensor[fid].where(member_daily)

    # ── Rank IC 相关性矩阵 + 冗余剔除 ──
    print(f"\n计算 Rank IC 相关性矩阵...")
    ic_ir_map = {r["factor"]: r["IC_IR"] for _, r in purify_df.iterrows()}
    corr_mat = compute_rank_ic_corr_matrix(factor_tensor, fwd_ret)
    corr_mat.to_csv(Path(__file__).parent / "rank_ic_corr_matrix.csv")
    print(f"  矩阵形状: {corr_mat.shape}")

    # 热力图
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_yticks(range(len(corr_mat.columns)))
    ax.set_xticklabels(corr_mat.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr_mat.columns, fontsize=6)
    ax.set_title(f"Rank IC Pearson 相关性矩阵 ({len(corr_mat.columns)} 候选因子)", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    plt.tight_layout()
    corr_fig_path = FIGURES_DIR / "01_rank_ic_corr_heatmap.png"
    fig.savefig(corr_fig_path, dpi=150)
    plt.close(fig)
    print(f"  热力图: {corr_fig_path}")

    # 冗余剔除
    selected, redundant_pairs = remove_redundant(
        corr_mat, ic_ir_map, RANK_IC_CORR_THRESHOLD, MUST_KEEP
    )
    print(f"\n冗余剔除 (Rank IC |corr| > {RANK_IC_CORR_THRESHOLD}):")
    print(f"  候选: {len(candidate_fids)} → 保留: {len(selected)}")
    if redundant_pairs:
        print(f"  剔除明细:")
        for pair in redundant_pairs:
            print(f"    {pair['removed']:10s} → 与 {pair['kept']:10s} 冗余 (corr={pair['corr']:+.3f}), "
                  f"IC_IR: {ic_ir_map.get(pair['kept'], 0):+.3f} > {ic_ir_map.get(pair['removed'], 0):+.3f}")

    # ── 最终因子池 ──
    # 按 |IC_IR| 排序，取前 MAX_POOL_SIZE
    selected_icir = [(f, abs(ic_ir_map.get(f, 0))) for f in selected]
    selected_icir.sort(key=lambda x: x[1], reverse=True)
    final_pool = [f for f, _ in selected_icir[:MAX_POOL_SIZE]]
    # 确保 must_keep 在内
    for mk in MUST_KEEP:
        if mk not in final_pool:
            final_pool.append(mk)

    print(f"\n最终因子池 ({len(final_pool)} 个):")
    for i, fid in enumerate(final_pool):
        info = purify_df[purify_df["factor"] == fid]
        if len(info):
            r = info.iloc[0]
            print(f"  {i+1:2d}. {fid:10s}  IC_IR={r['IC_IR']:+.4f}  "
                  f"FM_t={r['FM_t']:+.2f}  CS_eff={r['cs_effective_ratio']:.3f}")

    # ── 构建特征矩阵 X（PIT，复用 build_pit_matrix，已算好的掩码因子直接传入）──
    print(f"\n构建特征矩阵 X...")
    X = build_and_save(final_pool, factor_tensor={f: factor_tensor[f] for f in final_pool})
    print(f"  特征列: {list(X.columns)}")

    # ── IC_IR 柱状图 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    fids_plot = final_pool[:20]
    irs = [ic_ir_map.get(f, 0) for f in fids_plot]
    colors = ["#2ca02c" if ir >= 0 else "#d62728" for ir in irs]
    bars = ax.bar(range(len(fids_plot)), irs, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=IC_IR_THRESHOLD, color="green", linestyle="--", alpha=0.5, label=f"±{IC_IR_THRESHOLD}")
    ax.axhline(y=-IC_IR_THRESHOLD, color="green", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(fids_plot)))
    ax.set_xticklabels(fids_plot, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IC_IR")
    ax.set_title("最终因子池 IC_IR", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    icir_fig_path = FIGURES_DIR / "02_final_pool_icir.png"
    fig.savefig(icir_fig_path, dpi=150)
    plt.close(fig)
    print(f"  IC_IR 图: {icir_fig_path}")

    # ── 截面有效比分布 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    cs_vals = purify_df["cs_effective_ratio"].dropna()
    ax.hist(cs_vals, bins=40, edgecolor="white", alpha=0.8)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, label=f"阈值 = {CS_EFFECTIVE_THRESHOLD}")
    ax.set_xlabel("截面有效比")
    ax.set_ylabel("因子数量")
    ax.set_title(f"截面有效比分布 (191 个因子, PIT 成员掩码内)")
    ax.legend()
    plt.tight_layout()
    cs_fig_path = FIGURES_DIR / "03_cs_effective_ratio_dist.png"
    fig.savefig(cs_fig_path, dpi=150)
    plt.close(fig)
    print(f"  截面有效比图: {cs_fig_path}")

    print(f"\n  === 特征筛选完成 ===")
    return X, final_pool


if __name__ == "__main__":
    main()
