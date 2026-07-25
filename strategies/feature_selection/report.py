"""
report.py — 特征筛选主报告。

流程:
  1. 提纯管道 v2 (panel 模式 RANK + 第三维)
  2. 冗余剔除 + 最终特征矩阵构建
  3. 输出全部指标供撰写 report.md

用法: python report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from purify_v2 import main as purify_main
from select_features import main as select_main


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    section("Phase 1: 提纯管道 v2")
    purify_df = purify_main()

    section("Phase 2: 特征筛选")
    X, y, final_pool = select_main()

    # ── 汇总 ──
    section("报告汇总")

    purify_path = Path(__file__).parent / "purify_results.csv"
    df = pd.read_csv(purify_path)

    n_total = len(df)
    n_v1 = (df["IC_pass"] & df["FM_pass"]).sum()
    n_v2 = df["pass"].sum()
    n_paper_tiger = (df["cs_effective_ratio"] == 0).sum()
    n_paper_tiger_high_ic = ((abs(df["IC_IR"]) > 0.05) & (df["cs_effective_ratio"] == 0)).sum()
    n_cs_pass = (df["cs_effective_ratio"] > 0.5).sum()

    print(f"\n  数据概况:")
    print(f"    股票数: 300 (CSI 300 全量)")
    print(f"    时间范围: 2010-01-01 ~ 2025-12-31")
    print(f"    因子总数: {n_total}")
    print(f"")
    print(f"  提纯管道:")
    print(f"    v1 (IC_IR + FM): {n_v1} 个通过")
    print(f"    v2 (IC_IR + FM + CS_eff): {n_v2} 个通过")
    print(f"    Δ (v2 新增剔除): {n_v1 - n_v2} 个因截面有效比不足被筛掉")
    print(f"")
    print(f"  纸老虎:")
    print(f"    CS_eff == 0: {n_paper_tiger} 个")
    print(f"    其中 |IC_IR| > 0.05: {n_paper_tiger_high_ic} 个 (IC 虚高但无法分组)")
    print(f"")
    print(f"  第三维统计:")
    print(f"    截面有效比 > 0.5: {n_cs_pass} 个")
    print(f"    CS_eff 中位: {df['cs_effective_ratio'].median():.3f}")
    print(f"")
    print(f"  最终因子池: {len(final_pool)} 个")
    for i, fid in enumerate(final_pool):
        row = df[df["factor"] == fid]
        if len(row):
            r = row.iloc[0]
            print(f"    {i+1:2d}. {fid:10s}  IC_IR={r['IC_IR']:+.4f}  "
                  f"FM_t={r['FM_t']:+.2f}  CS_eff={r['cs_effective_ratio']:.3f}")

    print(f"\n  输出文件:")
    base = Path(__file__).parent
    for fname in ["purify_results.csv", "rank_ic_corr_matrix.csv",
                   "X_matrix.csv", "y_matrix.csv"]:
        if (base / fname).exists():
            print(f"    ✓ {fname}")
    for fname in sorted((base / "figures").glob("*.png")):
        print(f"    ✓ figures/{fname.name}")

    print(f"\n  === 报告完成 ===")


if __name__ == "__main__":
    main()
