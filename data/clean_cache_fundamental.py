"""
clean_cache_fundamental.py — 清理 cache_fundamental 中的非约定日期行（PIT 审计修复）。

背景（2026-09-09 PIT 审计发现）：
  roeAvg / gpMargin / npMargin / epsTTM 四个缓存混入了「真公告日」行（非约定日期），
  其中 gpMargin/roeAvg 为精确 100× 尺度错乱（pubDate 行存小数、截止日行存百分数）。
  二者并存导致月末采样时相邻月末在 0.14 与 14.2 之间跳变，污染 X_monthly（21% 股票受影响）。

修复方式：
  保留「法定截止日 + 1」约定集合内的行（当前规范管线的产物，覆盖全部股票），
  删除非约定日期行。删除前先备份整个 cache_fundamental 目录。

用法:
    python data/clean_cache_fundamental.py          # 备份 + 清理 + 覆盖率检查
    python data/clean_cache_fundamental.py --noop   # 只报告将删除的行，不实际删除
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.akshare_fundamental_fetcher import statutory_deadline  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_fundamental")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cache_fundamental_backup_20260909")
DIRTY_FIELDS = {"roeAvg", "gpMargin", "npMargin", "epsTTM"}


def _allowed_pit_dates() -> set:
    return {
        (statutory_deadline(y, q) + pd.Timedelta(days=1)).date()
        for y in range(2006, 2028)
        for q in range(1, 5)
    }


def _coverage(df: pd.DataFrame) -> int:
    """有 ≥1 个非空值的股票数。"""
    return int(df.notna().any(axis=1).pipe(lambda s: df.loc[s].columns.isin(df.columns)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noop", action="store_true", help="只报告不删除")
    args = ap.parse_args()

    allowed = _allowed_pit_dates()
    print("=" * 68)
    print(f"清理 cache_fundamental 非约定日期行 | noop={args.noop}")
    print("=" * 68)

    # 备份（非 noop 时）
    if not args.noop and not os.path.exists(BACKUP_DIR):
        shutil.copytree(CACHE_DIR, BACKUP_DIR)
        print(f"已备份原缓存 -> {BACKUP_DIR}")

    total_removed = 0
    for name in sorted(DIRTY_FIELDS):
        path = os.path.join(CACHE_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  跳过（不存在）: {name}")
            continue
        df = pd.read_csv(path, parse_dates=["date"], index_col="date", low_memory=False)
        n_before = int(df.notna().sum().sum())
        stocks_before = int(df.notna().any().sum())

        # 布尔掩码过滤（保留索引名，避免 .loc[新Index] 丢失 index.name）
        keep = df[df.index.map(lambda d: d.date() in allowed)]
        removed = int(df.notna().sum().sum()) - int(keep.notna().sum().sum())
        stocks_after = int(keep.notna().any().sum())
        total_removed += removed

        print(f"  {name:<20} 值 {n_before:>8} -> {int(keep.notna().sum().sum()):>8}"
              f" (删 {removed:>7}) | 股票 {stocks_before} -> {stocks_after}")
        if not args.noop:
            keep.to_csv(path, index_label="date")  # 显式写索引名，防止表头丢失

    print("-" * 68)
    print(f"共删除 {total_removed} 个非约定日期值。")

    # 覆盖率检查：清理后每只股票是否仍有数据
    if not args.noop:
        print("\n覆盖率检查（清理后）：")
        for name in sorted(DIRTY_FIELDS):
            path = os.path.join(CACHE_DIR, f"{name}.csv")
            df = pd.read_csv(path, parse_dates=["date"], index_col="date", low_memory=False)
            n_stocks = int(df.notna().any().sum())
            n_dates = len(df.index)
            print(f"  {name:<20} 日期数={n_dates} | 覆盖股票={n_stocks}")


if __name__ == "__main__":
    main()
