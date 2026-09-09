"""
run_pit_audit.py — 基本面数据 PIT 约定审计入口。

两种审计模式：
  1. 离线约定审计（默认，推荐）：不联网。检查 data/cache_fundamental 各因子缓存的
     日期是否全部落在「法定截止日 + 1」约定集合内。2026-09-09 实测发现
     roeAvg/gpMargin/npMargin/epsTTM 四个文件混入了非约定日期行（真公告日/pubDate 行），
     其中 gpMargin 为精确 100× 尺度错乱（小数 vs 百分数并存），污染 X_monthly。
  2. 在线抽样比对（audit_pit_leakage）：重新抓取新浪数据做比对，依赖网络，易失败。

用法:
    python data/run_pit_audit.py            # 离线约定审计
    python data/run_pit_audit.py --online   # 在线抽样比对（需网络）
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.akshare_fundamental_fetcher import statutory_deadline  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_fundamental")


def _allowed_pit_dates(start_year: int = 2006, end_year: int = 2028) -> set:
    """法定截止日 + 1 的约定日期集合。"""
    return {
        (statutory_deadline(y, q) + pd.Timedelta(days=1)).date()
        for y in range(start_year, end_year + 1)
        for q in range(1, 5)
    }


def offline_convention_audit() -> dict:
    """离线审计：每个因子缓存的所有日期必须属于约定集合。

    返回 {factor_name: {'total_dates', 'bad_dates', 'bad_values', 'stocks_affected'}}。
    bad_values > 0 即存在违反「截止日+1」约定的行（可能为真 pubDate 行，需人工甄别
    是否尺度混入 —— 2026-09-09 确认 gpMargin/roeAvg 为 100× 尺度错乱）。
    """
    allowed = _allowed_pit_dates()
    report: dict = {}
    print(f"{'因子':<20}{'总日期':>7}{'违规日期':>9}{'违规值':>10}{'涉事股票':>8}")
    print("-" * 58)
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.csv"))):
        name = os.path.basename(f).replace(".csv", "")
        try:
            df = pd.read_csv(f, parse_dates=["date"], index_col="date", low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<20} 读取失败: {e}")
            continue
        bad_idx = pd.Index([d for d in df.index if d.date() not in allowed])
        n_bad_dates = len(bad_idx)
        if n_bad_dates == 0:
            print(f"  ✅ {name:<18}{len(df.index):>7}{0:>9}{0:>10}{0:>8}")
            report[name] = {"total_dates": len(df.index), "bad_dates": 0,
                            "bad_values": 0, "stocks_affected": 0}
            continue
        bad_df = df.loc[bad_idx]
        n_bad_vals = int(bad_df.notna().sum().sum())
        n_stocks = int(bad_df.notna().any().sum())
        status = "❌" if n_bad_vals > 0 else "⚠️"
        print(f"  {status} {name:<18}{len(df.index):>7}{n_bad_dates:>9}{n_bad_vals:>10}{n_stocks:>8}")
        report[name] = {"total_dates": len(df.index), "bad_dates": n_bad_dates,
                        "bad_values": n_bad_vals, "stocks_affected": n_stocks}
    print("-" * 58)
    bad = {k: v for k, v in report.items() if v["bad_values"] > 0}
    print(f"\n总结: {len(report)} 个因子缓存，{len(bad)} 个含非约定日期行。")
    if bad:
        print("注意: 非约定日期多为真公告日行。2026-09-09 已确认 gpMargin/roeAvg 属")
        print("100× 尺度错乱（小数 vs 百分数并存），会污染截面特征矩阵，需清理后重跑。")
    return report


def _online_sampling_audit():
    """（旧模式）在线抓取比对，依赖网络，保留作参考。"""
    from data.akshare_fundamental_fetcher import (  # noqa: PLC0415
        SINA_TO_FACTOR, _report_period_to_pit, fetch_stock_financial,
    )
    from data.pit_auditor import audit_pit_leakage  # noqa: PLC0415

    cache_path = os.path.join(CACHE_DIR, "roeAvg.csv")
    if not os.path.exists(cache_path):
        print("未找到 roeAvg.csv 缓存，跳过。")
        return
    aligned_matrix = pd.read_csv(cache_path, parse_dates=["date"], index_col="date",
                                 low_memory=False)
    sample = [c for c in aligned_matrix.columns
              if "000001" in c or "600000" in c][:2] or aligned_matrix.columns[:2].tolist()
    raw_records = []
    for sym in sample:
        try:
            df = fetch_stock_financial(sym)
            if df is None or df.empty or "日期" not in df.columns:
                continue
            df["_period"] = df["日期"].astype(str)
            for col, factor in SINA_TO_FACTOR.items():
                if factor != "roeAvg" or col not in df.columns:
                    continue
                for _, row in df.iterrows():
                    val = row[col]
                    if pd.isna(val):
                        continue
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    raw_records.append({
                        "stock_code": sym,
                        "ann_date": _report_period_to_pit(row["_period"]),
                        "end_date": pd.Timestamp(row["_period"]),
                        "value": fv,
                    })
        except Exception:  # noqa: BLE001
            pass
    if not raw_records:
        print("无法构建原始财务表，退出在线审计。")
        return
    report = audit_pit_leakage(pd.DataFrame(raw_records),
                               aligned_matrix[sample])
    print("审计结论:", report["status"], "| 泄露:", report["leakage_count"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="在线抽样比对（需网络）")
    args = ap.parse_args()
    if args.online:
        _online_sampling_audit()
    else:
        offline_convention_audit()
