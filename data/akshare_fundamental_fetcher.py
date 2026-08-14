"""
akshare_fundamental_fetcher.py — 新浪财务指标拉取（akshare 换源，替代 baostock 财报）。

背景（2026-08-14）：baostock 对单 IP 有【按天累计查询配额】——单进程 2.9 q/s
串行拉 ~5 万次仍触发 10001011 黑名单。方向2 需 21 财报因子 × 1625 只 ≈ 74 万次
查询，跨天分批要 ~20 天。akshare 新浪接口 stock_financial_analysis_indicator：
每只股票 1 次调用拿全部历史 86 指标 → 1625 只 ≈ 70min 拉完，无 IP 配额墙。

PIT 对齐（关键差异，必须让用户知情）：
  - 新浪接口只返回【报告期】（如 2010-03-31），无公告日。
  - 本模块用法定截止日回退：Q1→5/1，Q2→9/1，Q3→11/1，Q4→次年5/1，再 +1 天。
  - 这是保守偏误（不违反未来函数铁律），但损失提前披露公司的 PEAD alpha，
    报告须如实披露此局限。
  - 对比 baostock 的 pubDate 精确对齐（已被配额墙锁死）。若未来换 tushare
    积分档，可升级为真公告日对齐——本模块缓存格式（date×symbol 宽表）兼容。

缓存格式与 baostock 完全一致（date×symbol 宽表，date=PIT 生效日），
factors.py 的 load_cached_field 接口不变，方向2 全链代码不受影响。

用法:
    python data/akshare_fundamental_fetcher.py --subset 5    # 小样本验证
    python data/akshare_fundamental_fetcher.py               # 全量 1625 只 ≈ 70min
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 报告期 → 法定披露截止日（与 baostock 版 fundamental_fetcher._statutory_deadline 一致）
_STATUTORY = {1: (5, 1), 2: (9, 1), 3: (11, 1), 4: (None, 5, 1)}  # Q4 跨年


def statutory_deadline(year: int, quarter: int) -> pd.Timestamp:
    """法定披露截止日（PIT 回退基准）。Q4 → 次年 5/1。"""
    if quarter == 4:
        return pd.Timestamp(year=year + 1, month=5, day=1)
    m, d = _STATUTORY[quarter]
    return pd.Timestamp(year=year, month=m, day=d)


def _report_period_to_pit(report_period: str) -> pd.Timestamp:
    """报告期 '2010-03-31' → PIT 有效日 = 法定截止日 + 1 天。"""
    ts = pd.Timestamp(report_period)
    return statutory_deadline(ts.year, ts.quarter) + pd.Timedelta(days=1)


# 新浪 86 指标列名 → 方向2 21 财报因子（无映射的因子标 None，需另路）
SINA_TO_FACTOR: dict[str, str] = {
    # 质量
    "加权净资产收益率(%)": "roeAvg",            # 注意：新浪无 roeAvg 精确对应，用加权 ROE
    "净资产收益率(%)": "roeAvg",                # 非加权备选（与 baostock roeAvg 更近）
    "销售净利率(%)": "npMargin",
    "销售毛利率(%)": "gpMargin",
    "摊薄每股收益(元)": "epsTTM",               # 新浪无 epsTTM，用摊薄 EPS（单季）
    # 成长
    "净利润增长率(%)": "YOYNI",
    "主营业务收入增长率(%)": "YOYOR",          # 新浪无 YOYNI 精确同比，用营收增长
    "总资产增长率(%)": "YOYAsset",
    # 现金流
    "经营现金净流量与净利润的比率(%)": "CFOToNP",
    # 营运
    "存货周转率(次)": "NRTurnRatio",
    "应收账款周转率(次)": "INVTurnRatio",
    "流动资产周转率(次)": "CATurnRatio",
    "总资产周转率(次)": "AssetTurnRatio",
    # 杠杆
    "资产负债率(%)": "liabilityToAsset",
    "流动比率": "currentRatio",
    "现金比率(%)": "cashRatio",
    "利息支付倍数": "ebitToInterest",
    # 现金流（续）
    "经营现金净流量对销售收入比率(%)": "CFOToOR",
}

# 未覆盖的 baostock 因子（新浪无对应列）：dupontROE, YOYPNI, YOYEPSBasic,
# CFOToGr, CAToAsset。其中 dupontROE 可用 roeAvg 近似；其余暂缺。


def fetch_stock_financial(symbol: str) -> pd.DataFrame:
    """单只股票全部历史财务指标（新浪接口）。返回 日期×指标 表。"""
    import akshare as ak
    return ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2007")


def _to_cache_path(factor: str) -> Path:
    return PROJECT_ROOT / "data" / "cache_fundamental" / f"{factor}.csv"


def _merge_write(df_wide: pd.DataFrame, factor: str):
    """合并写盘（date×symbol 宽表），与 baostock 缓存格式一致。"""
    path = _to_cache_path(factor)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"], index_col="date", dtype=str).astype(float)
        all_cols = cached.columns.union(df_wide.columns)
        all_idx = cached.index.union(df_wide.index)
        merged = pd.DataFrame(index=all_idx, columns=all_cols, dtype=float)
        for src in (cached, df_wide):
            for col in src.columns:
                merged.loc[src.index, col] = src[col].values
        merged.to_csv(path)
    else:
        df_wide.to_csv(path)


def fetch_all(symbols: list[str], subset: int | None = None, sleep_s: float = 0.1) -> dict[str, pd.DataFrame]:
    """拉取全部股票财务指标 → 转 21 因子 date×symbol 宽表 → 合并写盘。

    subset: 只拉前 N 只（小样本验证）。sleep_s: 每股间隔（新浪无配额墙，0.1s 足够）。
    """
    if subset is not None:
        symbols = symbols[:subset]

    # 报告期(行) × 因子(列) 的每因子记录累积
    records: dict[str, list[dict]] = {f: [] for f in set(SINA_TO_FACTOR.values())}
    n_err = 0
    t0 = time.time()
    for i, sym in enumerate(symbols):
        try:
            df = fetch_stock_financial(sym)
            if df is None or df.empty or "日期" not in df.columns:
                raise ValueError("空返回")
            df["_period"] = df["日期"].astype(str)
            for col, factor in SINA_TO_FACTOR.items():
                if col not in df.columns:
                    continue
                for _, row in df.iterrows():
                    val = row[col]
                    if val is None or pd.isna(val) or val is False:
                        continue
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    pit = _report_period_to_pit(row["_period"])
                    records[factor].append({"date": pit, "symbol": sym, "value": fv})
        except Exception:
            n_err += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"  [{i+1}/{len(symbols)}] {elapsed:.0f}s ETA {eta/60:.0f}min "
                  f"({(i+1)/elapsed:.1f} 只/s, {n_err} err)", flush=True)
        if sleep_s:
            time.sleep(sleep_s)

    # 每因子转宽表 + 合并写盘
    result: dict[str, pd.DataFrame] = {}
    for factor, recs in records.items():
        if not recs:
            print(f"  [WARN] {factor}: 无数据")
            continue
        long = pd.DataFrame(recs).drop_duplicates(subset=["date", "symbol"], keep="last")
        wide = long.pivot(index="date", columns="symbol", values="value").sort_index().sort_index(axis=1)
        wide.index = pd.to_datetime(wide.index)
        _merge_write(wide, factor)
        print(f"  {factor}: {len(wide)} 日期 × {len(wide.columns)} 股票")
        result[factor] = wide
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 72)
    print("akshare 财务指标拉取（新浪源，替代 baostock 财报）")
    print(f"PIT = 报告期法定截止日 + 1 天（非公告日，保守偏误）")
    print("=" * 72)
    from data.index_membership import load_membership
    mem = load_membership("zz500")
    symbols = sorted(mem.columns.astype(str))
    print(f"PIT zz500 股票池: {len(symbols)} 只")
    fetch_all(symbols, subset=args.subset, sleep_s=args.sleep)
    print("\n完成。因子缓存写入 data/cache_fundamental/")


if __name__ == "__main__":
    main()
