"""
data/etf_fetcher.py — ETF 日线行情获取与本地缓存模块。

使用 akshare.fund_etf_hist_sina 拉取 ETF 前复权历史日线行情，
并自动标准化缓存到 data/cache_etf/{symbol}.csv。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import akshare as ak
import pandas as pd

CACHE_ETF_DIR = Path(__file__).parent / "cache_etf"
CACHE_ETF_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(symbol: str) -> Path:
    """获取指定 ETF 的本地缓存路径。"""
    return CACHE_ETF_DIR / f"{symbol}.csv"


def _format_symbol_with_prefix(symbol: str) -> str:
    """添加 sh/sz 前缀以适配新浪接口。"""
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    else:
        return f"sz{symbol}"


def download_etf_daily(
    symbol: str,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """下载单只 ETF 日线数据并缓存到本地 CSV。"""
    cache_path = get_cache_path(symbol)
    full_sym = _format_symbol_with_prefix(symbol)

    df_raw = ak.fund_etf_hist_sina(symbol=full_sym)
    if df_raw is None or len(df_raw) == 0:
        raise RuntimeError(f"未能获取 ETF {symbol} 数据")

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 裁剪到目标区间
    df_filtered = df.loc[(df.index >= start) & (df.index <= end), [c for c in cols if c in df.columns]]
    df_filtered.to_csv(cache_path)
    return df_filtered


def load_etf_daily(symbol: str) -> Optional[pd.DataFrame]:
    """从本地缓存读取 ETF 日线数据。"""
    p = get_cache_path(symbol)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()


def sync_all_etfs(
    symbols: List[str],
    start: str = "2015-01-01",
    end: str = "2025-12-31",
) -> Tuple[List[str], List[str]]:
    """批量同步并缓存指定 ETF 池全部数据。"""
    ok, fail = [], []
    for sym in symbols:
        try:
            df = download_etf_daily(sym, start=start, end=end)
            ok.append(sym)
            print(f"ETF {sym} cached ✓ ({len(df)} bars, {df.index.min().date()} ~ {df.index.max().date()})")
        except Exception as e:
            fail.append(sym)
            print(f"ETF {sym} failed ✗ ({e})")
    return ok, fail
