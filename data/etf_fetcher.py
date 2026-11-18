"""
data/etf_fetcher.py — ETF 日线行情获取与本地缓存模块。

数据源：`akshare.fund_etf_hist_sina`，**返回未复权原始价格**（该接口无复权参数）。
原始数据缓存到 `data/cache_etf/{symbol}.csv`；后复权版本由
`data/etf_adjust.py` 构造并缓存为 `{symbol}_hfq.csv`。

`load_etf_daily()` 默认返回**后复权**数据 —— 未复权的分红与份额折算跳空
会使 ETF 策略的收益与回撤严重失真（详见该函数 Notes）。
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


def _load_raw(symbol: str) -> Optional[pd.DataFrame]:
    """读取未复权原始缓存（`{symbol}.csv`）。"""
    p = get_cache_path(symbol)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()


def load_etf_daily(symbol: str, adjust: str = "hfq") -> Optional[pd.DataFrame]:
    """从本地缓存读取 ETF 日线数据。

    Parameters
    ----------
    symbol : str
        6 位 ETF 代码。
    adjust : {"hfq", "raw"}, default "hfq"
        - ``"hfq"``：**后复权**，已修正分红与份额折算（缓存 `{symbol}_hfq.csv`）。
          回测必须用此口径。
        - ``"raw"``：原始未复权价格（缓存 `{symbol}.csv`）。仅用于需要真实
          报价的场景；成交额类因子用 `amount`，不受复权影响。

    Notes
    -----
    **2026-11-18 修复**：此前本函数直接返回未复权价格，而 docstring 误称
    「前复权」。实测污染（18 只 ETF，2015-2025）：

      - 510500 在 2015-04-15 有 +248.6% 的份额折算跳空 → 年化虚高 12.6pp
      - 512200 在 2024-08-12 有 +170.1% → 对等权组合单日贡献 +17.01%
      - 512010 在 2021-06-28 有 −73.9% → 年化虚低 12.6pp
      - 510050 漏计 10 次分红共 0.506 元 → 年化虚低 1.7pp

    复权因子构造见 `data/etf_adjust.py`（分红 45 次 + 折算 8 处，逐只独立计算）。
    ``hfq`` 缓存缺失时按原始数据现场生成，不静默回退到未复权价格。
    """
    if adjust == "raw":
        return _load_raw(symbol)
    if adjust != "hfq":
        raise ValueError(f"adjust 必须是 'hfq' 或 'raw'，得到 {adjust!r}")

    p = CACHE_ETF_DIR / f"{symbol}_hfq.csv"
    if not p.exists():
        raw = _load_raw(symbol)
        if raw is None:
            return None
        from data.etf_adjust import write_adjusted_cache

        write_adjusted_cache(symbol, raw, verbose=False)
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
