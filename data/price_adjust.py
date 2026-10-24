"""复权因子与真 VWAP 构造。

为什么需要这个模块（2026-10-24 alpha191 审计）：
  1. **前复权污染**：前复权价由「未来」的除权事件回溯调整，用它计算价格水平类
     因子等于把未来信息带进历史截面（CLAUDE.md 铁律 3.5）。
  2. **VWAP 量纲错配**：`amount / volume` 是**未复权**的真实成交均价，而
     `close` 是复权后价格，两者不同量纲。实测 000001：vwap_raw / close_adj2
     中位 1.31、范围 1.02~4.01，vwap 甚至高于当日最高价 → 所有
     `MIN(vwap - price, 0)` 类因子退化为常数（实测 15 个因子输出无方差）。

还原方式：
    f(t)     = close_adj1 / close_adj3            # 后复权 / 不复权
    vwap_adj1 = (amount / volume) * f(t)

  数学上精确：同一天 OHLC 共用一个价格复权因子，故
  `OHLC_adj1 = OHLC_adj3 * f(t)`（实测最大相对偏差 1e-16）。

实测校验（000001 / 600000，2010-2025）：
    (amount/volume) / close_adj3      中位 0.9999 / 1.0003   ← 原始口径自洽
    vwap_adj1 / close_adj1            中位 0.9999 / 1.0003   ← 还原正确
    p1~p99                            0.975~1.022
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.fetcher import load_daily

# symbol -> adjust factor Series（进程内缓存，避免重复读 CSV）
_FACTOR_CACHE: dict[str, pd.Series | None] = {}


def adjust_factor(symbol: str) -> pd.Series | None:
    """后复权价格因子 f(t) = close_adj1 / close_adj3。

    任一缓存缺失时返回 None（不静默返回 1.0——那会把量纲错误伪装成正常值）。
    """
    if symbol in _FACTOR_CACHE:
        return _FACTOR_CACHE[symbol]

    a1 = load_daily(symbol, adjust="1")
    a3 = load_daily(symbol, adjust="3")
    if a1 is None or a3 is None:
        _FACTOR_CACHE[symbol] = None
        return None

    common = a1.index.intersection(a3.index)
    f = (a1.loc[common, "close"] / a3.loc[common, "close"].replace(0, np.nan))
    f = f.replace([np.inf, -np.inf], np.nan).rename(symbol)
    _FACTOR_CACHE[symbol] = f
    return f


def build_true_vwap(panel: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    """把 `amount/volume` 拉到后复权口径。

    缺复权因子的股票，vwap 整列置 NaN——量纲不一致的数值比缺失更有害。
    """
    vwap_raw = panel["amount"] / panel["volume"]
    factors: dict[str, pd.Series] = {}
    for sym in symbols:
        f = adjust_factor(sym)
        if f is not None:
            factors[sym] = f.reindex(vwap_raw.index)

    if not factors:
        raise RuntimeError(
            "缺少 adj1/adj3 缓存，无法构造后复权 VWAP。"
            "先运行 data/download_adjust_cache.py"
        )

    F = pd.DataFrame(factors)
    vwap = vwap_raw * F.reindex(columns=vwap_raw.columns)
    return vwap


def vwap_scale_stats(vwap: pd.DataFrame, close: pd.DataFrame) -> dict[str, float]:
    """vwap/close 量纲自检统计，用于 gate 断言。"""
    ratio = (vwap / close).replace([np.inf, -np.inf], np.nan)
    stacked = ratio.stack()
    return {
        "median": float(stacked.median()),
        "p01": float(stacked.quantile(0.01)),
        "p99": float(stacked.quantile(0.99)),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
    }
