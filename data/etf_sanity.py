"""
data/etf_sanity.py — ETF 跳空诊断工具（已被 `data/etf_adjust.py` 取代为正式修复路径）。

> **定位说明（2026-11-18）**：本模块最初用于"调用侧局部前复权"，现已由
> `data/etf_adjust.py` 的后复权因子方案取代 —— 后者同时修正分红与份额折算，
> 且已下沉为 `load_etf_daily()` 的默认口径。
> 本模块保留两个用途：
>   1. `detect_price_jumps()` 作为**数据质量巡检**，任何新 ETF 入库后先扫一遍；
>   2. `sanitize_price_jumps()` 作为无分红数据时的应急兜底。
> 新代码应直接用 `load_etf_daily(symbol)`（默认 hfq）。

## 问题

`data/etf_fetcher.download_etf_daily()` 的 docstring 声称拉取「前复权」行情，
但实际调用的 `akshare.fund_etf_hist_sina()` **返回未复权价格**，没有复权参数。
分红、份额折算当日在原始价格上表现为断崖式跳空。

实测污染（18 只 ETF 池，2015-2025）：

| 日期 | 标的 | 单日涨跌 | 组合权重 | 对组合贡献 |
|---|---|---|---|---|
| 2024-08-12 | 512200 房地产ETF | **+170.07%** | 0.100 | **+17.01%** |
| 2025-07-07 | 512800 银行ETF | **−49.69%** | 0.150 | **−8.01%** |
| 2021-06-25 | 159928 主要消费ETF | −74.47% | 0.000 | 0.00% |

单日 +17% 的虚假收益足以让任何 ETF 轮动策略的年化与夏普失去意义。

## 修正方法

A 股 ETF 日涨跌幅上限为 10%（创业板/科创板标的 20%），
因此 `|r| > 0.35` 可判定为数据错误而非真实行情。对识别出的跳空点，
把该日及之后的全部价格除以跳空比例 $k_t = P_t / P_{t-1}$，
使序列在断点处连续（等价于局部前复权），后续收益不受影响。

本模块**不修改** `data/etf_fetcher.py` 的既有缓存，只在调用侧清洗。
是否将清洗下沉到数据层、并重跑已发布的 ETF 报告，需另行评估。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

#: 判定为数据错误的单日涨跌幅阈值。20% 是创业板 ETF 的合法上限，故留足余量。
JUMP_THRESHOLD: float = 0.35


def detect_price_jumps(
    close: pd.DataFrame,
    threshold: float = JUMP_THRESHOLD,
) -> pd.DataFrame:
    """列出所有超过阈值的单日跳空。"""
    r = close.pct_change(fill_method=None)
    stacked = r.stack()
    bad = stacked[stacked.abs() > threshold]
    if bad.empty:
        return pd.DataFrame(columns=["date", "symbol", "ret"])
    out = bad.rename("ret").reset_index()
    out.columns = ["date", "symbol", "ret"]
    return out.sort_values("ret", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def sanitize_price_jumps(
    close: pd.DataFrame,
    threshold: float = JUMP_THRESHOLD,
    *,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """修正未复权价格的跳空，返回 (清洗后价格矩阵, 事件清单)。

    对每个标的独立处理：累积修正因子
    $$f_t = \\prod_{s \\le t,\\ |r_s| > \\tau} (1 + r_s), \\qquad P^{\\text{clean}}_t = P_t / f_t$$
    使得断点处收益率归零、其后收益率保持原样。
    """
    if close.empty:
        return close, pd.DataFrame(columns=["date", "symbol", "ret"])

    events = detect_price_jumps(close, threshold)
    r = close.pct_change(fill_method=None)
    bad = (r.abs() > threshold).fillna(False)

    # 累积修正因子：仅在异常点累乘 (1+r)
    factor = (1.0 + r.where(bad, 0.0)).cumprod()
    clean = close / factor
    clean = clean.where(close.notna())

    if verbose and not events.empty:
        print(f"[etf_sanity] 修正 {len(events)} 处未复权跳空（阈值 {threshold:.0%}）：")
        for _, e in events.iterrows():
            print(f"    {e['date'].date()}  {e['symbol']}  {e['ret']:+.2%}")

    return clean, events
