"""
data/etf_adjust.py — ETF 后复权因子构造（分红 + 份额折算双重修正）。

## 为什么不能直接用现成接口

| 数据源 | 问题 |
|---|---|
| `akshare.fund_etf_hist_sina` | **无复权参数**，返回原始价格（`etf_fetcher` 现有实现） |
| `akshare.fund_etf_hist_em(adjust="hfq")` | 有真后复权，但 2026-11-18 实测连续 6 只全部请求失败，不可依赖 |
| `baostock.query_history_k_data_plus(adjustflag="1")` | **对 ETF 不生效**：512200 的后复权与不复权收盘价逐位相同（1.1910） |

## 实测污染规模（18 只 ETF，2015-2025）

份额折算 6 处，幅度 53%~249%：

| 日期 | 标的 | 跳空 | 前收 → 当日 |
|---|---|---|---|
| 2015-04-15 | 510500 中证500ETF | **+248.6%** | 2.243 → 7.818 |
| 2022-09-05 | 512100 中证1000ETF | **+176.3%** | 0.982 → 2.713 |
| 2024-08-12 | 512200 房地产ETF | **+170.1%** | 0.441 → 1.191 |
| 2020-09-10 | 512760 芯片ETF | −53.1% | 2.271 → 1.065 |
| 2021-06-25 | 159928 主要消费ETF | −74.5% | 4.940 → 1.261 |
| 2021-06-28 | 512010 医药ETF | −73.9% | 3.206 → 0.836 |

分红：510050 累计 0.797 元（占末价 25.7%，18 次），510300 累计 0.880（18.5%，14 次），
511880 累计 33.267（33.2%，13 次）。不修正则长周期年化被系统性低估 1.5~2%。

## 修正方法

后复权因子 $f_t$ 由两部分相乘：

$$f_t = \\prod_{s \\le t,\\, s \\in \\text{div}} \\frac{P_{s-1}}{P_{s-1}-d_s} \\times \\prod_{s \\le t,\\, |r_s|>\\tau} \\frac{1}{1+r_s}$$

第一项来自除息日 $s$ 的每份分红 $d_s$（$r_s$ 为除息后真实收益）；
第二项处理份额折算 —— 折算不改市值，故当日真实收益接近 0，用跳空比例 $1+r_s$ 直接还原。
$\tau$ 按标的分层：创业板 ETF（159915）合法涨跌停 20%，用 0.22；其余 ETF 为 10%，用 0.15。

复权后 `close_adj = close × f`，`volume` / `amount` **不变**（成交额占比类因子用原始值）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CACHE_ETF_DIR = Path(__file__).parent / "cache_etf"

#: 份额折算 / 数据错误跳空阈值。
#: 主板 ETF 涨跌停 10%，取 0.12 留 2pp 余量 —— 实测 510500 在 2022-08-29 有
#: −12.74% 的异常（前收 7.220 → 开盘 6.200，同期沪深300ETF 仅 −0.6%），
#: 属数据源错误而非真实行情，阈值 0.15 会漏掉它。
JUMP_THRESHOLD_DEFAULT: float = 0.12
#: 创业板 ETF（159915）跟踪指数涨跌幅上限 20%，须豁免合法涨停
JUMP_THRESHOLD_CYB: float = 0.22
#: 创业板 ETF：跟踪指数涨跌幅上限 20%
CYB_SYMBOLS: set[str] = {"159915"}


def _prefix(symbol: str) -> str:
    return ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol


def fetch_dividends(symbol: str, tries: int = 4) -> Optional[pd.DataFrame]:
    """拉取 ETF 累计分红记录（除息日 + 累计每份分红）。失败返回 None。"""
    import akshare as ak

    for i in range(tries):
        try:
            df = ak.fund_etf_dividend_sina(symbol=_prefix(symbol))
            if df is None:
                return None
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            df["累计分红"] = pd.to_numeric(df["累计分红"], errors="coerce")
            return df.dropna().sort_values("日期").reset_index(drop=True)
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def build_adjustment_factor(
    close: pd.Series,
    dividends: Optional[pd.DataFrame] = None,
    jump_threshold: float = JUMP_THRESHOLD_DEFAULT,
) -> Tuple[pd.Series, dict]:
    """构造后复权因子与诊断信息。

    Returns
    -------
    (factor, diag)
        `factor` 与 `close` 同索引，`close * factor` 即后复权价。
        `diag` 含 n_div（实际生效分红次数）、div_total（每份分红合计）、
        n_jump（折算修正次数）、residual_max（修正后最大 |日收益|）。
    """
    s = close.astype(float).copy()
    s = s[~s.index.duplicated()].sort_index()
    factor = pd.Series(1.0, index=s.index)

    n_div = 0
    div_total = 0.0

    # ── 1) 分红因子 ──────────────────────────────────────────
    if dividends is not None and len(dividends) > 0:
        cum = dividends["累计分红"].to_numpy(float)
        single = np.diff(np.concatenate([[0.0], cum]))
        for date, d in zip(dividends["日期"], single):
            if not np.isfinite(d) or d <= 1e-8:
                continue  # 累计值为 0 的记录是份额折算，不是分红
            prev = s.index[s.index < date]
            if len(prev) == 0:
                continue
            p = float(s.loc[prev[-1]])
            if p <= d:
                continue
            k = p / (p - d)
            factor.loc[factor.index >= date] *= k
            n_div += 1
            div_total += float(d)

    # ── 2) 份额折算因子（在分红修正后的价格上检测残留跳空）──
    adj = s * factor
    r = adj.pct_change(fill_method=None)
    n_jump = 0
    for date, v in r[r.abs() > jump_threshold].items():
        factor.loc[factor.index >= date] /= (1.0 + float(v))
        n_jump += 1

    # ── 3) 诊断：修正后的最大日收益 ──────────────────────────
    final_r = (s * factor).pct_change(fill_method=None)
    diag = {
        "n_div": n_div,
        "div_total": div_total,
        "n_jump": n_jump,
        "residual_max": float(final_r.abs().max()) if final_r.notna().any() else 0.0,
    }
    return factor, diag


_DIV_CACHE: dict[str, Optional[pd.DataFrame]] = {}


def load_or_fetch_dividends(symbol: str) -> Optional[pd.DataFrame]:
    """分红表进程内缓存（同一标的多次复权只拉一次）。"""
    if symbol not in _DIV_CACHE:
        _DIV_CACHE[symbol] = fetch_dividends(symbol)
    return _DIV_CACHE[symbol]


def build_adjusted_etf(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, dict]:
    """对单只 ETF 的原始 DataFrame 做后复权，返回 (复权后 DataFrame, 诊断)。"""
    threshold = JUMP_THRESHOLD_CYB if symbol in CYB_SYMBOLS else JUMP_THRESHOLD_DEFAULT
    div = load_or_fetch_dividends(symbol)
    close = df["close"].astype(float)
    factor, diag = build_adjustment_factor(close, div, threshold)

    out = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = out[c].astype(float) * factor
    return out, diag


def write_adjusted_cache(symbol: str, df_raw: pd.DataFrame, *, verbose: bool = True) -> dict:
    """生成 `{symbol}_hfq.csv` 并返回诊断。"""
    adj, diag = build_adjusted_etf(df_raw, symbol)
    CACHE_ETF_DIR.mkdir(parents=True, exist_ok=True)
    adj.to_csv(CACHE_ETF_DIR / f"{symbol}_hfq.csv")
    if verbose:
        print(f"  {symbol}: 分红 {diag['n_div']:2d} 次 / 累计 {diag['div_total']:8.3f}  "
              f"折算 {diag['n_jump']} 处  修正后最大 |r| = {diag['residual_max']:.3f}")
    return diag
