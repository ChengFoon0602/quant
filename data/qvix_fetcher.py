"""
data/qvix_fetcher.py — 期权隐含波动率指数（QVIX）数据层。

## 这是什么

QVIX 是基于 CBOE VIX 方法论、由**期权论坛（optbbs.com）**计算的 A 股期权
隐含波动率指数。中证指数公司原「中国波指 iVIX」已于 2018-02 停止发布，
QVIX 是当前业内事实上的替代口径。

⚠️ **非官方指数**。数据源为第三方论坛，口径未经交易所审计，使用前必须
通过 `audit_consistency()` 与已实现波动率做一致性自检（见 report G1 关卡）。

## 覆盖范围（2026-11 实测）

| 序列 | 起始 | 样本 | 说明 |
|---|---|---|---|
| 50ETF  | 2015-02-09 | 2851 | 唯一覆盖 ≥10 年，主样本 |
| 300ETF | 2019-12-23 | 1663 | 300ETF 期权 2019-12 上市 |
| 500ETF | 2022-09-19 |  999 | **仅 4 年，不足以做 walk-forward** |
| 300index / 1000index | 2019-12 / 2022-07 | — | 含 0.00 异常值，本模块剔除 |

## 为什么不自算 VIX

`akshare.option_finance_board` 仅返回**当日快照**（无历史期权链）；
`option_sse_codes_sina` / `option_sse_expire_day_sina` 在当前 akshare 版本已失效。
逐合约下载后聚合需数千条序列，工程成本与验证成本均不可接受。
结论：**A 股无法低成本复原 CBOE 算法**，只能使用第三方成品指数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"

# 序列名 → akshare 函数名。key 是本地缓存与报告中使用的中性命名。
QVIX_SERIES: Dict[str, str] = {
    "50etf": "index_option_50etf_qvix",
    "300etf": "index_option_300etf_qvix",
    "500etf": "index_option_500etf_qvix",
    "300index": "index_option_300index_qvix",
    "1000index": "index_option_1000index_qvix",
}

#: 各序列的期权标的首个交易日（用于报告中标注「样本起点由上市日决定」）
LISTING_NOTE: Dict[str, str] = {
    "50etf": "上证50ETF期权 2015-02-09 上市",
    "300etf": "沪深300ETF期权 2019-12-23 上市",
    "500etf": "中证500ETF期权 2022-09-19 上市",
    "300index": "沪深300股指期权 2019-12-23 上市",
    "1000index": "中证1000股指期权 2022-07-22 上市",
}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"qvix_{name}.csv"


def download_qvix(name: str) -> pd.DataFrame:
    """从 akshare 下载单个 QVIX 序列（原始形态，含 NaN 与重复日期）。"""
    if name not in QVIX_SERIES:
        raise KeyError(f"未知 QVIX 序列 {name!r}，可选：{sorted(QVIX_SERIES)}")
    import akshare as ak

    fn = getattr(ak, QVIX_SERIES[name])
    df = fn()
    df["date"] = pd.to_datetime(df["date"])
    return df


def sync_qvix(name: str, *, verbose: bool = True) -> pd.DataFrame:
    """下载并写入本地缓存（CSV，与项目其他 cache 同级扁平结构）。"""
    df = download_qvix(name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(name), index=False)
    if verbose:
        s = clean_qvix(df)
        print(f"QVIX[{name}] cached ✓  n={len(s)}  "
              f"{s.index.min().date()} ~ {s.index.max().date()}")
    return df


def load_qvix(name: str, *, auto_download: bool = True) -> pd.Series:
    """读取清洗后的 QVIX 收盘序列（index=date, 值为指数点）。

    Parameters
    ----------
    auto_download : bool, default True
        缓存缺失时自动下载。置 False 则返回空 Series（离线复现用）。
    """
    p = _cache_path(name)
    if not p.exists():
        if not auto_download:
            return pd.Series(dtype=float, name=name)
        df = sync_qvix(name, verbose=False)
    else:
        df = pd.read_csv(p)
    return clean_qvix(df).rename(name)


def clean_qvix(df: pd.DataFrame) -> pd.Series:
    """清洗为可用的收盘序列。

    处理三类已知缺陷：
      1. 重复日期（akshare 原始数据偶有）；
      2. 非正值 —— 300index / 1000index 实测存在 0.00，属数据缺失而非真值；
      3. 尾部未来日期空行（原始文件按交易日历预生成，末几行为 NaN）。
    """
    if df.empty:
        return pd.Series(dtype=float)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")

    # 只保留首列出现过的字段，避免 open/high/low 缺失时报错
    s = d.set_index("date")["close"]
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    s = s[s > 0.0]
    return s.sort_index()


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """年化已实现波动率（%，与 VIX 点数量纲一致）。

    RV = std(log return) × sqrt(252) × 100
    """
    r = np.log(close.astype(float) / close.astype(float).shift(1))
    return r.rolling(window).std() * np.sqrt(252.0) * 100.0


def audit_consistency(
    iv: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> Dict[str, float]:
    """G1 关卡：QVIX 与标的已实现波动率的一致性审计。

    返回指标及判定。判据见 `assert_qvix_usable()`。
    """
    rv = realized_vol(close, window)
    df = pd.DataFrame({"iv": iv, "rv": rv}).dropna()
    if len(df) < 100:
        raise ValueError(f"一致性审计样本不足（n={len(df)} < 100）")

    spread = df["iv"] - df["rv"]
    return {
        "n": float(len(df)),
        "pearson": float(df["iv"].corr(df["rv"])),
        "spearman": float(df["iv"].corr(df["rv"], method="spearman")),
        "iv_mean": float(df["iv"].mean()),
        "iv_median": float(df["iv"].median()),
        "rv_median": float(df["rv"].median()),
        "spread_mean": float(spread.mean()),
        "spread_median": float(spread.median()),
        "spread_positive_ratio": float((spread > 0).mean()),
        "iv_max": float(df["iv"].max()),
        "iv_min": float(df["iv"].min()),
        "rv_max": float(df["rv"].max()),
    }


#: G1 判据（写死在代码里，改动即改变「数据是否可用」的结论）
G1_MIN_PEARSON: float = 0.60
G1_MIN_SPREAD_MEDIAN: float = 0.0


def assert_qvix_usable(stats: Dict[str, float], name: str = "") -> None:
    """G1 硬门槛：不达标直接中断，不降级为告警（工程铁律 E1）。"""
    if stats["pearson"] < G1_MIN_PEARSON:
        raise AssertionError(
            f"QVIX[{name}] 与已实现波动率相关性 {stats['pearson']:.4f} "
            f"< 门槛 {G1_MIN_PEARSON} —— 数据不可信，研究终止"
        )
    if stats["spread_median"] < G1_MIN_SPREAD_MEDIAN:
        raise AssertionError(
            f"QVIX[{name}] 方差风险溢价中位数 {stats['spread_median']:.2f} "
            f"< 0 —— 隐含波动率系统性低于已实现波动率，口径存疑"
        )


def load_underlying_close(symbol: str = "510050") -> pd.Series:
    """读取 QVIX 标的 ETF 的收盘价（复用项目 ETF 数据层）。"""
    from data.etf_fetcher import load_etf_daily

    df = load_etf_daily(symbol)
    if df is None:
        raise FileNotFoundError(
            f"ETF {symbol} 未缓存。先运行 `python -c \"from data.etf_fetcher import "
            f"sync_all_etfs; sync_all_etfs(['{symbol}'])\"`"
        )
    s = df["close"].astype(float)
    return s[~s.index.duplicated()].sort_index()
