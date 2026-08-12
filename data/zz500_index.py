"""
zz500_index.py — 中证 500 指数日线行情拉取与缓存。

P1 多头增强需要以中证500指数本身为基准（算超额收益 / 跟踪误差 / 信息比率），
但 data/fetcher.py 的 sync_index 只支持沪深 300（000300）。本模块补拉 sh.000905。

字段: date, open, high, low, close, volume（close 为价格指数点位，非全收益）
缓存: data/cache_index/sh000905.csv（不入 git）

用法:
    python data/zz500_index.py          # 拉取并缓存
    python -c "from data.zz500_index import load_zz500_index; df = load_zz500_index(); print(len(df))"
"""

import random
import time
from pathlib import Path

import pandas as pd
import baostock as bs

INDEX_CODE = "sh.000905"      # baostock 格式：中证500指数
CACHE_DIR = Path(__file__).parent / "cache_index"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_logged_in = False


def _ensure_login():
    global _logged_in
    if _logged_in:
        return
    bs.login()
    _logged_in = True


def _cache_path() -> Path:
    return CACHE_DIR / "sh000905.csv"


def download_index(start: str = "2010-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """下载中证500指数日线（增量：缓存已覆盖则直接返回）。"""
    cache_path = _cache_path()

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date").sort_index()
        cache_start = cached.index.min().strftime("%Y-%m-%d")
        cache_end = cached.index.max().strftime("%Y-%m-%d")
        if cache_start <= start and cache_end >= end:
            return cached.loc[start:end]
        start = min(start, cache_start)
        end = max(end, cache_end)

    _ensure_login()
    last_error = None
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                INDEX_CODE,
                "date,open,high,low,close,volume",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3",
            )
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(1.0 + random.uniform(0, 2.0))
    else:
        raise RuntimeError(f"指数行情拉取失败: {last_error}")

    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise RuntimeError("sh.000905 返回空数据")

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("date").sort_index()

    # 与旧缓存合并
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date").sort_index()
        df = pd.concat([cached[~cached.index.isin(df.index)], df]).sort_index()

    df.to_csv(cache_path)
    print(f"中证500指数缓存: {len(df)} 条 ({df.index.min().date()} ~ {df.index.max().date()})")
    return df


def load_zz500_index() -> pd.DataFrame | None:
    """从缓存读取中证500指数日线，未缓存返回 None。"""
    cache_path = _cache_path()
    if not cache_path.exists():
        return None
    return pd.read_csv(cache_path, parse_dates=["date"], index_col="date").sort_index()


if __name__ == "__main__":
    df = download_index()
    print(df["close"].describe())
