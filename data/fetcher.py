"""
数据获取模块 — 封装 baostock + akshare，拉取 A 股日线行情并本地缓存。

数据源选择：
- 日线历史数据：baostock（免费、不需 key、反爬宽松）
- 成分股列表：优先 baostock，失败回退 akshare
- 增量更新：akshare（数据更新更及时）

baostock 代码格式：sh.600000 / sz.000001，缓存文件去掉前缀存为纯数字。
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs
import akshare as ak

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 内部工具 ──────────────────────────────────────────────

_logged_in = False

def _ensure_login():
    """确保 baostock 已登录（幂等），抑制重复登录输出。"""
    global _logged_in
    if _logged_in:
        return
    bs.login()
    _logged_in = True

def _add_prefix(symbol: str) -> str:
    """根据 6 位数字代码推断交易所前缀。"""
    code = int(symbol)
    if 600000 <= code <= 689999:
        return f"sh.{symbol}"
    elif (code >= 300000 and code <= 309999) or (code >= 0 and code <= 3999):
        return f"sz.{symbol}"
    else:
        # 通用规则：6 开头 = 上海，其余 = 深圳
        return f"sh.{symbol}" if symbol.startswith(("6", "68")) else f"sz.{symbol}"


# ── 成分股 ──────────────────────────────────────────────

def get_csi300_constituents() -> pd.DataFrame:
    """获取沪深 300 最新成分股列表。

    Returns
        DataFrame: symbol, name, in_date
    """
    _ensure_login()
    rs = bs.query_hs300_stocks()
    rows = []
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        rows.append({
            "symbol": row[1].replace("sh.", "").replace("sz.", ""),
            "name": row[2],
            "in_date": pd.Timestamp(row[0]) if row[0] else pd.NaT,
        })
    if rows:
        return pd.DataFrame(rows)
    # 回退 akshare
    df = ak.index_stock_cons(symbol="000300")
    df = df.rename(columns={"品种代码": "symbol", "品种名称": "name", "纳入日期": "in_date"})
    df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce")
    return df[["symbol", "name", "in_date"]]


# ── 日线数据 ──────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def download_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    adjust: str = "2",
) -> pd.DataFrame:
    """下载单只 A 股日线数据，自动缓存到本地 CSV。

    Parameters
        symbol: 6 位数字代码，如 "000001"
        start, end: 起止日期 YYYY-MM-DD
        adjust: "2"=前复权 / "1"=后复权 / "3"=不复权

    Returns
        标准化 DataFrame: open, high, low, close, volume, amount
    """
    cache_path = _cache_path(symbol)

    # 检查缓存覆盖范围：全量覆盖则直接返回，否则补全缺失的前后段
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        cached = cached.sort_index()
        cache_start = cached.index.min().strftime("%Y-%m-%d")
        cache_end = cached.index.max().strftime("%Y-%m-%d")
        if cache_start <= start and cache_end >= end:
            return cached.loc[start:end] if len(cached) > 0 else cached
        # 扩展到缓存未覆盖的范围（向前和向后都要补）
        start = min(start, cache_start)
        end = max(end, cache_end)

    _ensure_login()
    code = _add_prefix(symbol)

    # 重试循环：随机延迟 + 指数退避
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag=adjust,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 1.0 + random.uniform(0, 2.0) * (2 ** attempt)
                print(f"  [{symbol}] 第{attempt+1}次失败，{delay:.1f}s 后重试...")
                time.sleep(delay)
    else:
        if cache_path.exists():
            print(f"[WARN] {symbol} 全部重试失败，返回本地缓存: {last_error}")
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return cached.sort_index()
        raise last_error

    # 解析返回数据
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return cached.sort_index()
        raise RuntimeError(f"{symbol} 返回空数据，可能停牌或退市")

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])

    # baostock 返回字符串，转浮点；空值填 NaN
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("date").sort_index()

    # 与旧缓存合并
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        cached = cached.sort_index()
        df = pd.concat([cached[~cached.index.isin(df.index)], df]).sort_index()

    df.to_csv(cache_path)
    return df


def load_daily(symbol: str) -> pd.DataFrame | None:
    """从本地缓存读取日线数据，未缓存则返回 None。"""
    p = _cache_path(symbol)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    return df


# ── 批量同步 ──────────────────────────────────────────────

def sync_index(
    index_code: str = "000300",
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> tuple[list[str], list[str]]:
    """同步指数全部成分股日线数据（使用 baostock 批量登录，一次登录拉全部）。

    Returns
        (success_list, fail_list) — 成功和失败的 symbol 列表
    """
    if index_code != "000300":
        raise ValueError(f"不支持的指数代码: {index_code}")

    _ensure_login()
    constituents = get_csi300_constituents()
    symbols = constituents["symbol"].tolist()
    print(f"{index_code} 成分股数量: {len(symbols)}")

    ok, fail = [], []
    for i, sym in enumerate(symbols):
        try:
            df = download_daily(sym, start=start, end=end)
            ok.append(sym)
            print(f"[{i+1}/{len(symbols)}] {sym} ✓  {len(df)} 条日线")
        except Exception as e:
            fail.append(sym)
            print(f"[{i+1}/{len(symbols)}] {sym} ✗  {e}")

    bs.logout()
    print(f"\n完成: {len(ok)} 成功, {len(fail)} 失败")
    if fail:
        print(f"失败列表: {fail}")
    return ok, fail


# ── 可用性检查 ───────────────────────────────────────────

def cache_summary() -> pd.DataFrame:
    """扫描本地缓存，返回每只股票的记录数和日期范围。"""
    rows = []
    for p in sorted(CACHE_DIR.glob("*.csv")):
        df = pd.read_csv(p, parse_dates=["date"])
        rows.append({
            "symbol": p.stem,
            "rows": len(df),
            "start": df["date"].min(),
            "end": df["date"].max(),
        })
    return pd.DataFrame(rows)
