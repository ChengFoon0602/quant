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

from backtest.invariants import (  # 输入契约断言（零项目内依赖，不构成循环导入）
    InvariantViolation,
    assert_price_panel,
)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 内部工具 ──────────────────────────────────────────────

_logged_in = False

# 日线必需列（= baostock query_history_k_data_plus 的字段集）
_REQUIRED_COLS = ("open", "high", "low", "close", "volume", "amount")


def _assert_daily_frame(df: pd.DataFrame, symbol: str) -> None:
    """日线加载/下载返回前的契约校验。

    只断言**必然错误**（见 CLAUDE.md 工程铁律 E1）：
      - 必需列齐备；
      - close 是合法价格面板：索引有序且唯一、价格非负、非全 NaN。

    ⚠️ 不把 volume / amount 纳入价格断言——它们不是价格，且个别标的可能整列缺失。
    ⚠️ 允许 close 含孤立 NaN（停牌 / 未上市）。实测真实截面 NaN 占比约 50%，
       若在这里要求「无 NaN」会立刻全线误报。
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise InvariantViolation(f"{symbol}: 日线缺少必需列 {missing}")
    assert_price_panel(df[["close"]], name=f"{symbol}.close")

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

def _cache_path(symbol: str, adjust: str = "2") -> Path:
    """缓存路径，按复权方式区分，避免前/后复权数据互相覆盖。

    adjust="2"（前复权）沿用旧路径 {symbol}.csv（向后兼容存量缓存），
    其余复权方式用 {symbol}_adj{adjust}.csv 独立存储。
    """
    if adjust == "2":
        return CACHE_DIR / f"{symbol}.csv"
    return CACHE_DIR / f"{symbol}_adj{adjust}.csv"


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

    ⚠️ 复权方式选择（2026-09 修订）：
      - 前复权 ("2")：历史价随最新除权事件回溯调整，只适合单票时序回测，
        会污染截面因子与 ML 训练的时点可比性。
      - 后复权 ("1")：历史价相对恒定，适合截面比较与因子计算（推荐截面/ML 用）。
      - 不复权 ("3")：真实成交价，需配复权因子表才能算总回报。

    Returns
        标准化 DataFrame: open, high, low, close, volume, amount
    """
    cache_path = _cache_path(symbol, adjust=adjust)

    # 检查缓存覆盖范围：全量覆盖则直接返回，否则补全缺失的前后段
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        cached = cached.sort_index()
        cache_start = cached.index.min().strftime("%Y-%m-%d")
        cache_end = cached.index.max().strftime("%Y-%m-%d")
        if cache_start <= start and cache_end >= end:
            result = cached.loc[start:end] if len(cached) > 0 else cached
            _assert_daily_frame(result, symbol)
            return result
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
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date").sort_index()
            _assert_daily_frame(cached, symbol)
            return cached
        raise last_error

    # 解析返回数据
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date").sort_index()
            _assert_daily_frame(cached, symbol)
            return cached
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
    _assert_daily_frame(df, symbol)
    return df


def load_daily(symbol: str, adjust: str = "2") -> pd.DataFrame | None:
    """从本地缓存读取日线数据，未缓存则返回 None。

    Parameters
        adjust: 复权方式，与 download_daily 一致，决定读取哪个缓存文件。

    Notes
        返回前做契约校验（必需列 + close 价格面板，见 `_assert_daily_frame`），
        违反即 raise `InvariantViolation`，不返回可疑数据。
    """
    p = _cache_path(symbol, adjust=adjust)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    _assert_daily_frame(df, symbol)
    return df


def load_field_panel(
    symbols,
    fields: tuple[str, ...] = ("close",),
    start: str | None = None,
    end: str | None = None,
    min_rows: int = 100,
) -> dict[str, pd.DataFrame]:
    """从本地缓存构建多标的面板（每个字段一个 `DataFrame`）。

    统一了此前散落在各离线运行器里的「逐票读 cache → 拼宽表」逻辑。

    Parameters
    ----------
    symbols : Iterable[str]
        标的列表（顺序即最终列序，缺失标的自动跳过）。
    fields : tuple[str, ...], default ("close",)
        要取的字段，须属于 `open/high/low/close/volume/amount`。
    start, end : Optional[str]
        日期过滤（含端点）；为 None 不过滤。
    min_rows : int, default 100
        少于该行数的标的视为无效样本，跳过。⚠️ 以**缓存中的可得历史长度**判断，
        **不是**过滤后的窗口长度 —— 否则窄日期窗口会让所有标的都被判无效。
    start, end : Optional[str]
        日期过滤（含端点）；为 None 不过滤。

    Returns
    -------
    dict[str, pd.DataFrame]
        `{field: 面板}`，面板 index=date、columns=symbol，按日期升序。
        所有字段共用同一批标的。
    """
    missing = [f for f in fields if f not in _REQUIRED_COLS]
    if missing:
        raise ValueError(f"未知字段 {missing}；可用字段 {list(_REQUIRED_COLS)}")

    out: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    for sym in symbols:
        df = load_daily(str(sym))
        if df is None or not set(fields).issubset(df.columns):
            continue
        if len(df) < min_rows:      # 按可得历史长度判断，先于日期过滤
            continue
        sub = df
        if start is not None:
            sub = sub.loc[sub.index >= start]
        if end is not None:
            sub = sub.loc[sub.index <= end]
        if sub.empty:
            continue
        for f in fields:
            out[f][str(sym)] = sub[f]

    return {f: pd.DataFrame(v).sort_index() for f, v in out.items()}


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


# ── 扩展股票池 (PIT Universe 支持) ──────────────────────

def get_extended_stock_list() -> pd.DataFrame:
    """获取扩展股票池：CSI 300 + CSI 500 成分股（去重），含退市股。

    优先用 baostock 获取当前成分股，再通过 akshare 获取退市股列表补充。
    返回的 DataFrame 包含 symbol, name。
    注意: 真正的 PIT 过滤在 universe.py 中通过动态条件实现。
    """
    _ensure_login()

    symbols_set: set[str] = set()
    rows: list[dict] = []

    # CSI 300
    try:
        rs = bs.query_hs300_stocks()
        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            code = row[1].replace("sh.", "").replace("sz.", "")
            symbols_set.add(code)
            rows.append({"symbol": code, "name": row[2]})
    except Exception:
        pass

    # CSI 500
    try:
        rs = bs.query_zz500_stocks()
        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            code = row[1].replace("sh.", "").replace("sz.", "")
            if code not in symbols_set:
                symbols_set.add(code)
                rows.append({"symbol": code, "name": row[2]})
    except Exception:
        pass

    # 退市股（通过 akshare 补充）
    try:
        import akshare as ak
        sh_delist = ak.stock_info_sh_delist(indicator="终止上市公司")
        sz_delist = ak.stock_info_sz_delist(indicator="终止上市公司")
        for _, row_d in pd.concat([sh_delist, sz_delist]).iterrows():
            code = str(row_d.get("证券代码", "")).zfill(6)
            if len(code) != 6 or code in symbols_set:
                continue
            symbols_set.add(code)
            rows.append({"symbol": code, "name": row_d.get("证券简称", "")})
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["symbol", "name"])

    df = pd.DataFrame(rows).drop_duplicates(subset="symbol").reset_index(drop=True)
    print(f"扩展股票池: {len(df)} 只（CSI 300 + CSI 500 + 退市股）")
    return df


def sync_extended_universe(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    max_new: int = 200,
) -> tuple[list[str], list[str]]:
    """拉取扩展股票池的日线数据（增量——只拉取缓存中没有的新股票）。

    为避免一次性拉取过多导致 baostock 限流：
    - 优先拉取缓存中尚不存在的股票
    - 限制每次调用最多新拉取 max_new 只
    - 已有缓存的股票不做更新

    Returns
        (ok_list, fail_list)
    """
    stock_df = get_extended_stock_list()
    if stock_df.empty:
        return [], []

    all_symbols = stock_df["symbol"].tolist()

    # 检查哪些已有缓存
    cached = set()
    for p in CACHE_DIR.glob("*.csv"):
        cached.add(p.stem)

    new_symbols = [s for s in all_symbols if s not in cached]
    existing = len(all_symbols) - len(new_symbols)

    print(f"总计: {len(all_symbols)} 只, 已有缓存: {existing}, 需新拉: {len(new_symbols)}")
    print(f"本次最多新拉取: {max_new} 只")

    to_pull = new_symbols[:max_new]

    _ensure_login()

    ok, fail = [], []
    for i, sym in enumerate(to_pull):
        try:
            df = download_daily(sym, start=start, end=end)
            ok.append(sym)
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(to_pull)}] {sym} ✓  {len(df)} 条  ({len(ok)} ok, {len(fail)} fail)")
        except Exception as e:
            fail.append(sym)
            print(f"[{i+1}/{len(to_pull)}] {sym} ✗  {e}")

    bs.logout()
    print(f"\n扩展同步完成: {len(ok)} 成功, {len(fail)} 失败")
    print(f"总缓存: {existing + len(ok)} 只")
    if fail:
        print(f"失败列表: {fail}")
    return ok, fail
