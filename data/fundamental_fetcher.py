"""
fundamental_fetcher.py — 基本面数据拉取模块（baostock 季频财务数据）。

从 baostock 获取季度财务指标，按实际公告日 (pubDate) 索引（PIT-safe），
缓存到 data/cache_fundamental/ 供 signals/fundamental/ 使用。

baostock 季频 API（均从 2007 年起）:
  - query_growth_data()   → YOYAsset（总资产同比增长率）等，含 pubDate
  - query_cash_flow_data() → CFOToNP（经营现金流 / 净利润）等，含 pubDate
  - query_profit_data()    → netProfit, ROE 等
  - query_balance_data()   → 偿债能力指标
  - query_operation_data() → 营运能力指标
  - query_dupont_data()    → 杜邦分析

PIT 策略:
  - 优先使用 baostock 返回的 pubDate（实际财报公告日）
  - pubDate 为空时回退到法定截止日（保守但仍 PIT-safe）

每个因子存为独立 CSV（date × symbol 宽表），索引 = pubDate。
前向填充由 signals/fundamental/factors.py 负责。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs

FUNDAMENTAL_CACHE_DIR = Path(__file__).parent / "cache_fundamental"
FUNDAMENTAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── PIT 日期回退 ──────────────────────────────────────────────

def _statutory_deadline(year: int, quarter: int) -> pd.Timestamp:
    """法定披露截止日（pubDate 缺失时的回退方案）。"""
    if quarter == 1:
        return pd.Timestamp(year=year, month=5, day=1)
    elif quarter == 2:
        return pd.Timestamp(year=year, month=9, day=1)
    elif quarter == 3:
        return pd.Timestamp(year=year, month=11, day=1)
    elif quarter == 4:
        return pd.Timestamp(year=year + 1, month=5, day=1)
    raise ValueError(f"quarter must be 1-4, got {quarter}")


# ── 数据拉取核心 ──────────────────────────────────────────────

def _to_baostock_code(symbol: str) -> str:
    """6 位代码 → baostock 前缀格式。"""
    code_val = int(symbol)
    if (code_val >= 600000 and code_val <= 689999) or symbol.startswith(("6", "68")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def _records_to_wide(records: list[dict], field: str) -> pd.DataFrame:
    """将 [{date, symbol, value}] 记录转为 date × symbol 宽表并去重排序。"""
    df_long = pd.DataFrame(records)
    df_long = df_long.drop_duplicates(subset=["date", "symbol"], keep="last")
    df_wide = df_long.pivot(index="date", columns="symbol", values="value")
    df_wide = df_wide.sort_index().sort_index(axis=1)
    df_wide.index.name = "date"
    df_wide.index = pd.to_datetime(df_wide.index)
    return df_wide


def _save_checkpoint(records: list[dict], field: str):
    """增量写 CSV checkpoint（与已有缓存合并）。"""
    new_df = _records_to_wide(records, field)
    cached = load_cached_field(field)
    if cached is not None:
        all_cols = cached.columns.union(new_df.columns)
        all_idx = cached.index.union(new_df.index)
        merged = pd.DataFrame(index=all_idx, columns=all_cols, dtype=float)
        for df_src in [cached, new_df]:
            for col in df_src.columns:
                merged.loc[df_src.index, col] = df_src[col].values
        new_df = merged
    new_df.to_csv(_cache_path(field))


def _query_field_with_date(
    bs_code: str,
    year: int,
    quarter: int,
    api_func,
    field: str,
) -> tuple[float | None, pd.Timestamp | None]:
    """查询单只股票单个季度的字段值 + 发布日期。

    Returns
        (value, pit_date) — value 为 None 表示无数据；
        pit_date 优先用 pubDate，缺失时回退 statutory deadline。
    """
    try:
        rs = api_func(code=bs_code, year=year, quarter=quarter)
        field_lower = field.lower()
        # 字段索引（所有行共享同一个 rs.fields）
        fields_list = list(rs.fields)
        fields_lower = [f.lower() for f in fields_list]
        if field_lower not in fields_lower:
            return None, None
        val_idx = fields_lower.index(field_lower)
        pub_idx = fields_lower.index("pubdate") if "pubdate" in fields_lower else None

        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            val_str = row[val_idx]
            if not val_str or val_str == "":
                continue
            value = float(val_str)
            # pubDate → PIT 有效日 = pubDate + 1 天
            # 财报通常在收盘后发布，pubDate 当天交易时段内数据不可用；
            # +1 天确保信号只在数据真正公开后才使用（保守 ≈ −1 天信号延迟）
            if pub_idx is not None:
                pub_str = row[pub_idx]
                if pub_str and pub_str != "":
                    pit_date = pd.Timestamp(pub_str) + pd.Timedelta(days=1)
                else:
                    pit_date = _statutory_deadline(year, quarter) + pd.Timedelta(days=1)
            else:
                pit_date = _statutory_deadline(year, quarter) + pd.Timedelta(days=1)
            return value, pit_date
        return None, None
    except Exception:
        return None, None


def fetch_fundamental_field(
    symbols: list[str],
    field: str,
    api_func,
    years: range | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """批量拉取指定财务字段，返回 date(symbols) PIT 宽表。

    Parameters
        symbols: 6 位股票代码列表
        field: baostock 字段名，如 "YOYAsset", "CFOToNP"
        api_func: baostock 查询函数，如 bs.query_growth_data
        years: 年份范围，默认 2007–2025
        verbose: 打印进度

    Returns
        DataFrame: index=pubDate, columns=symbols, values=field
    """
    bs.login()
    if years is None:
        years = range(2007, 2026)

    # 构造所有查询任务
    tasks = []
    for sym in symbols:
        bs_code = _to_baostock_code(sym)
        for y in years:
            for q in (1, 2, 3, 4):
                tasks.append((bs_code, sym, y, q))

    records: list[dict] = []  # [{date, symbol, value}]
    n_done = 0
    n_hit = 0     # 命中计数（非空值）
    n_err = 0     # 异常计数
    n_total = len(tasks)
    t0 = time.time()

    for bs_code, sym, y, q in tasks:
        try:
            val, pit_date = _query_field_with_date(bs_code, y, q, api_func, field)
        except Exception:
            val, pit_date = None, None
            n_err += 1

        if val is not None and not np.isnan(val) and pit_date is not None:
            records.append({"date": pit_date, "symbol": sym, "value": val})
            n_hit += 1
        n_done += 1

        # 每 200 次查询打印进度 + 小幅延迟避免限流
        if n_done % 200 == 0:
            time.sleep(0.02)
        if verbose and n_done % 200 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 1
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{n_total}] {n_done/n_total*100:.1f}%  "
                  f"命中 {n_hit} 条 | {elapsed:.0f}s | ETA {eta/60:.0f}min  "
                  f"({rate:.0f} q/s, {n_err} err)", flush=True)

        # 每 10000 条记录增量写 CSV（防止中途崩溃丢数据）
        if len(records) > 0 and len(records) % 10000 == 0:
            _save_checkpoint(records, field)

    bs.logout()

    if not records:
        print(f"  [WARN] {field}: 未命中任何数据")
        return pd.DataFrame()

    df_wide = _records_to_wide(records, field)

    if verbose:
        print(f"  {field}: {len(df_wide)} 个 PIT 日期 × {len(df_wide.columns)} 只股票 "
              f"({df_wide.index[0].date()} → {df_wide.index[-1].date()})")
    return df_wide


# ── 按接口批处理（多字段一次查询）──────────────────────────────

# 6 大财报接口 × 字段分组。baostock 单次查询返回整行（同接口所有字段），
# 逐字段重查浪费 ~6 倍，故按接口一次 pass 提取全部所需字段。
INTERFACE_GROUPS: dict[str, tuple[object, list[str]]] = {
    "profit":    (bs.query_profit_data,     ["roeAvg", "npMargin", "gpMargin", "epsTTM"]),
    "growth":    (bs.query_growth_data,     ["YOYNI", "YOYPNI", "YOYEPSBasic", "YOYAsset"]),
    "cash_flow": (bs.query_cash_flow_data,  ["CFOToNP", "CFOToOR", "CFOToGr", "CAToAsset", "ebitToInterest"]),
    "operation": (bs.query_operation_data,  ["NRTurnRatio", "INVTurnRatio", "CATurnRatio", "AssetTurnRatio"]),
    "balance":   (bs.query_balance_data,    ["liabilityToAsset", "currentRatio", "cashRatio"]),
    "dupont":    (bs.query_dupont_data,     ["dupontROE"]),
}


def fetch_interface(
    symbols: list[str],
    api_func,
    fields: list[str],
    years: range | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """按接口批处理拉取多个财务字段（每 (code, y, q) 只查询一次）。

    Parameters
        symbols: 6 位股票代码列表
        api_func: baostock 查询函数，如 bs.query_profit_data
        fields: 该接口下要提取的字段名列表
        years: 年份范围，默认 2007–2025
        verbose: 打印进度

    Returns
        dict: {field: date×symbol PIT 宽表}，并逐个写缓存到 cache_fundamental/{field}.csv
    """
    bs.login()
    if years is None:
        years = range(2007, 2026)

    tasks = []
    for sym in symbols:
        bs_code = _to_baostock_code(sym)
        for y in years:
            for q in (1, 2, 3, 4):
                tasks.append((bs_code, sym, y, q))

    records: dict[str, list[dict]] = {f: [] for f in fields}
    n_done = 0
    n_err = 0
    n_total = len(tasks)
    t0 = time.time()

    for bs_code, sym, y, q in tasks:
        try:
            rs = api_func(code=bs_code, year=y, quarter=q)
            fields_list = list(rs.fields)
            fields_lower = [f.lower() for f in fields_list]
            pub_idx = fields_lower.index("pubdate") if "pubdate" in fields_lower else None

            if rs.error_code == "0":
                while rs.next():
                    row = rs.get_row_data()
                    # PIT 有效日 = pubDate + 1 天；缺失回退法定截止日 + 1 天
                    if pub_idx is not None:
                        pub_str = row[pub_idx]
                        if pub_str and pub_str != "":
                            pit_date = pd.Timestamp(pub_str) + pd.Timedelta(days=1)
                        else:
                            pit_date = _statutory_deadline(y, q) + pd.Timedelta(days=1)
                    else:
                        pit_date = _statutory_deadline(y, q) + pd.Timedelta(days=1)

                    for f in fields:
                        f_lower = f.lower()
                        if f_lower not in fields_lower:
                            continue
                        val_str = row[fields_lower.index(f_lower)]
                        if val_str and val_str != "":
                            try:
                                records[f].append({"date": pit_date, "symbol": sym, "value": float(val_str)})
                            except ValueError:
                                pass
        except Exception:
            n_err += 1
        n_done += 1

        if n_done % 200 == 0:
            time.sleep(0.02)
        if verbose and n_done % 200 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 1
            eta = (n_total - n_done) / rate if rate > 0 else 0
            n_hit = sum(len(v) for v in records.values())
            print(f"  [{n_done}/{n_total}] {n_done/n_total*100:.1f}%  命中 {n_hit} 条 | "
                  f"{elapsed:.0f}s | ETA {eta/60:.0f}min ({rate:.0f} q/s, {n_err} err)", flush=True)

        # checkpoint：每字段每 10000 条增量写盘（防中断）
        for f in fields:
            if len(records[f]) > 0 and len(records[f]) % 10000 == 0:
                _save_checkpoint(records[f], f)

    bs.logout()

    result: dict[str, pd.DataFrame] = {}
    for f in fields:
        if not records[f]:
            print(f"  [WARN] {f}: 未命中任何数据")
            result[f] = pd.DataFrame()
            continue
        df_wide = _records_to_wide(records[f], f)
        df_wide.to_csv(_cache_path(f))
        if verbose:
            print(f"  {f}: {len(df_wide)} 个 PIT 日期 × {len(df_wide.columns)} 只股票 "
                  f"({df_wide.index[0].date()} → {df_wide.index[-1].date()})")
        result[f] = df_wide
    return result


def get_zz500_pit_symbols() -> list[str]:
    """PIT 中证500 全部历史成员（含退市/调出）的 6 位代码列表。"""
    from data.index_membership import load_membership
    mem = load_membership("zz500")
    return sorted(mem.columns.astype(str))


# ── 缓存接口 ──────────────────────────────────────────────────

def _cache_path(field: str) -> Path:
    return FUNDAMENTAL_CACHE_DIR / f"{field}.csv"


def load_cached_field(field: str) -> pd.DataFrame | None:
    """从本地缓存读取财务字段宽表，未缓存返回 None。"""
    p = _cache_path(field)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date", dtype=str)
    df = df.astype(float)
    return df.sort_index().sort_index(axis=1)


def fetch_and_cache(
    symbols: list[str],
    field: str,
    api_func,
    years: range | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """拉取 + 缓存财务字段。已有缓存且不强制刷新则直接返回。
    新数据会与旧缓存合并（增量更新）。

    Returns
        DataFrame: index=pubDate, columns=symbols
    """
    cached = load_cached_field(field)
    if cached is not None and not force_refresh:
        print(f"[{field}] 缓存已存在 ({len(cached)} 日期 × {len(cached.columns)} 股票)，"
              f"跳过拉取。用 force_refresh=True 强制重拉。")
        return cached

    print(f"[{field}] 开始拉取 {len(symbols)} 只股票...")
    new_df = fetch_fundamental_field(symbols, field, api_func, years)

    # 与旧缓存合并
    if cached is not None and not new_df.empty:
        all_cols = cached.columns.union(new_df.columns)
        all_idx = cached.index.union(new_df.index)
        merged = pd.DataFrame(index=all_idx, columns=all_cols, dtype=float)
        for df_src in [cached, new_df]:
            for col in df_src.columns:
                merged.loc[df_src.index, col] = df_src[col].values
        new_df = merged
    elif cached is not None:
        new_df = cached

    if new_df.empty:
        print(f"  [WARN] {field}: 结果为空，不写入缓存")
        return new_df

    new_df.to_csv(_cache_path(field))
    print(f"[{field}] 已缓存: {len(new_df)} 日期 × {len(new_df.columns)} 股票")
    return new_df.sort_index().sort_index(axis=1)


# ── 便捷入口 ──────────────────────────────────────────────────

def fetch_asset_growth(symbols: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """YOYAsset — 总资产同比增长率（单位: %）。"""
    return fetch_and_cache(symbols, "YOYAsset", bs.query_growth_data, force_refresh=force_refresh)


def fetch_earnings_quality(symbols: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """CFOToNP — 经营活动现金流 / 净利润。"""
    return fetch_and_cache(symbols, "CFOToNP", bs.query_cash_flow_data, force_refresh=force_refresh)


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv

    # 支持 --interface {name}：只拉一个接口组（后台分段跑）
    interface_arg = None
    for i, a in enumerate(sys.argv):
        if a == "--interface" and i + 1 < len(sys.argv):
            interface_arg = sys.argv[i + 1]

    symbols = get_zz500_pit_symbols()
    print(f"PIT zz500 股票池: {len(symbols)} 只")

    if interface_arg is not None:
        if interface_arg not in INTERFACE_GROUPS:
            print(f"[ERROR] 未知接口 {interface_arg!r}. 可选: {list(INTERFACE_GROUPS)}")
            sys.exit(1)
        api_func, fields = INTERFACE_GROUPS[interface_arg]
        print(f"\n=== {interface_arg} 接口: {fields} ===")
        result = fetch_interface(symbols, api_func, fields)
        for f, df in result.items():
            if len(df) > 0:
                print(f"  {f}: {len(df)} 日期 × {len(df.columns)} 股票, "
                      f"非空 {df.notna().sum().sum():,}")
        print(f"\n完成 {interface_arg}。")
        sys.exit(0)

    # 默认路径：两个便捷入口（兼容旧用法）
    print("\n=== Asset Growth (YOYAsset) ===")
    df_ag = fetch_asset_growth(symbols, force_refresh=force)
    if len(df_ag) > 0:
        print(f"  覆盖日期: {df_ag.index[0].date()} → {df_ag.index[-1].date()}")
        print(f"  非空值: {df_ag.notna().sum().sum():,}")

    print("\n=== Earnings Quality (CFOToNP) ===")
    df_eq = fetch_earnings_quality(symbols, force_refresh=force)
    if len(df_eq) > 0:
        print(f"  覆盖日期: {df_eq.index[0].date()} → {df_eq.index[-1].date()}")
        print(f"  非空值: {df_eq.notna().sum().sum():,}")

    print("\n完成。")
