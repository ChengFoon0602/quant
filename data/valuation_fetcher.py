"""
valuation_fetcher.py — 估值因子拉取（日频快照）。

从 baostock 日线接口 query_history_k_data_plus 获取 PE/PB/PS/PCF 四类估值，
转 date × symbol 宽表，缓存到 data/cache_valuation/。

与财报因子（fundamental_fetcher.py）的语义区别:
  - 估值是日频快照（当日收盘价 × 最新财报），无 pubDate 公告延迟语义，
    价格当天已知即可用；+1 日交易延迟由回测引擎的 t→t+1 约定天然提供。
  - 使用 adjustflag="3"（不复权）：估值用真实市值口径（市场资本化用实际价格，
    非前复权价）。
  - 独立缓存目录，不污染 data/cache_fundamental/ 的 PIT 语义。

用法:
    python data/valuation_fetcher.py --subset 20   # 小样本校准
    python data/valuation_fetcher.py                # 全量（1625 只 ≈ 5-10 分钟）
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs

# 允许 `python data/valuation_fetcher.py` 直接运行时 import data.*（sys.path[0]=脚本目录）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALUATION_CACHE_DIR = Path(__file__).parent / "cache_valuation"
VALUATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALUATION_FIELDS = ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"


def _to_baostock_code(symbol: str) -> str:
    """6 位代码 → baostock 前缀格式。"""
    code_val = int(symbol)
    if (code_val >= 600000 and code_val <= 689999) or symbol.startswith(("6", "68")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def _cache_path(field: str) -> Path:
    return VALUATION_CACHE_DIR / f"{field}.csv"


def load_cached_valuation(field: str) -> pd.DataFrame | None:
    """从本地缓存读取估值字段宽表，未缓存返回 None。"""
    p = _cache_path(field)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date", dtype=str)
    df = df.astype(float)
    return df.sort_index().sort_index(axis=1)


def _records_to_wide(records: list[dict]) -> pd.DataFrame:
    """将 [{date, symbol, value}] 记录转为 date × symbol 宽表并去重排序。"""
    df_long = pd.DataFrame(records)
    df_long = df_long.drop_duplicates(subset=["date", "symbol"], keep="last")
    df_wide = df_long.pivot(index="date", columns="symbol", values="value")
    df_wide = df_wide.sort_index().sort_index(axis=1)
    df_wide.index.name = "date"
    df_wide.index = pd.to_datetime(df_wide.index)
    return df_wide


def fetch_valuation(
    symbols: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    verbose: bool = True,
    qps: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """逐只拉取估值字段，返回 {field: date×symbol 宽表}，并写缓存。

    单只一次 query_history_k_data_plus 拿全历史 → 每只 1 次查询。
    qps: 每查询最小间隔（避免触发 baostock 风控，见 fundamental_fetcher）。
    """
    bs.login()
    records: dict[str, list[dict]] = {f: [] for f in VALUATION_FIELDS}
    n_total = len(symbols)
    n_err = 0
    t0 = time.time()
    last_q_time = time.time()

    for i, sym in enumerate(symbols):
        # 限流
        elapsed = time.time() - last_q_time
        interval = 1.0 / qps
        if elapsed < interval:
            time.sleep(interval - elapsed)
        last_q_time = time.time()

        code = _to_baostock_code(sym)
        try:
            rs = bs.query_history_k_data_plus(
                code,
                ",".join(["date"] + VALUATION_FIELDS),
                start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )
            if rs.error_code == "0":
                fields_lower = [f.lower() for f in rs.fields]
                date_idx = fields_lower.index("date")
                while rs.next():
                    row = rs.get_row_data()
                    date_str = row[date_idx]
                    if not date_str:
                        continue
                    d = pd.Timestamp(date_str)
                    for f in VALUATION_FIELDS:
                        f_idx = fields_lower.index(f.lower())
                        val_str = row[f_idx]
                        if val_str and val_str != "":
                            try:
                                records[f].append({"date": d, "symbol": sym, "value": float(val_str)})
                            except ValueError:
                                pass
        except Exception:
            n_err += 1
        n_done = i + 1

        if n_done % 20 == 0:
            time.sleep(0.02)
        if verbose and n_done % 100 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 1
            eta = (n_total - n_done) / rate if rate > 0 else 0
            n_hit = sum(len(v) for v in records.values())
            print(f"  [{n_done}/{n_total}] 命中 {n_hit} 条 | {elapsed:.0f}s | "
                  f"ETA {eta/60:.0f}min ({rate:.0f} 只/s, {n_err} err)", flush=True)

    bs.logout()

    result: dict[str, pd.DataFrame] = {}
    for f in VALUATION_FIELDS:
        if not records[f]:
            print(f"  [WARN] {f}: 未命中任何数据")
            result[f] = pd.DataFrame()
            continue
        df_wide = _records_to_wide(records[f])
        df_wide.to_csv(_cache_path(f))
        if verbose:
            print(f"  {f}: {len(df_wide)} 日期 × {len(df_wide.columns)} 只股票 "
                  f"({df_wide.index[0].date()} → {df_wide.index[-1].date()})")
        result[f] = df_wide
    return result


def get_zz500_pit_symbols() -> list[str]:
    """PIT 中证500 全部历史成员（含退市/调出）的 6 位代码列表。"""
    from data.index_membership import load_membership
    mem = load_membership("zz500")
    return sorted(mem.columns.astype(str))


if __name__ == "__main__":
    symbols = get_zz500_pit_symbols()

    # --subset N：只拉前 N 只（小样本校准）
    subset = None
    for i, a in enumerate(sys.argv):
        if a == "--subset" and i + 1 < len(sys.argv):
            subset = int(sys.argv[i + 1])

    # --qps：限流（默认 3 q/s，单进程安全）
    qps = 3.0
    for i, a in enumerate(sys.argv):
        if a == "--qps" and i + 1 < len(sys.argv):
            qps = float(sys.argv[i + 1])

    if subset is not None:
        symbols = symbols[:subset]
        print(f"[subset={subset}] 样本股票池: {len(symbols)} 只")

    print(f"估值拉取股票池: {len(symbols)} 只 PIT zz500 成员 | qps={qps}")
    result = fetch_valuation(symbols, qps=qps)
    for f, df in result.items():
        if len(df) > 0:
            print(f"  {f}: 非空 {df.notna().sum().sum():,} | "
                  f"peTTM 样例 {df[df.columns[0]].dropna().iloc[0]:.3f}" if f == "peTTM" else
                  f"  {f}: 非空 {df.notna().sum().sum():,}")
    print("\n完成。")
