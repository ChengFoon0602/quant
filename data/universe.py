"""
universe.py — Point-in-Time (PIT) 股票池构造。

解决幸存者偏差：不依赖最新成分股列表，而是在每个交易日动态过滤
全市场 A 股，模拟历史时点的真实可投资 Universe。

过滤条件（按优先级）:
  1. 上市满 252 个交易日（剔除新股效应 / IPO 首日暴涨）
  2. 非 ST / *ST（剔除财务困境股）
  3. 非停牌（当日成交量 > 0，剔除无法交易的股票）
  4. 20 日均成交额排名前 300（流动性代理市值，模拟沪深 300 选股逻辑）
  5. 退市股在退市日后永久剔除

用法:
    from data.universe import build_dynamic_universe
    mask = build_dynamic_universe(close_matrix, amount_matrix, volume_matrix)
"""

import numpy as np
import pandas as pd


def build_dynamic_universe(
    close_matrix: pd.DataFrame,
    amount_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    st_flags: pd.DataFrame | None = None,
    n_top: int = 300,
    min_listed_days: int = 252,
    amount_lookback: int = 20,
) -> pd.DataFrame:
    """构建每日动态 Universe 布尔矩阵。

    Parameters
        close_matrix: index=date, columns=symbols, 收盘价
        amount_matrix: index=date, columns=symbols, 成交额
        volume_matrix: index=date, columns=symbols, 成交量
        listing_dates: symbol → 上市日期 (pd.Timestamp)
        delist_dates: symbol → 退市日期（最后交易日），None 表示未退市
        st_flags: index=date, columns=symbols, True=当日ST
        n_top: 按成交额排名选取的股票数量
        min_listed_days: 最低上市天数
        amount_lookback: 成交额排名回看天数

    Returns
        pd.DataFrame: bool，index=date, columns=symbols，
        True=该日该股票在 Universe 内
    """
    dates = close_matrix.index
    symbols = close_matrix.columns

    # ── Filter 1: 上市天数 > min_listed_days ──
    listing_mask = pd.DataFrame(True, index=dates, columns=symbols)
    if listing_dates:
        for sym, list_date in listing_dates.items():
            if sym not in symbols:
                continue
            if pd.isna(list_date):
                continue
            # 上市满 min_listed_days 个交易日后才纳入
            # 用交易日计数而非日历日
            eligible_date = _find_nth_trading_day(list_date, dates, min_listed_days)
            if eligible_date is not None:
                listing_mask.loc[dates < eligible_date, sym] = False

    # ── Filter 2: 非 ST ──
    st_mask = pd.DataFrame(True, index=dates, columns=symbols)
    if st_flags is not None:
        common_syms = symbols.intersection(st_flags.columns)
        common_dates = dates.intersection(st_flags.index)
        st_mask.loc[common_dates, common_syms] = ~st_flags.loc[common_dates, common_syms]

    # ── Filter 3: 非停牌 (volume > 0) ──
    suspended = volume_matrix <= 0

    # ── Filter 4: 退市日后剔除 ──
    delisted = pd.DataFrame(False, index=dates, columns=symbols)
    if delist_dates:
        for sym, delist_date in delist_dates.items():
            if sym not in symbols or pd.isna(delist_date):
                continue
            # 退市日当天仍可交易（最后交易日），次日及之后不可交易
            delisted.loc[dates > delist_date, sym] = True

    # ── Filter 5: 20 日均成交额排名前 n_top ──
    amount_rank = pd.DataFrame(False, index=dates, columns=symbols)
    if amount_matrix is not None and len(amount_matrix) > amount_lookback:
        roll_amount = amount_matrix.rolling(amount_lookback, min_periods=max(5, amount_lookback // 4)).mean()
        for i, d in enumerate(dates):
            amt = roll_amount.loc[d].dropna()
            if len(amt) < n_top:
                continue
            # 取成交额最大的 n_top 只
            top_syms = amt.nlargest(n_top).index
            amount_rank.loc[d, top_syms] = True

    # ── 合成 ──
    universe = (
        listing_mask
        & st_mask
        & ~suspended
        & ~delisted
        & amount_rank
    )

    return universe


def _find_nth_trading_day(
    list_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    n: int,
) -> pd.Timestamp | None:
    """找到上市后第 n 个交易日。"""
    future_dates = trading_dates[trading_dates >= list_date]
    if len(future_dates) >= n:
        return future_dates[n - 1]
    return None


def get_listing_info() -> tuple[dict, dict]:
    """从 akshare 获取全市场 A 股上市/退市日期。

    Returns
        (listing_dates, delist_dates): 两个 dict，symbol → pd.Timestamp
    """
    import akshare as ak

    listing_dates: dict[str, pd.Timestamp] = {}
    delist_dates: dict[str, pd.Timestamp] = {}

    try:
        # 沪深 A 股列表（含退市）
        df = ak.stock_info_a_code_name()
        for _, row in df.iterrows():
            code = str(row.get("code", "")).zfill(6)
            if len(code) != 6:
                continue
            listing_dates[code] = pd.Timestamp(row.get("listing_date", pd.NaT))

    except Exception as e:
        print(f"[WARN] akshare stock_info_a_code_name 失败: {e}")
        # 回退：从缓存文件推断上市日期（第一根 K 线日期）
        print("[INFO] 从本地缓存推断上市日期...")

    # 退市列表
    try:
        sh_delist = ak.stock_info_sh_delist(symbol="全部")
        sz_delist = ak.stock_info_sz_delist(symbol="终止上市公司")

        # 上海: 公司代码 + 暂停上市日期 + 上市日期
        n_delist = 0
        for _, row in sh_delist.iterrows():
            code = str(row.get("公司代码", "")).zfill(6)
            if len(code) != 6:
                continue
            d = row.get("暂停上市日期", None)
            if d and not pd.isna(pd.Timestamp(d)):
                delist_dates[code] = pd.Timestamp(d)
                n_delist += 1
            # 退市股上市日期（stock_info_a_code_name 不含退市股）
            if code not in listing_dates:
                ld = row.get("上市日期", None)
                if ld and not pd.isna(pd.Timestamp(ld)):
                    listing_dates[code] = pd.Timestamp(ld)

        for _, row in sz_delist.iterrows():
            code = str(row.get("证券代码", "")).zfill(6)
            if len(code) != 6:
                continue
            d = row.get("终止上市日期", None)
            if d and not pd.isna(pd.Timestamp(d)):
                delist_dates[code] = pd.Timestamp(d)
                n_delist += 1
            if code not in listing_dates:
                ld = row.get("上市日期", None)
                if ld and not pd.isna(pd.Timestamp(ld)):
                    listing_dates[code] = pd.Timestamp(ld)

        print(f"[INFO] 退市股: {n_delist} 只 (上交所{len(sh_delist)} + 深交所{len(sz_delist)}原始)")
    except Exception as e:
        print(f"[WARN] akshare 退市列表获取失败: {e}")

    return listing_dates, delist_dates


def build_st_flag_matrix(
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """构建每日 ST 标记矩阵。

    简化方案：通过 akshare 获取历史 ST 记录，
    或简单检查股票名称是否含 "ST"。
    当前回退：默认全部非 ST（False）。

    Returns
        pd.DataFrame: index=date, columns=symbols, True=该日ST
    """
    # 当前版本：默认无 ST
    # 后续版本可通过 akshare stock_zh_a_st_hist 获取完整 ST 历史
    print("[INFO] ST 标记暂用默认（全部非 ST），后续版本接入 akshare ST 历史数据。")
    return pd.DataFrame(False, index=dates, columns=symbols)
