"""
signals/fundamental/factors.py — 基本面因子计算。

从 data/cache_fundamental/ 读取缓存的季频财务数据，前向填充到日频，
对齐到 PIT 面板的 (date × symbol) 格式，直接可喂入 factor_tensor。

PIT 语义:
  - baostock 返回 pubDate（实际公告日），+1 天作为有效日（财报收盘后发布）
  - 前向填充: 每个交易日使用"最近一次有效日前已发布的财报数据"
  - 因子 mask 到 member_daily，非指数成员日置 NaN

用法:
    from signals.fundamental.factors import compute_asset_growth, compute_earnings_quality

    close, _, member = load_pit_panel("zz500")
    ag = compute_asset_growth(close)   # DataFrame(date × symbol)
    eq = compute_earnings_quality(close)
"""

import pandas as pd
from data.fundamental_fetcher import load_cached_field


def _forward_fill_to_daily(
    fund_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """将 PIT-date 索引的财务数据前向填充到日频。

    fund_df: index=PIT deadline dates, columns=symbols
    daily_index: 目标日频日期索引（来自 close_matrix）

    前向填充：某日在 fund_df 中最近一个 <= 该日的 PIT date 的值。
    PIT date 当天的数据在收盘后可用 → 填入当天（下一个交易日开盘可用）。
    """
    # 确保索引一致
    fund_df = fund_df.sort_index()
    all_dates = fund_df.index.union(daily_index).sort_values()

    # reindex 到全部日期（fund 日期 + 交易日），前向填充
    result = fund_df.reindex(all_dates).ffill()

    # 只保留交易日
    result = result.reindex(daily_index)
    return result


def compute_asset_growth(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """计算 Asset Growth 因子（总资产同比增长率 YOYAsset）。

    baostock query_growth_data → YOYAsset（单位: %）
    预测方向: 负（高资产增长 → 低未来收益）

    Parameters
        close_matrix: 价格面板 DataFrame(date × symbol)，用于对齐日期和股票

    Returns
        DataFrame(date × symbol)，因子值 = YOYAsset（%），非覆盖区为 NaN
    """
    cached = load_cached_field("YOYAsset")
    if cached is None or cached.empty:
        raise RuntimeError("YOYAsset 缓存为空，请先运行 python data/fundamental_fetcher.py")

    symbols = [s for s in close_matrix.columns if s in cached.columns]
    if not symbols:
        raise RuntimeError("YOYAsset 缓存与 close_matrix 无交集股票")

    fund_sub = cached[symbols]
    daily_filled = _forward_fill_to_daily(fund_sub, close_matrix.index)

    # 对齐列顺序
    result = pd.DataFrame(index=close_matrix.index, columns=close_matrix.columns, dtype=float)
    result[symbols] = daily_filled[symbols].values
    return result


def compute_earnings_quality(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """计算 Earnings Quality 因子（经营现金流 / 净利润，CFOToNP）。

    baostock query_cash_flow_data → CFOToNP
    高值 = 盈利质量高（现金流扎实），低值 = 应计项目多（可能粉饰报表）。
    预测方向: 正（高现金流质量 → 高未来收益）

    Parameters
        close_matrix: 价格面板 DataFrame(date × symbol)，用于对齐日期和股票

    Returns
        DataFrame(date × symbol)，因子值 = CFOToNP，非覆盖区为 NaN
    """
    cached = load_cached_field("CFOToNP")
    if cached is None or cached.empty:
        raise RuntimeError("CFOToNP 缓存为空，请先运行 python data/fundamental_fetcher.py")

    symbols = [s for s in close_matrix.columns if s in cached.columns]
    if not symbols:
        raise RuntimeError("CFOToNP 缓存与 close_matrix 无交集股票")

    fund_sub = cached[symbols]
    daily_filled = _forward_fill_to_daily(fund_sub, close_matrix.index)

    result = pd.DataFrame(index=close_matrix.index, columns=close_matrix.columns, dtype=float)
    result[symbols] = daily_filled[symbols].values
    return result
