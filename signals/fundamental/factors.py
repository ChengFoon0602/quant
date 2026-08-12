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

import numpy as np
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


# ── 因子工厂（批量生成，方向2 PIT 基本面）────────────────────────
#
# 25 因子 = 21 财报（baostock 季频，PIT=pubDate+1 对齐）+ 4 估值
# （日频快照，价格当天已知，无公告延迟）。方向标注为先验，最终判定
# 以月末 IC 实测符号为准（报告给方向一致性诊断）。

FACTOR_SPECS: dict[str, dict] = {
    # 质量（+）
    "roeAvg":       {"field": "roeAvg",      "source": "fundamental", "category": "质量", "direction": "+", "api": "query_profit_data"},
    "dupontROE":    {"field": "dupontROE",   "source": "fundamental", "category": "质量", "direction": "+", "api": "query_dupont_data"},
    "npMargin":     {"field": "npMargin",    "source": "fundamental", "category": "质量", "direction": "+", "api": "query_profit_data"},
    "gpMargin":     {"field": "gpMargin",    "source": "fundamental", "category": "质量", "direction": "+", "api": "query_profit_data"},
    "epsTTM":       {"field": "epsTTM",      "source": "fundamental", "category": "质量", "direction": "+", "api": "query_profit_data"},
    # 成长（YOYAsset 为资产增长异象，预测负）
    "YOYNI":        {"field": "YOYNI",       "source": "fundamental", "category": "成长", "direction": "+", "api": "query_growth_data"},
    "YOYPNI":       {"field": "YOYPNI",      "source": "fundamental", "category": "成长", "direction": "+", "api": "query_growth_data"},
    "YOYEPSBasic":  {"field": "YOYEPSBasic", "source": "fundamental", "category": "成长", "direction": "+", "api": "query_growth_data"},
    "YOYAsset":     {"field": "YOYAsset",    "source": "fundamental", "category": "成长", "direction": "-", "api": "query_growth_data"},
    # 现金流（+）
    "CFOToNP":      {"field": "CFOToNP",     "source": "fundamental", "category": "现金流", "direction": "+", "api": "query_cash_flow_data"},
    "CFOToOR":      {"field": "CFOToOR",     "source": "fundamental", "category": "现金流", "direction": "+", "api": "query_cash_flow_data"},
    "CFOToGr":      {"field": "CFOToGr",     "source": "fundamental", "category": "现金流", "direction": "+", "api": "query_cash_flow_data"},
    "CAToAsset":    {"field": "CAToAsset",   "source": "fundamental", "category": "现金流", "direction": "+", "api": "query_cash_flow_data"},
    "ebitToInterest": {"field": "ebitToInterest", "source": "fundamental", "category": "现金流", "direction": "+", "api": "query_cash_flow_data"},
    # 营运（+）
    "NRTurnRatio":  {"field": "NRTurnRatio",  "source": "fundamental", "category": "营运", "direction": "+", "api": "query_operation_data"},
    "INVTurnRatio": {"field": "INVTurnRatio", "source": "fundamental", "category": "营运", "direction": "+", "api": "query_operation_data"},
    "CATurnRatio":  {"field": "CATurnRatio",  "source": "fundamental", "category": "营运", "direction": "+", "api": "query_operation_data"},
    "AssetTurnRatio": {"field": "AssetTurnRatio", "source": "fundamental", "category": "营运", "direction": "+", "api": "query_operation_data"},
    # 杠杆（liabilityToAsset 预测负，其余正）
    "liabilityToAsset": {"field": "liabilityToAsset", "source": "fundamental", "category": "杠杆", "direction": "-", "api": "query_balance_data"},
    "currentRatio": {"field": "currentRatio", "source": "fundamental", "category": "杠杆", "direction": "+", "api": "query_balance_data"},
    "cashRatio":    {"field": "cashRatio",    "source": "fundamental", "category": "杠杆", "direction": "+", "api": "query_balance_data"},
    # 估值（日频快照，预测负 = 价值溢价）
    "peTTM":        {"field": "peTTM", "source": "valuation", "category": "估值", "direction": "-", "api": "query_history_k_data_plus",
                     "filter_nonpositive": True, "winsorize": (0.01, 0.99), "log": True},
    "pbMRQ":        {"field": "pbMRQ", "source": "valuation", "category": "估值", "direction": "-", "api": "query_history_k_data_plus",
                     "filter_nonpositive": True, "winsorize": (0.01, 0.99), "log": True},
    "psTTM":        {"field": "psTTM", "source": "valuation", "category": "估值", "direction": "-", "api": "query_history_k_data_plus",
                     "filter_nonpositive": True, "winsorize": (0.01, 0.99), "log": True},
    "pcfNcfTTM":    {"field": "pcfNcfTTM", "source": "valuation", "category": "估值", "direction": "-", "api": "query_history_k_data_plus",
                     "filter_nonpositive": True, "winsorize": (0.01, 0.99), "log": True},
}


def _winsorize_cross(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """逐截面（逐行）winsorize：每行 clip 到 [lower, upper] 分位。

    向量化实现，不改顺序 → 对 rank IC 无影响（主要为 LGBM 输入去极端值）。
    """
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def _make_compute(field: str, spec: dict):
    """生成单个因子的 compute_{field}(close_matrix) → date×symbol 日频因子。"""
    source = spec["source"]

    def compute(close_matrix: pd.DataFrame) -> pd.DataFrame:
        if source == "fundamental":
            cached = load_cached_field(field)
        else:
            from data.valuation_fetcher import load_cached_valuation
            cached = load_cached_valuation(field)
        if cached is None or cached.empty:
            raise RuntimeError(f"{field} 缓存为空，先跑 data/fundamental_fetcher.py / data/valuation_fetcher.py")

        symbols = [s for s in close_matrix.columns if s in cached.columns]
        if not symbols:
            raise RuntimeError(f"{field} 缓存与 close_matrix 无交集股票")

        daily_filled = _forward_fill_to_daily(cached[symbols], close_matrix.index)

        # 估值预处理：剔除非正（亏损股）→ 截面 winsorize → 可选 log
        if spec.get("filter_nonpositive"):
            daily_filled = daily_filled.where(daily_filled > 0)
        if spec.get("winsorize"):
            daily_filled = _winsorize_cross(daily_filled, *spec["winsorize"])
        if spec.get("log"):
            daily_filled = np.log(daily_filled)

        result = pd.DataFrame(index=close_matrix.index, columns=close_matrix.columns, dtype=float)
        result[symbols] = daily_filled[symbols].values
        return result

    compute.__name__ = f"compute_{field}"
    compute.__doc__ = (
        f"计算 {field} 因子（{spec['category']}，先验方向 {spec['direction']}）。\n"
        f"来源: {spec['source']}（{spec['api']}）"
    )
    return compute


# 批量生成 compute_{field} 到模块命名空间
for _field, _spec in FACTOR_SPECS.items():
    globals()[f"compute_{_field}"] = _make_compute(_field, _spec)


def list_factor_specs() -> pd.DataFrame:
    """返回因子清单表（供报告遍历：field/category/direction/source/api）。"""
    return pd.DataFrame.from_dict(FACTOR_SPECS, orient="index")


def compute_factor_tensor(close_matrix: pd.DataFrame, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """批量计算指定因子的日频张量 {field: date×symbol}。

    Parameters
        close_matrix: 价格面板（对齐日期与股票）
        fields: 因子名列表；None = 全部 25 个
    """
    if fields is None:
        fields = list(FACTOR_SPECS.keys())
    tensor = {}
    for f in fields:
        tensor[f] = globals()[f"compute_{f}"](close_matrix)
    return tensor
