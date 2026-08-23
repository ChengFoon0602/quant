"""
strategies/all_weather_risk_parity/universe.py — 全天候多资产池与宏观属性定义。
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd

ALL_WEATHER_UNIVERSE = {
    # 国内权益
    "510300": {"name": "沪深300ETF", "class": "equity_cn", "macro_driver": "cn_growth"},
    # 海外权益 (全球科技 / 美元对冲)
    "513100": {"name": "纳斯达克100ETF", "class": "equity_us", "macro_driver": "global_growth"},
    # 中国主权国债 (经济衰退/流动性宽松对冲)
    "511010": {"name": "国债ETF", "class": "bond", "macro_driver": "deflation_recession"},
    # 黄金 (滞胀/信用贬值对冲)
    "518880": {"name": "黄金ETF", "class": "gold", "macro_driver": "stagflation"},
    # 有色大宗商品 (通胀上行/工业繁荣)
    "159980": {"name": "有色大宗ETF", "class": "commodity", "macro_driver": "inflation_growth"},
    # 货币现金 (流动性蓄水池)
    "511880": {"name": "银华日利ETF", "class": "cash", "macro_driver": "liquidity"},
}

CORE_ASSETS = ["510300", "513100", "511010", "518880", "159980"]


def filter_active_multi_assets(
    close_matrix: pd.DataFrame,
    min_history_bars: int = 60,
) -> pd.DataFrame:
    """生成多资产动态有效掩码（上市满 min_history_bars 天才纳入有效协方差计算）。"""
    valid_counts = close_matrix.notna().cumsum()
    return valid_counts >= min_history_bars
