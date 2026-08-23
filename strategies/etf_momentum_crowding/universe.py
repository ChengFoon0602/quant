"""
strategies/etf_momentum_crowding/universe.py — ETF 资产池定义与有效性过滤。
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd

ETF_UNIVERSE = {
    # 宽基与风格
    "510050": {"name": "上证50ETF", "category": "broad"},
    "510300": {"name": "沪深300ETF", "category": "broad"},
    "510500": {"name": "中证500ETF", "category": "broad"},
    "159915": {"name": "创业板ETF", "category": "broad"},
    "512100": {"name": "中证1000ETF", "category": "broad"},

    # 核心行业与主题
    "512880": {"name": "证券ETF", "category": "sector"},
    "512800": {"name": "银行ETF", "category": "sector"},
    "512010": {"name": "医药ETF", "category": "sector"},
    "159928": {"name": "主要消费ETF", "category": "sector"},
    "512660": {"name": "军工ETF", "category": "sector"},
    "512480": {"name": "半导体ETF", "category": "sector"},
    "515050": {"name": "5G通信ETF", "category": "sector"},
    "515790": {"name": "光伏ETF", "category": "sector"},
    "512200": {"name": "房地产ETF", "category": "sector"},
    "512760": {"name": "芯片ETF", "category": "sector"},

    # 避险与现金资产
    "511010": {"name": "国债ETF", "category": "safe"},
    "518880": {"name": "黄金ETF", "category": "safe"},
    "511880": {"name": "银华日利ETF", "category": "cash"},
}

RISK_ASSETS = [sym for sym, info in ETF_UNIVERSE.items() if info["category"] in ["broad", "sector"]]
SAFE_ASSETS = [sym for sym, info in ETF_UNIVERSE.items() if info["category"] in ["safe", "cash"]]
ALL_ASSETS = list(ETF_UNIVERSE.keys())


def filter_active_universe(
    close_matrix: pd.DataFrame,
    min_history_bars: int = 60,
) -> pd.DataFrame:
    """生成动态有效性布尔掩码（防范冷启动与幸存者偏差）。

    某 ETF 仅在其实际上市交易满 min_history_bars 天后才被纳入备选池。
    """
    valid_counts = close_matrix.notna().cumsum()
    active_mask = valid_counts >= min_history_bars
    return active_mask
