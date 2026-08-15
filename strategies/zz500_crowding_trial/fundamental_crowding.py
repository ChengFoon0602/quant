"""
fundamental_crowding.py — 基本面因子拥挤度驱动（方向C 延伸 ①）。

与量价池（X_matrix.csv 16 因子）对比：方向2 已缓存的 20 基本面因子
（16 财报 akshare + 4 估值 baostock）的拥挤度时序 + 条件收益。

核心差异：
  1. 数据源：compute_factor_tensor（{field: date×symbol} 日频宽表），
     而非 X_matrix.csv 长表 → wide_to_long 转换
  2. 方向翻转：20 因子中 6 个负方向（YOYAsset/liabilityToAsset/peTTM/
     pbMRQ/psTTM/pcfNcfTTM，低=好）×−1，使「高 z = 好方向」跨因子一致
     （align_direction，用 FACTOR_SPECS['direction']）
  3. 量价 vs 基本面分开算、对比呈现（C2 风格同质化跨池合并会失真）

用法:
    python strategies/zz500_crowding_trial/fundamental_crowding.py
"""

from __future__ import annotations

import sys

import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, INDEX,
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from build_pit_matrix import load_pit_panel
from crowding import (
    month_end_dates, wide_to_long, align_direction, compute_all,
    factor_monthly_returns,
)
from signals.fundamental.factors import FACTOR_SPECS, compute_factor_tensor

# 20 个有缓存的基本面因子（16 财报 + 4 估值）
FUNDAMENTAL_COLS = [
    "roeAvg", "npMargin", "gpMargin", "epsTTM",      # 质量
    "YOYNI", "YOYAsset",                              # 成长
    "CFOToNP", "CFOToOR", "ebitToInterest",           # 现金流
    "NRTurnRatio", "INVTurnRatio", "CATurnRatio", "AssetTurnRatio",  # 营运
    "liabilityToAsset", "currentRatio", "cashRatio",  # 杠杆
    "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",           # 估值
]

DIRECTION_MAP = {f: FACTOR_SPECS[f]["direction"] for f in FUNDAMENTAL_COLS}


def load_fundamental_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成基本面 20 因子 → 方向翻转 → 长表。返回 (X_long, member_daily)。"""
    close, _, member = load_pit_panel(INDEX)
    tensor = compute_factor_tensor(close, fields=FUNDAMENTAL_COLS)
    # PIT 成员掩码 + 方向翻转（负方向因子 ×−1）
    tensor = {f: df.where(member) for f, df in tensor.items()}
    tensor = align_direction(tensor, DIRECTION_MAP)
    X_long = wide_to_long(tensor)
    return X_long, close


def compute_fundamental_crowding() -> pd.DataFrame:
    """基本面池拥挤度时序（C1-C4）。"""
    X_long, close = load_fundamental_long()
    # 市场换手代理（复用 build_market_features_pit 的输出语义）
    from build_pit_matrix import build_market_features_pit
    _, volume, member = load_pit_panel(INDEX)
    mkt = build_market_features_pit(close, volume, member)
    turnover = mkt["market_turnover_20d"]
    ts = compute_all(X_long, close, turnover, factor_cols=FUNDAMENTAL_COLS)
    ts.to_csv(THIS_DIR / "fundamental_crowding_time_series.csv")
    return ts


if __name__ == "__main__":
    print("=" * 72)
    print("基本面因子拥挤度（方向C 延伸①，对比量价池）")
    print(f"因子: {len(FUNDAMENTAL_COLS)} 个 | 负方向翻转 {sum(v=='-' for v in DIRECTION_MAP.values())} 个")
    print("=" * 72)
    ts = compute_fundamental_crowding()
    print(f"时序: {ts.shape}（{ts.index[0].date()} → {ts.index[-1].date()}）")
    print(ts.describe().round(4).to_string())
    print("\n完成。保存 fundamental_crowding_time_series.csv")
