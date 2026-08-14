"""
config.py — 方向C 中证500 因子拥挤度时序研究（市场结构测度）共享配置。

与方向2（zz500_fundamental_trial）的方法论差异：这里不找 alpha（不预测截面
收益），而是测度「风格因子拥挤度」如何随时间变化——市场结构理解，非 trading
signal。复用方向2 已验证的月末采样框架（避免季频/日频 IC 自相关）。

数据源：strategies/zz500_pit_trial/X_matrix.csv（16 alpha 因子最终池暴露矩阵，
零重算直接复用）+ build_market_features_pit（市场特征）。
"""

from __future__ import annotations

from pathlib import Path

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
FEATURE_SEL_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
PIT_TRIAL_DIR = PROJECT_ROOT / "strategies" / "zz500_pit_trial"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── 基础参数 ──────────────────────────────────────────────────
INDEX = "zz500"
DATE_START = "2010-01-01"
DATE_END = "2025-12-31"

# ── 拥挤度指标参数 ────────────────────────────────────────────
Z_EXTREME_THRESHOLD = 2.0     # C1: |z|>2 视为极端暴露
STYLE_LOOKBACK = 12           # C2/C3: 滚动月数（风格同质化 / 换手拥挤）
SPIKE_LOOKBACK = 6            # C4: 因子收益尖峰滚动月数
CONDITION_LOOKBACK = 12       # 条件收益：拥挤度 t 后的未来月数

# 市场大事件（图标注）
MARKET_EVENTS = {
    "2015-06-30": "2015 股灾",
    "2021-02-26": "2021 核心资产",
    "2024-02-29": "2024 微盘崩",
}

# X_matrix 的 16 因子列（X_matrix.csv 除 date/symbol/市场特征外）
FACTOR_COLS = [
    "alpha116", "alpha001", "alpha142", "alpha087", "alpha108",
    "alpha003", "alpha144", "alpha023", "alpha051", "alpha110",
    "alpha162", "alpha011", "alpha169", "alpha021", "alpha176", "alpha055",
]
MARKET_COLS = ["market_vol_20d", "market_turnover_20d"]
