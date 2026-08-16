"""
config.py — 方向C 延伸④ 沪深300 因子拥挤度（跨指数验证）共享配置。

与 zz500_crowding_trial/config.py 的唯一差异是 INDEX="hs300"（其余参数一致，
保证两指数同一代码路径、同一口径可比）。复用 zz500_crowding_trial 的
crowding / fundamental_crowding / event_study / tradability 模块（它们
`from config import ...`，本目录的 config 优先进 sys.path → 自动切换到 hs300）。

数据源：
  - 量价池：16 alpha 因子在 hs300 PIT 面板上计算（X_matrix_hs300.csv，成员掩码）
  - 基本面池：方向2 缓存（cache_fundamental + cache_valuation）与 hs300 成员的
    交集。⚠️ 缓存为 zz500 域（不含 mega-cap），hs300 月末可用 ~138/300 只，
    报告中如实披露覆盖局限。
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
INDEX = "hs300"
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

# X_matrix 的 16 因子列（与 zz500 完全一致，跨指数可比）
FACTOR_COLS = [
    "alpha116", "alpha001", "alpha142", "alpha087", "alpha108",
    "alpha003", "alpha144", "alpha023", "alpha051", "alpha110",
    "alpha162", "alpha011", "alpha169", "alpha021", "alpha176", "alpha055",
]
MARKET_COLS = ["market_vol_20d", "market_turnover_20d"]
