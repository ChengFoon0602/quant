"""
config.py — 方向2 PIT 基本面 × 中证500 共享配置（单点改）。

与价量链路（zz500_pit_trial）的方法论差异集中在这里：
  - 月末截面采样（季频因子日频前向填充的 IC 自相关 → 月末独立观测）
  - fwd_days=21（月调仓）
  - purge 按月计（PurgedTimeSeriesSplit.purge_days 是样本索引位置数）
  - 提纯阈值按月末样本量重标定
"""

from __future__ import annotations

from pathlib import Path

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
FEATURE_SEL_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── 基础参数 ──────────────────────────────────────────────────
INDEX = "zz500"
DATE_START = "2010-01-01"
DATE_END = "2025-12-31"
FWD_DAYS = 21            # 月调仓：close(t+22)/close(t+1)-1
TOP_Q = BOTTOM_Q = 0.20
COST_BPS = 0.00102      # 双边合计铁律 0.1%（2026-09 收口，build_portfolio 默认走方向分离）
MIN_N = 30               # 月末截面 IC/FM 的最少有效股票数（月末约 500 成员）

# ── ML ────────────────────────────────────────────────────────
N_SPLITS = 5
# 关键：X 已降采样到月末 → purge_days 是样本数组的索引位置数（=月末个数）。
# 相邻月末标签窗口 (d+1,d+22] 与 (d+22,d+43] 基本不重叠，purge 2 个月末已安全。
# 绝不能传 22（会 purge 掉 22 个月末 ≈ 1/5 训练集）。
PURGE_DAYS = 2
WF_TEST_YEARS = list(range(2015, 2026))
WF_NUM_BOOST = 79
EARLY_STOP = 50

# ── 提纯阈值（月末 ~190 截面重标定）──────────────────────────
IC_IR_THRESHOLD = 0.15   # IR>0.15 ⟺ t≈2.0（IR=0.05 时 t≈0.7 无意义）
IC_T_THRESHOLD = 2.0
FM_T_THRESHOLD = 2.0
CS_EFFECTIVE_THRESHOLD = 0.5
N_GROUPS = 5
RANK_IC_CORR_THRESHOLD = 0.8
MAX_POOL_SIZE = 10
MUST_KEEP: set[str] = set()   # 基本面无先验锚，不 reintroduce selection bias

# 25 因子多重检验 Bonferroni 从严口径（两尾，α/25）：|t| > norm.ppf(1 - 0.05/25/2) ≈ 3.09
BONFERRONI_T = 3.09

MARKET_COLS = ["market_vol_20d", "market_turnover_20d"]
