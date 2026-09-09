"""
rebuild_x_monthly.py — 用清理后的缓存重建 X_monthly/y_monthly（口径保持与现版一致）。

背景：2026-09-09 PIT 审计清理了 cache_fundamental 的尺度污染（gpMargin ×100 统一、
roeAvg/npMargin/epsTTM 删非约定行）。本脚本用与现版 X_monthly **完全相同的因子池**
重建矩阵，以便对比清理前后特征分布与信号变化。

因子池 = 现版 X_monthly 的非市场特征列（避免 purify 0 通过时的回退池改变列）。

用法:
    cd strategies/zz500_fundamental_trial
    python rebuild_x_monthly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
FEATURE_SEL_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))
sys.path.insert(0, str(THIS_DIR))

from config import INDEX, MARKET_COLS  # noqa: E402
from build_pit_matrix import load_pit_panel  # noqa: E402
from build_monthly_matrix import build_monthly_matrix  # noqa: E402


def main():
    # 1. 现版 X_monthly 的因子池（非市场列）
    cur = pd.read_csv(THIS_DIR / "X_monthly.csv")
    market = set(MARKET_COLS)
    pool = [c for c in cur.columns if c not in ("date", "symbol") and c not in market]
    print(f"复用现版因子池 ({len(pool)}): {pool}")

    # 2. 加载 PIT 面板
    print("\n[1] 加载 PIT 面板...")
    close, volume, member = load_pit_panel(INDEX)

    # 3. 计算因子张量（读清理后缓存）
    print("[2] 计算因子张量...")
    from signals.fundamental.factors import compute_factor_tensor

    factor_tensor = compute_factor_tensor(close, pool)
    available = {f for f in pool if f in factor_tensor}
    missing = set(pool) - available
    if missing:
        print(f"  [WARN] 缺缓存/无法计算: {sorted(missing)}")
    factor_tensor = {f: df.where(member) for f, df in factor_tensor.items()}

    # 4. 重建
    print("[3] 构建月末矩阵...")
    X, y = build_monthly_matrix(close, volume, member, factor_tensor, pool)
    print("\n完成。X_monthly / y_monthly 已更新。")


if __name__ == "__main__":
    main()
