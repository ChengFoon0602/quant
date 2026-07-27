"""
build_pit_matrix.py — 在 PIT CSI 300 universe 上构建特征矩阵。

背景（2026-07 幸存者偏差修复）：
  旧 X_matrix.csv 的股票池来自 sorted(cache)[:300] —— 缓存扩到 1068 只后
  这个切片变成 298 只纯深市股票（CSI300/CSI500/退市股混合），既非沪深 300
  也不可复现。本模块用 baostock 历史成分股名单（data/index_membership.py）
  构建真正的 PIT 面板。

方法与假设：
  - 股票池 = 2010-2025 每月末 CSI 300 名单的并集（790 只，含退市股）
  - 每个 (date, symbol) 样本仅当该股当日在指数内才保留（月末快照前向填充）
  - 截面 RANK 类因子的排名池是 790 只全集的当日存活部分，非当日 300
    成员。无未来信息，仅排名基准更宽，报告中注明
  - 市场特征在成员掩码内做截面均值（当日指数成员的平均波动率/换手率）
  - 前向收益跨越成员变更边界时用真实价格（出指数 ≠ 退市，仍可交易）

用法:
  python strategies/feature_selection/build_pit_matrix.py   # 用「方法论修正 II」冻结的 16 因子池
  或 select_features.py 筛出新池后调用 build_and_save(final_pool)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from data.index_membership import load_membership, expand_to_daily
from signals.alpha191.calculator import compute_factor_matrix

DATE_START, DATE_END = "2010-01-01", "2025-12-31"
OUT_DIR = Path(__file__).parent

# 「方法论修正 II」时冻结的因子池（旧 universe 上筛出的 16 个，
# 用于复现 models/report.md 修正 II 章节的数字）
ALPHA_FIDS_MOD2 = [
    "alpha116", "alpha142", "alpha001", "alpha144", "alpha003", "alpha011",
    "alpha051", "alpha110", "alpha075", "alpha169", "alpha108", "alpha068",
    "alpha166", "alpha171", "alpha162", "alpha055",
]


def load_pit_panel(index: str = "hs300") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 PIT 面板基础数据: (close_matrix, volume_matrix, member_daily)。

    close/volume 覆盖该指数全部历史成员的全量历史；member_daily 为日频成员掩码。
    """
    from data.fetcher import load_daily

    membership = load_membership(index)
    symbols = sorted(membership.columns)

    close_data, volume_data = {}, {}
    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 100:
            continue
        s = df.loc[(df.index >= DATE_START) & (df.index <= DATE_END)]
        if len(s) < 100:
            continue
        close_data[sym] = s["close"]
        if "volume" in s.columns:
            volume_data[sym] = s["volume"]

    close_matrix = pd.DataFrame(close_data).sort_index()
    volume_matrix = pd.DataFrame(volume_data).reindex_like(close_matrix)

    member_daily = expand_to_daily(membership, close_matrix.index)
    member_daily = member_daily.reindex(columns=close_matrix.columns, fill_value=False)
    return close_matrix, volume_matrix, member_daily


def build_market_features_pit(close_matrix, volume_matrix, member_daily):
    """市场状态特征：先在全量数据上算逐股 rolling 统计，再按成员掩码做截面均值。"""
    daily_ret = close_matrix.pct_change()
    vol_20d = daily_ret.rolling(20).std()
    market_vol_20d = vol_20d.where(member_daily).mean(axis=1)

    vol_ma = volume_matrix.rolling(252).mean()
    rel_turnover = volume_matrix / vol_ma.replace(0, np.nan)
    to_20d = rel_turnover.rolling(20).mean()
    market_turnover_20d = to_20d.where(member_daily).mean(axis=1)

    return pd.DataFrame({
        "market_vol_20d": market_vol_20d,
        "market_turnover_20d": market_turnover_20d,
    })


def build_and_save(
    alpha_fids: list[str],
    factor_tensor: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """构建 PIT 长格式特征矩阵并保存 X_matrix.csv / y_matrix.csv。

    Parameters
        alpha_fids: 因子池
        factor_tensor: 可选的已算好的因子矩阵（须为 790 只全量面板），
                       None 时重新计算
    """
    close_matrix, volume_matrix, member_daily = load_pit_panel()

    if factor_tensor is None:
        print(f"计算 {len(alpha_fids)} 个因子 × {close_matrix.shape[1]} 只股票...")
        _, factor_tensor = compute_factor_matrix(
            list(close_matrix.columns), alpha_fids,
            start=DATE_START, end=DATE_END, verbose=True,
        )

    print(f"日均成员数（有行情数据）: {(member_daily & close_matrix.notna()).sum(axis=1).mean():.1f}")

    print("构建长格式特征矩阵...")
    frames_x = [factor_tensor[fid].stack().rename(fid) for fid in alpha_fids]
    X_factor = pd.concat(frames_x, axis=1)

    mkt_feat = build_market_features_pit(close_matrix, volume_matrix, member_daily)
    mkt_long = mkt_feat.loc[X_factor.index.get_level_values(0)]
    mkt_long.index = X_factor.index
    X = pd.concat([X_factor, mkt_long], axis=1)

    member_long = member_daily.stack()
    keep = member_long.reindex(X.index).fillna(False).astype(bool)
    n_before = len(X)
    X = X[keep.values]
    print(f"成员过滤: {n_before:,} → {len(X):,} 行")

    fwd_ret = close_matrix.pct_change().shift(-1).fillna(0)
    y = fwd_ret.stack().rename("fwd_return").reindex(X.index)

    X.index.names = ["date", "symbol"]
    y.index.names = ["date", "symbol"]
    X.to_csv(OUT_DIR / "X_matrix.csv")
    y.to_csv(OUT_DIR / "y_matrix.csv")
    print(f"X: {X.shape}  股票数: {X.index.get_level_values(1).nunique()}")
    print(f"保存: {OUT_DIR / 'X_matrix.csv'}")
    return X


if __name__ == "__main__":
    build_and_save(ALPHA_FIDS_MOD2)
