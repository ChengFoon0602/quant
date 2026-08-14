"""
build_monthly_matrix.py — 月末快照 X/y 长表（方向2 ML 输入）。

与价量链路（日频全采样）的本质差异：
  季频因子日频前向填充后，日频全采样会重复采样同一财报值 + 引入 IC 自相关。
  这里把 X 和 y 都降采样到月末截面（每月最后一个交易日），
  y = close(t+22)/close(t+1)-1（21 日前向收益，月调仓口径）。

样本量 ≈ 190 月末 × ~500 成员 ≈ 95k 行 × (最终池因子 + 2 市场特征)。

用法:
    python strategies/zz500_fundamental_trial/build_monthly_matrix.py
"""

from __future__ import annotations

import sys

import pandas as pd

from config import (
    THIS_DIR, FEATURE_SEL_DIR, PROJECT_ROOT, INDEX, FWD_DAYS, MARKET_COLS,
)
from purify import month_end_dates, fwd_return

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from build_pit_matrix import load_pit_panel, build_market_features_pit


def _fallback_diagnostic_pool() -> list[str]:
    """四维 0 通过时的退化池：IC_t 绝对值 top-10（含 selection bias，仅诊断上限）。"""
    pur_path = THIS_DIR / "purify_results_monthly.csv"
    if not pur_path.exists():
        raise SystemExit("缺 purify_results_monthly.csv，先跑 purify.py")
    pur = pd.read_csv(pur_path)
    pool = pur.sort_values("IC_t", key=abs, ascending=False)["factor"].head(10).tolist()
    print(f"[WARN] 四维 0 通过，退化 IC_t top-10 诊断池: {pool}（含 selection bias，仅上限）")
    return pool


def build_monthly_matrix(close_matrix, volume_matrix, member_daily,
                         factor_tensor: dict[str, pd.DataFrame],
                         final_pool: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """构建月末长表 X_monthly / y_monthly，写盘并返回。

    Parameters
        factor_tensor: {field: date×symbol 日频因子}（已含成员掩码）
        final_pool: 提纯后因子池；为空时退化为 IC_t top-10 诊断池
    """
    if not final_pool:
        final_pool = _fallback_diagnostic_pool()
    frames = [factor_tensor[f].stack().rename(f) for f in final_pool]
    X_factor = pd.concat(frames, axis=1)

    # 市场特征（价量 regime 补充）：全时段算好再切片到月末
    mkt = build_market_features_pit(close_matrix, volume_matrix, member_daily)
    mkt_long = mkt.loc[X_factor.index.get_level_values(0)]
    mkt_long.index = X_factor.index
    X = pd.concat([X_factor, mkt_long], axis=1)

    # 月末降采样
    med = set(month_end_dates(X.index.get_level_values(0).unique()))
    keep_med = X.index.get_level_values(0).isin(med)
    X = X[keep_med]

    # 成员过滤（PIT 掩码）
    member_long = member_daily.stack()
    keep_mem = member_long.reindex(X.index).fillna(False).astype(bool)
    X = X[keep_mem.values]

    # 21 日前向收益（月末对齐）
    fwd = fwd_return(close_matrix, FWD_DAYS)
    y = fwd.stack().rename("fwd_return").reindex(X.index)
    X.index.names = y.index.names = ["date", "symbol"]

    X.to_csv(THIS_DIR / "X_monthly.csv")
    y.to_csv(THIS_DIR / "y_monthly.csv")
    n_months = X.index.get_level_values(0).nunique()
    print(f"  X_monthly: {len(X):,} 行 × {len(X.columns)} 特征 | "
          f"{n_months} 个月末 | {X.index.get_level_values(1).nunique()} 只股票 | "
          f"特征 {list(X.columns)}")
    return X, y


def main():
    print("=" * 72)
    print("方向2：构建月末 X/y 矩阵")
    print("=" * 72)

    # 需要先跑 purify.py 得到最终池；复用 purify_results_monthly.csv 读池
    import pandas as pd
    pur_path = THIS_DIR / "purify_results_monthly.csv"
    if not pur_path.exists():
        raise SystemExit("缺 purify_results_monthly.csv，先跑 purify.py")
    pur = pd.read_csv(pur_path)
    passed = pur[pur["pass"]]
    if passed.empty:
        print("[WARN] 无四维通过因子，退化为全部候选（含未通过者仅作诊断）")
        final_pool = pur.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()
    else:
        # 严格按提纯输出顺序：IC_IR 降序 + 冗余剔除后池（见 purify.py 的 final_pool 逻辑）
        # 这里简化：取通过因子按 |IC_IR| 降序前 MAX_POOL_SIZE（冗余剔除在 purify 已做）
        final_pool = passed.sort_values("IC_IR", key=abs, ascending=False)["factor"].head(10).tolist()
    print(f"因子池: {final_pool}")

    print("\n[1] 加载 PIT 面板...")
    close, volume, member = load_pit_panel(INDEX)

    print("[2] 计算因子张量...")
    from signals.fundamental.factors import compute_factor_tensor
    factor_tensor = compute_factor_tensor(close, final_pool)
    # 成员掩码（与 purify 一致）
    factor_tensor = {f: df.where(member) for f, df in factor_tensor.items()}

    print("[3] 构建月末矩阵...")
    X, y = build_monthly_matrix(close, volume, member, factor_tensor, final_pool)
    print("\n完成。")


if __name__ == "__main__":
    main()
