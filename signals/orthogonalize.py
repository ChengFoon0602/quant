"""
signals/orthogonalize.py — 横截面正交化（剥离基准因子暴露）。

方法：对每个交易日截面，用目标因子对基准因子群做 OLS 回归（含截距），
取残差作为净化后因子。等价于把目标因子投影到基准张成的正交补空间上
（即施密特正交化的残差形式）。

⚠️ 2026-09-09 修复：原实现用 `except Exception: pass` 静默吞掉 lstsq 失败，
导致奇异矩阵（如基准因子共线）的交易日被静默填 NaN，研究者误以为"该日无数据"。
现改为：捕获具体的 LinAlgError -> logging 警告 -> 通过 `DataFrame.attrs` 暴露失败日期。
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cross_sectional_orthogonalize(
    target_factor: pd.DataFrame,
    base_factors: Dict[str, pd.DataFrame],
    min_obs: int | None = None,
) -> pd.DataFrame:
    """横截面 OLS 正交化，返回净化后的残差因子。

    Parameters
    ----------
    target_factor : pd.DataFrame
        目标被净化因子矩阵（index=日期, columns=股票）。
    base_factors : Dict[str, pd.DataFrame]
        基准因子群，key 为因子名，value 为同形状矩阵（如 {'alpha012': df1}）。
        空 dict 时直接返回 target_factor 的副本。
    min_obs : int | None, default None
        截面最小有效样本数。None 时沿用默认自由度检查
        （有效样本 > 基准因子数 + 1，即截距 + 各自变量）。

    Returns
    -------
    pd.DataFrame
        净化后的残差因子，形状与 target_factor 相同。
        诊断信息挂在返回值的 `.attrs` 上：
        - `failed_dates`：lstsq 求解失败（奇异矩阵）的日期
        - `insufficient_obs_dates`：有效样本不足、未做回归的日期

    Notes
    -----
    失败日期对应行整行为 NaN。调用方应检查 `.attrs['failed_dates']`，
    而非把 NaN 当作"该日无数据"。
    """
    if not base_factors:
        return target_factor.copy()

    dates = target_factor.index
    columns = target_factor.columns

    residual_df = pd.DataFrame(index=dates, columns=columns, dtype=float)

    # 对齐所有基准因子矩阵到目标因子的日期/股票索引
    base_dfs = [v.reindex(index=dates, columns=columns) for v in base_factors.values()]

    n_base = len(base_dfs)
    failed_dates: list = []
    insufficient_obs_dates: list = []

    for date in dates:
        y = target_factor.loc[date].values
        X = np.column_stack([df.loc[date].values for df in base_dfs])

        # 严格过滤缺失值：因变量与所有自变量均非空的标的才进入截面回归
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        n_valid = int(valid_mask.sum())

        # 自由度检查：需要多于 (基准因子数 + 截距) 的样本
        required = min_obs if min_obs is not None else (n_base + 2)
        if n_valid < required:
            insufficient_obs_dates.append(date)
            continue

        X_valid = np.column_stack([np.ones(n_valid), X[valid_mask]])
        y_valid = y[valid_mask]

        try:
            beta, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
        except np.linalg.LinAlgError as exc:
            # 奇异矩阵（基准因子共线）：记录而非静默丢弃
            failed_dates.append(date)
            logger.warning(
                "正交化失败 %s：截面回归奇异（可能基准因子共线），该日残差留 NaN。%s",
                date, exc,
            )
            continue

        resid = y_valid - X_valid @ beta
        res = np.full(len(y), np.nan)
        res[valid_mask] = resid
        residual_df.loc[date] = res

    # 诊断信息外挂，保持返回值仍为 DataFrame（向后兼容）
    residual_df.attrs["failed_dates"] = failed_dates
    residual_df.attrs["insufficient_obs_dates"] = insufficient_obs_dates

    if failed_dates:
        logger.warning(
            "正交化完成但 %d/%d 个截面求解失败（奇异矩阵）。"
            "请检查基准因子是否存在共线；失败日期见返回值 .attrs['failed_dates']。",
            len(failed_dates), len(dates),
        )
    if insufficient_obs_dates:
        logger.info(
            "正交化：%d/%d 个截面有效样本不足，未做回归（残差为 NaN）。",
            len(insufficient_obs_dates), len(dates),
        )

    return residual_df
