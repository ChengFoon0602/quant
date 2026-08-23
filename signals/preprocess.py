"""
signals/preprocess.py — 因子预处理与截面中性化模块。

提供标准量化因子预处理流水线：
  1. winsorize_mad: 基于中位数绝对偏差（MAD）的稳健去极值；
  2. standardize_zscore: 截面均值归零与方差归一化；
  3. neutralize: 多元 OLS 残差正交中性化（行业中性化 + 对数市值中性化）；
  4. preprocess_factor_pipeline: 链式流水线封装。
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd


def winsorize_mad(
    factor_df: pd.DataFrame,
    n: float = 3.0,
) -> pd.DataFrame:
    """基于中位数绝对偏差（MAD）的稳健截面去极值。

    上下界定义为: median +/- n * 1.4826 * MAD
    其中 MAD = median(|x - median|)
    """
    out = factor_df.copy()

    for idx, row in out.iterrows():
        s = row.dropna()
        if len(s) < 5:
            continue
        med = s.median()
        mad = (s - med).abs().median()
        if mad <= 1e-8:
            continue
        scale = 1.4826 * mad
        lower = med - n * scale
        upper = med + n * scale
        out.loc[idx] = row.clip(lower=lower, upper=upper)

    return out


def standardize_zscore(
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """截面 Z-score 标准化：均值归零，标准差归一。"""
    mean = factor_df.mean(axis=1)
    std = factor_df.std(axis=1).replace(0, np.nan)
    return factor_df.sub(mean, axis=0).div(std, axis=0)


def neutralize(
    factor_df: pd.DataFrame,
    industry_matrix: Optional[pd.DataFrame] = None,
    market_cap_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """截面多元 OLS 残差正交中性化（剥离行业哑变量与对数市值暴露）。

    模型:
      factor_i = alpha + sum(beta_k * Ind_k,i) + gamma * ln(MCAP_i) + epsilon_i
    返回残差 epsilon 作为纯净中性化因子值。

    Parameters
    ----------
    factor_df : pd.DataFrame
        原始因子矩阵 (index=date, columns=symbols)。
    industry_matrix : Optional[pd.DataFrame]
        行业分类矩阵 (index=date, columns=symbols 或 包含 'industry' 列的 Series/DataFrame)。
    market_cap_df : Optional[pd.DataFrame]
        总市值矩阵 (index=date, columns=symbols)。
    """
    out = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)

    # 预计算对数市值
    log_mcap = np.log(market_cap_df.replace(0, np.nan)) if market_cap_df is not None else None

    for d in factor_df.index:
        y_raw = factor_df.loc[d].dropna()
        if len(y_raw) < 10:
            out.loc[d] = factor_df.loc[d]
            continue

        valid_symbols = y_raw.index
        X_parts = [np.ones((len(valid_symbols), 1))]  # 截距项

        # 1. 行业哑变量
        if industry_matrix is not None:
            if d in industry_matrix.index:
                ind_row = industry_matrix.loc[d].reindex(valid_symbols)
            elif isinstance(industry_matrix, pd.Series):
                ind_row = industry_matrix.reindex(valid_symbols)
            else:
                ind_row = pd.Series(index=valid_symbols)

            dummies = pd.get_dummies(ind_row, drop_first=True, dtype=float)
            if not dummies.empty and dummies.shape[1] > 0:
                X_parts.append(dummies.values)

        # 2. 对数市值
        if log_mcap is not None and d in log_mcap.index:
            mcap_row = log_mcap.loc[d].reindex(valid_symbols).fillna(log_mcap.loc[d].median())
            X_parts.append(mcap_row.values.reshape(-1, 1))

        if len(X_parts) == 1:
            # 无任何中性化自变量，直接均值归零
            out.loc[d, valid_symbols] = y_raw - y_raw.mean()
            continue

        X = np.hstack(X_parts)
        y = y_raw.values

        # 最小二乘求解残差: residuals = y - X @ (X^T X)^(-1) X^T y
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            out.loc[d, valid_symbols] = residuals
        except Exception:
            out.loc[d, valid_symbols] = y_raw - y_raw.mean()

    return out


def preprocess_factor_pipeline(
    factor_df: pd.DataFrame,
    winsorize_n: float = 3.0,
    industry_matrix: Optional[pd.DataFrame] = None,
    market_cap_df: Optional[pd.DataFrame] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """一键执行完整因子预处理流水线：去极值 -> 中性化 -> 标准化。"""
    # 1. MAD 去极值
    df_win = winsorize_mad(factor_df, n=winsorize_n)

    # 2. 行业与市值中性化
    if industry_matrix is not None or market_cap_df is not None:
        df_neu = neutralize(df_win, industry_matrix=industry_matrix, market_cap_df=market_cap_df)
    else:
        df_neu = df_win

    # 3. 截面标准化
    if standardize:
        df_out = standardize_zscore(df_neu)
    else:
        df_out = df_neu

    return df_out
