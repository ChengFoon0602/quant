"""
neutralize.py — 预测值截面中性化（行业哑变量 + log 成交额 OLS 取残差）。

P1 多头增强核心：把模型预测的截面信号对行业归属与市值（成交额代理）回归，
取残差作为"纯 alpha"预测，再排序选股。防止 LO 组合只是赌行业集中 / 小市值暴露。

无超参（行业哑变量 + log成交额回归不调参）→ 对已消除 selection bias 的
OOF 预测再加工不会重新引入选择偏差。

市值用成交额 amount（元）代理（项目 data/universe.py 已有先例：20日均成交额
作为流动性代理市值）。ZZ500 全中盘，成交额代理防小票 Beta 更直接。

用法:
    from strategies.zz500_pit_trial.neutralize import (
        load_amount_matrix, neutralize_cross_section)
    amount = load_amount_matrix(close_matrix)
    pred_neutral, diag_corr = neutralize_cross_section(pred_matrix, industry_series, amount)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_amount_matrix(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """从 data/cache 构建 amount_matrix (date × symbol, 元)。

    只取 close_matrix 的 index/columns；未缓存 / 缺 amount 字段的股票列为 NaN。
    """
    from data.fetcher import load_daily

    amounts = {}
    for sym in close_matrix.columns:
        df = load_daily(sym)
        if df is not None and "amount" in df.columns:
            s = df["amount"].reindex(close_matrix.index)
            amounts[sym] = s
    amount_matrix = pd.DataFrame(amounts).reindex(index=close_matrix.index, columns=close_matrix.columns)
    return amount_matrix


def neutralize_cross_section(
    pred_matrix: pd.DataFrame,
    industry_series: pd.Series,
    amount_matrix: pd.DataFrame,
    min_obs: int = 30,
) -> pd.DataFrame:
    """预测值截面中性化：每个日期 d 做 pred ~ 行业哑变量 + log(amount) OLS 取残差。

    Parameters
        pred_matrix: date × symbol 预测值
        industry_series: Series(index=symbol, value=行业名)
        amount_matrix: date × symbol 成交额（元）
        min_obs: 当日有效股票数下限，低于则整日置 NaN

    Returns
        pred_neutral: date × symbol 残差。与 pred_matrix 同 index/columns。
    """
    log_amount = np.log(amount_matrix.replace(0, np.nan))
    industries = sorted(industry_series.dropna().astype(str).unique())
    ind_idx = {ind: i for i, ind in enumerate(industries)}
    sym_ind = industry_series.dropna().astype(str)

    out = pd.DataFrame(np.nan, index=pred_matrix.index, columns=pred_matrix.columns)
    diag = []
    for d in pred_matrix.index:
        pv = pred_matrix.loc[d]
        la = log_amount.loc[d]
        mask = pv.notna() & la.notna() & pv.index.isin(sym_ind.index)
        obs = pv.index[mask]
        if len(obs) < min_obs:
            diag.append(np.nan)
            continue

        y = pv[obs].values
        # 设计矩阵 X = [1, industry_dummies, log(amount)]
        X = np.zeros((len(obs), 1 + len(industries) + 1))
        X[:, 0] = 1.0
        for i, s in enumerate(obs):
            X[i, 1 + ind_idx[sym_ind[s]]] = 1.0
        X[:, -1] = la[obs].values

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out.loc[d, obs] = resid
        diag.append(np.corrcoef(y, resid)[0, 1])  # 残差与原始预测的相关性

    diag_series = pd.Series(diag, index=pred_matrix.index, name="corr")
    return out, diag_series
