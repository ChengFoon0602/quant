"""
models/labels.py — 标签构建模块。

模型无关：为 LightGBM / 深度学习 / 任何后续模型统一生成训练标签。

核心设计：
  - 严格遵守项目未来函数铁律：信号日 t 收盘可用 → 持仓从 t+1 开盘开始。
  - 预测未来 5 个交易日收益 = close(t+6) / close(t+1) - 1。
  - 每个交易日，在当日有标签的股票池内做截面分位数：
      top 20% → 1 (买入)
      bottom 20% → 0 (做空/低配)
      middle 60% → NaN (不参与训练)

用法:
    from models.labels import build_labels
    y = build_labels(close_matrix, fwd_days=5, top_q=0.2, bottom_q=0.2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_fwd_return(close_matrix: pd.DataFrame, fwd_days: int) -> pd.DataFrame:
    """按未来函数铁律计算 t+1 到 t+fwd_days+1 的收益率。

    close(t+1) 是信号日收盘后第一个可执行价，close(t+fwd_days+1) 是平仓价。
    因此 fwd_days=5 对应 close(t+6) / close(t+1) - 1。
    """
    assert fwd_days >= 1, "fwd_days must be positive"
    entry = close_matrix.shift(-1)      # 次日收盘作为执行价
    exit_ = close_matrix.shift(-(fwd_days + 1))
    return exit_ / entry - 1


def _daily_quantile_label(
    fwd_ret: pd.Series,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> pd.Series:
    """对单日的 fwd_ret 序列做截面分位数标注。

    返回 Series，index 与原序列一致：
        1  : top 20%
        0  : bottom 20%
        NaN: middle 60%
    """
    valid = fwd_ret.dropna()
    n = len(valid)
    if n < 5:
        return pd.Series(np.nan, index=fwd_ret.index)

    top_thr = valid.quantile(1 - top_q)
    bottom_thr = valid.quantile(bottom_q)

    labels = pd.Series(np.nan, index=fwd_ret.index)
    labels[valid.index] = np.nan  # 默认中间
    labels[valid[valid >= top_thr].index] = 1
    labels[valid[valid <= bottom_thr].index] = 0
    return labels


def build_labels(
    close_matrix: pd.DataFrame,
    fwd_days: int = 5,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> pd.DataFrame:
    """构建截面分类标签 DataFrame (date × stocks)。

    Parameters
    ----------
    close_matrix : pd.DataFrame
        收盘价矩阵，index=date, columns=symbols。
    fwd_days : int
        未来持有期交易日数。标签 = close(t+fwd_days+1)/close(t+1)-1。
    top_q, bottom_q : float
        多头/空头分位数阈值，默认 0.2 表示前/后 20%。

    Returns
    -------
    pd.DataFrame
        与 close_matrix 同 index/columns，元素为 {1, 0, NaN}。
    """
    fwd_ret = _compute_fwd_return(close_matrix, fwd_days)
    labels = fwd_ret.apply(
        lambda row: _daily_quantile_label(row, top_q=top_q, bottom_q=bottom_q),
        axis=1,
        result_type="expand",
    )
    labels.index = fwd_ret.index
    labels.columns = fwd_ret.columns
    return labels


def get_valid_samples(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """过滤掉 y 为 NaN 的样本，返回 (X_valid, y_valid)。"""
    mask = y.notna()
    return X[mask], y[mask]


def align_X_y(
    X_long: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """把长格式 X 与 宽格式 labels 对齐，生成训练样本表 (date, stock)。"""
    y_long = labels.stack().rename("label").reset_index()
    y_long.columns = ["date", "symbol", "label"]
    X_reset = X_long.reset_index()
    X_reset["symbol"] = X_reset["symbol"].astype(str)
    y_long["symbol"] = y_long["symbol"].astype(str)
    aligned = X_reset.merge(y_long, on=["date", "symbol"], how="inner")
    aligned = aligned.set_index(["date", "symbol"])
    return aligned


def build_sample_weights(
    y: pd.Series,
    class_weight: str = "balanced",
) -> np.ndarray:
    """为二分类样本生成权重，平衡 1/0 两类。

    Parameters
    ----------
    y : pd.Series
        标签 {0, 1}。
    class_weight : str
        目前仅支持 "balanced"。

    Returns
    -------
    np.ndarray
        与 y 等长的样本权重。
    """
    if class_weight != "balanced":
        raise ValueError("Only class_weight='balanced' is supported")
    counts = y.value_counts()
    n = len(y)
    w0 = n / (2 * counts.get(0, 1))
    w1 = n / (2 * counts.get(1, 1))
    weights = y.map({0: w0, 1: w1}).values
    return weights.astype(np.float64)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetcher import load_daily, cache_summary

    cache = cache_summary()
    symbols = sorted(cache["symbol"].tolist())[:300]
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()
    labels = build_labels(close_matrix)
    valid_ratio = labels.notna().sum().sum() / labels.size
    print(f"close_matrix: {close_matrix.shape}")
    print(f"labels: {labels.shape}")
    print(f"valid label ratio: {valid_ratio:.3%}")
    print(f"class 1 ratio among valid: {(labels == 1).sum().sum() / labels.notna().sum().sum():.3%}")
    print(f"class 0 ratio among valid: {(labels == 0).sum().sum() / labels.notna().sum().sum():.3%}")
