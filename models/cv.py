"""
models/cv.py — 时间序列交叉验证 + Purge。

所有模型（LightGBM / NN）共用：按时间顺序分折，训练/验证之间 purge 等于标签前瞻期的交易日，
防止未来收益自相关造成时序泄露。

用法:
    cv = PurgedTimeSeriesSplit(n_splits=5, purge_days=6)
    for train_idx, val_idx in cv.split(X):
        ...
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Purged 时间序列 K-Fold。

    特点:
      - 严格按时间顺序：验证集全部在训练集之后。
      - 训练集与验证集之间剔除 `purge_days` 天，避免标签自相关泄露。
      - 每折的训练集都早于该折的验证集（expanding window），不混入未来。

    Parameters
    ----------
    n_splits : int
        折数，默认 5。
    purge_days : int
        训练集和验证集之间剔除的交易日数。
        若标签是 fwd_days=5 的收益，理论上前瞻期为 fwd_days+1=6 个交易日，purge_days 应 ≥6。
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 6):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        self.n_splits = n_splits
        self.purge_days = purge_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """生成 (train_idx, val_idx) 对。

        X 的 index 需要是时间序列排序后的有序数据。这里假设样本已按日期排序。
        """
        n_samples = len(X)
        if n_samples < self.n_splits * 2:
            raise ValueError("Too few samples for the requested number of splits")

        # 等分时间轴为 n_splits+2 个边界，保留中间 n_splits 个验证窗口
        boundaries = np.linspace(0, n_samples, self.n_splits + 2, dtype=int)
        test_starts = boundaries[1:-1]

        for i, test_start in enumerate(test_starts):
            # 验证集: [test_start, test_end)
            if i + 1 < len(test_starts):
                test_end = test_starts[i + 1]
            else:
                test_end = n_samples

            # purge 窗口: 验证集之前 purge_days 不参与训练
            train_end = max(0, test_start - self.purge_days)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx


if __name__ == "__main__":
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    X = pd.DataFrame({"a": range(100)}, index=dates)

    print("PurgedTimeSeriesSplit(n_splits=5, purge_days=6):")
    cv = PurgedTimeSeriesSplit(n_splits=5, purge_days=6)
    for i, (tr, val) in enumerate(cv.split(X), 1):
        print(f"  Fold {i}: train [{tr.min()}-{tr.max()}] ({len(tr)})  "
              f"val [{val.min()}-{val.max()}] ({len(val)})  "
              f"gap={val.min()-tr.max()-1} days")
