"""
批量截面因子计算器。

将多只股票的 OHLCV 数据加载并按给定因子 ID 计算，输出截面矩阵，
供回测引擎直接使用。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fetcher import load_daily
from .factors import factor_001 as f001  # 用于 get_factor_func


def list_factors() -> list[str]:
    """返回全部 191 个因子 ID 列表。"""
    return [f"alpha{i:03d}" for i in range(1, 192)]


def get_factor_func(factor_id: str):
    """根据因子 ID 字符串获取函数对象。

    Parameters
        factor_id: 如 "alpha001", "alpha042", "alpha191"

    Returns
        callable: factor_XXX(df) -> pd.Series
    """
    num = int(factor_id.replace("alpha", "").replace("Alpha", ""))
    module = __import__(
        f"signals.alpha191.factors", fromlist=[f"factor_{num:03d}"]
    )
    return getattr(module, f"factor_{num:03d}")


def compute_factor_matrix(
    symbols: list[str],
    factor_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算给定 symbols × factors 的截面因子矩阵。

    Parameters
        symbols: 股票代码列表，如 ["000001", "600000"]
        factor_ids: 因子 ID 列表，如 ["alpha001", "alpha072"]
        start, end: 日期筛选 YYYY-MM-DD，None = 全部
        verbose: 打印进度

    Returns
        (close_matrix, factor_tensor):
            close_matrix — index=date, columns=symbols
            factor_tensor — dict[factor_id] -> DataFrame[index=date, columns=symbols]
    """
    # 加载所有股票的 close 和日线数据
    all_dfs = {}
    close_data = {}
    for sym in symbols:
        df_raw = load_daily(sym)
        if df_raw is None:
            continue
        cols = ["open", "high", "low", "close", "volume", "amount"]
        df = df_raw[cols].copy()
        if start is not None:
            df = df.loc[df.index >= start]
        if end is not None:
            df = df.loc[df.index <= end]
        if len(df) < 100:
            continue
        all_dfs[sym] = df
        close_data[sym] = df["close"]

    if not close_data:
        raise RuntimeError("没有可用的股票数据。检查缓存或日期范围。")

    close_matrix = pd.DataFrame(close_data).sort_index()

    # 对每个因子，逐只股票计算，堆叠成矩阵
    factor_tensor = {}
    for fid in factor_ids:
        if verbose:
            print(f"  计算 {fid} ...", end=" ", flush=True)
        fn = get_factor_func(fid)
        factor_rows = {}
        for sym, df in all_dfs.items():
            try:
                factor_rows[sym] = fn(df)
            except Exception:
                factor_rows[sym] = pd.Series(np.nan, index=df.index)
        mat = pd.DataFrame(factor_rows).sort_index()
        factor_tensor[fid] = mat
        if verbose:
            print(f"形状 {mat.shape}")

    return close_matrix, factor_tensor
