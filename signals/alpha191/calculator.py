"""
批量截面因子计算器。

将多只股票的 OHLCV 数据加载并按给定因子 ID 计算，输出截面矩阵，
供回测引擎直接使用。

两种计算模式：
- per-stock: 逐只股票调用 factor(df)（无 RANK 因子，与历史兼容）
- panel:    构建宽表 dict[str, DataFrame]，一次调用 factor(panel)，
            使 RANK 成为真正的截面排名（axis=1 跨股票排名）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fetcher import load_daily
from .factors import factor_001 as f001  # 用于 get_factor_func
from .factors import _factor_has_rank


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


def _build_panel(
    all_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """将 {symbol: DataFrame} 转换为宽表面板 {field: DataFrame(date × stocks)}。"""
    fields = ["open", "high", "low", "close", "volume", "amount"]
    panel = {}
    for field in fields:
        mat = pd.DataFrame(
            {sym: df[field] for sym, df in all_dfs.items()}
        ).sort_index()
        panel[field] = mat
    # VWAP
    panel["vwap"] = panel["amount"] / panel["volume"]
    return panel


def compute_factor_matrix(
    symbols: list[str],
    factor_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    verbose: bool = True,
    factor_kwargs: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算给定 symbols × factors 的截面因子矩阵。

    Parameters
        symbols: 股票代码列表，如 ["000001", "600000"]
        factor_ids: 因子 ID 列表，如 ["alpha001", "alpha072"]
        start, end: 日期筛选 YYYY-MM-DD，None = 全部
        verbose: 打印进度
        factor_kwargs: 因子参数字典，如 {"alpha055": {"delta_days": 10}}

    Returns
        (close_matrix, factor_tensor):
            close_matrix — index=date, columns=symbols
            factor_tensor — dict[factor_id] -> DataFrame[index=date, columns=symbols]
    """
    # 加载所有股票数据
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

    # 预建面板（RANK 因子需要截面数据）
    panel = _build_panel(all_dfs)

    # 对每个因子计算
    factor_tensor = {}
    for fid in factor_ids:
        if verbose:
            print(f"  计算 {fid} ...", end=" ", flush=True)
        fn = get_factor_func(fid)
        kwargs = (factor_kwargs or {}).get(fid, {})

        num = int(fid.replace("alpha", "").replace("Alpha", ""))
        uses_rank = _factor_has_rank(num)

        if uses_rank:
            # ── 面板模式：截面 RANK ──
            try:
                result_df = fn(panel, **kwargs)
                # 修复 np.where/np.fmax 导致的索引丢失
                if isinstance(result_df, pd.DataFrame):
                    if not result_df.index.equals(close_matrix.index):
                        result_df.index = close_matrix.index
                    if not result_df.columns.equals(close_matrix.columns):
                        result_df.columns = close_matrix.columns
                factor_tensor[fid] = result_df
            except Exception:
                factor_tensor[fid] = pd.DataFrame(
                    np.nan, index=close_matrix.index, columns=close_matrix.columns
                )
            if verbose:
                shape = factor_tensor[fid].shape
                nan_pct = factor_tensor[fid].isna().mean().mean()
                print(f"形状 {shape} [panel] NaN={nan_pct:.1%}")
        else:
            # ── 逐股票模式：与历史兼容 ──
            factor_rows = {}
            for sym, df in all_dfs.items():
                try:
                    result = fn(df, **kwargs)
                    # 处理 ndarray / 索引丢失（np.where 导致）
                    if isinstance(result, np.ndarray):
                        result = pd.Series(result, index=df.index)
                    elif isinstance(result, pd.Series) and len(result) == len(df):
                        result = pd.Series(result.values, index=df.index)
                    factor_rows[sym] = result
                except Exception:
                    factor_rows[sym] = pd.Series(np.nan, index=df.index)
            mat = pd.DataFrame(factor_rows).sort_index()
            factor_tensor[fid] = mat
            if verbose:
                print(f"形状 {mat.shape}")

    return close_matrix, factor_tensor
