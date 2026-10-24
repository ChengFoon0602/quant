"""
批量截面因子计算器。

将多只股票的 OHLCV 数据加载并按给定因子 ID 计算，输出截面矩阵，
供回测引擎直接使用。

两种计算模式：
- per-stock: 逐只股票调用 factor(df)（无 RANK 因子，与历史兼容）
- panel:    构建宽表 dict[str, DataFrame]，一次调用 factor(panel)，
            使 RANK 成为真正的截面排名（axis=1 跨股票排名）。

复权口径（2026-10-24 修复）：
- 因子计算用**后复权**（`factor_adjust="1"`）——铁律 3.5，前复权会把未来
  除权事件带进历史截面。
- 返回给调用方的 close_matrix 默认仍是前复权（`price_adjust="2"`），
  保证回测口径与既有报告可比；收益率在两种复权下一致，涨跌停判断
  （`risk/portfolio.py`）是相对判断，不受影响。
- VWAP 不能直接用 `amount / volume`（那是未复权均价），见 `data/price_adjust.py`。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fetcher import load_daily
from data.price_adjust import adjust_factor, build_true_vwap
from .factors import factor_001 as f001  # 用于 get_factor_func
from .factors import _needs_panel


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
    vwap_mode: str = "true",
) -> dict[str, pd.DataFrame]:
    """将 {symbol: DataFrame} 转换为宽表面板 {field: DataFrame(date × stocks)}。

    vwap_mode:
      - "true"：从 adj1/adj3 复权因子还原真后复权 VWAP（默认，研报口径）
      - "tp"  ：典型价代理 (high+low+close)/3，仅在 adj1/adj3 缓存缺失时使用
    """
    fields = ["open", "high", "low", "close", "volume", "amount"]
    panel = {}
    for field in fields:
        mat = pd.DataFrame(
            {sym: df[field] for sym, df in all_dfs.items()}
        ).sort_index()
        panel[field] = mat

    if vwap_mode == "true":
        panel["vwap"] = build_true_vwap(panel, list(all_dfs))
    elif vwap_mode == "tp":
        panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    else:
        raise ValueError(f"未知 vwap_mode: {vwap_mode}（应为 'true' 或 'tp'）")
    return panel


def compute_panel_factor(
    all_dfs: dict[str, pd.DataFrame],
    fn,
    vwap_mode: str = "true",
    **kwargs,
) -> pd.DataFrame:
    """用面板模式计算**单个**因子函数（`RANK` = 真正的截面排名）。

    供「已自行加载 all_dfs」的脚本使用，替代原先的
    `for sym, df in all_dfs.items(): fn(df)` 写法——后者会让 RANK 退化成
    单股票时序排名。

    要求
        all_dfs 必须是**后复权**数据（`load_daily(sym, adjust="1")`）。
        若传入前复权数据，`_build_panel` 用 adj1/adj3 复权因子还原的 vwap
        会与 close 量纲不一致，因子值不可信。
    """
    panel = _build_panel(all_dfs, vwap_mode=vwap_mode)
    out = fn(panel, **kwargs)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(
            out, index=panel["close"].index, columns=panel["close"].columns
        )
    return out.sort_index()


def compute_factor_matrix(
    symbols: list[str],
    factor_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    verbose: bool = True,
    factor_kwargs: dict[str, dict] | None = None,
    factor_adjust: str = "1",
    price_adjust: str = "2",
    vwap_mode: str = "true",
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算给定 symbols × factors 的截面因子矩阵。

    Parameters
        symbols: 股票代码列表，如 ["000001", "600000"]
        factor_ids: 因子 ID 列表，如 ["alpha001", "alpha072"]
        start, end: 日期筛选 YYYY-MM-DD，None = 全部
        verbose: 打印进度
        factor_kwargs: 因子参数字典，如 {"alpha055": {"delta_days": 10}}
        factor_adjust: 因子计算用的复权方式，默认 "1"（后复权，铁律 3.5）
        price_adjust: 返回 close_matrix 的复权方式，默认 "2"（前复权，与既有
                      回测口径一致）。收益率在两种复权下相同。
        vwap_mode: "true"=真后复权 VWAP（需要 adj1/adj3 缓存）/ "tp"=典型价代理
        strict: panel 模式因子失败时是否抛异常（True 抛，False 只 warning）

    注意
        自 2026-10-24 起因子默认改用后复权计算，与旧版的数字**不可直接比较**。
        旧版用前复权价 + 未复权 amount 拼 VWAP，导致 15 个因子退化、
        价格水平类因子含前视信息。详见 docs/alpha191_复现审计_2026-10-24.md。

    Returns
        (close_matrix, factor_tensor):
            close_matrix — index=date, columns=symbols
            factor_tensor — dict[factor_id] -> DataFrame[index=date, columns=symbols]
    """
    # 加载所有股票数据（因子口径 = 后复权；回测口径默认前复权）
    all_dfs = {}
    close_data = {}
    missing: list[str] = []
    cols = ["open", "high", "low", "close", "volume", "amount"]

    for sym in symbols:
        df_raw = load_daily(sym, adjust=factor_adjust)
        if df_raw is None or not set(cols).issubset(df_raw.columns):
            missing.append(sym)
            continue
        df = df_raw[cols].copy()
        if start is not None:
            df = df.loc[df.index >= start]
        if end is not None:
            df = df.loc[df.index <= end]
        if len(df) < 100:
            missing.append(sym)
            continue

        # 注入与后复权价同口径的 vwap，供 per-stock 模式下的 _ensure_vwap 直接取用。
        # 否则它会退回 amount/volume（未复权），量纲与 close 不一致。
        f = adjust_factor(sym)
        if f is not None:
            df["vwap"] = (df["amount"] / df["volume"]) * f.reindex(df.index)

        all_dfs[sym] = df

        if price_adjust == factor_adjust:
            close_data[sym] = df["close"]
        else:
            px = load_daily(sym, adjust=price_adjust)
            close_data[sym] = (px["close"].reindex(df.index) if px is not None
                               else df["close"])

    if not close_data:
        raise RuntimeError(
            f"没有可用的股票数据（因子口径 adj{factor_adjust}）。"
            f"先运行 data/download_adjust_cache.py 补齐缓存。"
        )
    if missing:
        warnings.warn(
            f"{len(missing)}/{len(symbols)} 只股票缺 adj{factor_adjust} 缓存或数据不足，"
            f"已跳过（前 5 只：{missing[:5]}）"
        )

    close_matrix = pd.DataFrame(close_data).sort_index()

    # 预建面板（RANK 因子需要截面数据）
    panel = _build_panel(all_dfs, vwap_mode=vwap_mode)

    # 对每个因子计算
    factor_tensor = {}
    panel_failures: dict[str, str] = {}
    for fid in factor_ids:
        if verbose:
            print(f"  计算 {fid} ...", end=" ", flush=True)
        fn = get_factor_func(fid)
        kwargs = (factor_kwargs or {}).get(fid, {})

        num = int(fid.replace("alpha", "").replace("Alpha", ""))
        needs_panel = _needs_panel(num)

        if needs_panel:
            # ── 面板模式：RANK = 截面排名 ──
            try:
                result_df = fn(panel, **kwargs)
                # 修复 np.where/np.fmax 导致的索引丢失
                if isinstance(result_df, pd.DataFrame):
                    if not result_df.index.equals(close_matrix.index):
                        result_df.index = close_matrix.index
                    if not result_df.columns.equals(close_matrix.columns):
                        result_df.columns = close_matrix.columns
                factor_tensor[fid] = result_df
            except Exception as e:
                panel_failures[fid] = f"{type(e).__name__}: {e}"
                factor_tensor[fid] = pd.DataFrame(
                    np.nan, index=close_matrix.index, columns=close_matrix.columns
                )
            if verbose:
                shape = factor_tensor[fid].shape
                nan_pct = factor_tensor[fid].isna().mean().mean()
                print(f"形状 {shape} [panel] NaN={nan_pct:.1%}")
        else:
            # ── 逐股票模式：TSRANK 等时序算子，两种模式等价 ──
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

    # panel 模式失败必须暴露：静默填 NaN 会一路伪装成「弱信号」流进 IC 统计
    if panel_failures:
        detail = "\n".join(f"  {k}: {v}" for k, v in panel_failures.items())
        msg = f"{len(panel_failures)} 个因子在 panel 模式失败:\n{detail}"
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg)

    return close_matrix, factor_tensor
