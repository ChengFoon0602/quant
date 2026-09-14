"""
backtest/reconciliation.py — 回测账本自洽性对账（单一真源）。

立场
----
组合收益是一本账，必须能自证闭合。三个对账项：

    仓位对账   持仓无 NaN，列集合与价格矩阵一致
    流水对账   turnover == Σ|ΔW|；cost == buy_turnover·BUY + sell_turnover·SELL
    盈亏核对   port_ret == gross_ret - cost；cum == Π(1+port_ret)

**对账方式必须是独立重算**：从实际持仓矩阵 `W_held` 重新推一遍损益，与实现给出的
结果逐位比对。用实现自己的中间量验证实现是空转 —— 与「测试必须断言真实代码，
不能断言自己写的常量」是同一个教训。

两处共用
--------
    tests/test_reconciliation.py   合成数据：快、可复现，随测试套件每次运行
    run_reconciliation.py          真实数据：慢、离线运行，对真实报告链路对账

用法
----
    from backtest.reconciliation import assert_closed
    gaps = assert_closed(result_df, W_held, close_matrix)   # 失败即 raise
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from backtest.invariants import assert_weight_matrix
from risk.cost_model import BUY_COST, SELL_COST

# 对账容差：各实现均用同一套 pandas 向量运算，浮点误差应在 1e-12 量级
TOL: float = 1e-10

__all__ = ["TOL", "reconcile_from_weights", "assert_closed", "assert_books_equal"]


def reconcile_from_weights(
    W_held: pd.DataFrame,
    close: pd.DataFrame,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
) -> pd.DataFrame:
    """从实际持仓矩阵独立重算参考账本。

    收益锚定与全仓库一致：`daily_ret[t] = close[t]/close[t-1] - 1`，
    组合收益 `port_ret[t] = W[t-1] · daily_ret[t]`。

    Parameters
    ----------
    W_held : pd.DataFrame
        实际持仓权重（index=date, columns=symbols）。
    close : pd.DataFrame
        收盘价矩阵。
    buy_cost, sell_cost : float
        方向分离费率，须与回测调用口径**完全一致**（含 `cost=` 对半拆场景）。

    Returns
    -------
    pd.DataFrame
        columns: gross_ret, cost, port_ret, turnover。
    """
    cols = W_held.columns.intersection(close.columns)
    daily_ret = close.loc[W_held.index, cols].pct_change()
    W = W_held[cols]

    gross_ret = (W.shift(1).fillna(0.0) * daily_ret).sum(axis=1)

    delta_w = W - W.shift(1).fillna(0.0)
    buy_turnover = delta_w.clip(lower=0.0).sum(axis=1)
    sell_turnover = (-delta_w).clip(lower=0.0).sum(axis=1)
    cost = buy_turnover * buy_cost + sell_turnover * sell_cost

    return pd.DataFrame({
        "gross_ret": gross_ret,
        "cost": cost,
        "port_ret": gross_ret - cost,
        "turnover": delta_w.abs().sum(axis=1),
    })


def _max_abs_gap(impl: pd.Series, expected: pd.Series, index) -> float:
    """两侧按 index 对齐后的最大绝对偏差；无有效值返回 0.0。"""
    a = impl.reindex(index).to_numpy(dtype=float)
    b = expected.reindex(index).to_numpy(dtype=float)
    if a.size == 0:
        return 0.0
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


def assert_closed(
    res: pd.DataFrame,
    W_held: pd.DataFrame,
    close: pd.DataFrame,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
    tol: float = TOL,
) -> dict[str, float]:
    """三类对账 + 净值路径一致性；失败即 raise。

    Parameters
    ----------
    res : pd.DataFrame
        回测结果，须含 `port_ret` / `turnover` / `cum`；
        若含 `gross_ret` 与 `cost`，则额外做盈亏闭合与毛/费对账。
    W_held : pd.DataFrame
        实际持仓权重矩阵。
    close : pd.DataFrame
        收盘价矩阵。
    buy_cost, sell_cost : float
        与回测调用一致的费率口径。
    tol : float
        容差，默认 1e-10。

    Returns
    -------
    dict[str, float]
        各项最大绝对偏差，供报告使用。

    Notes
    -----
    首行跳过：实现返回的 `W_held` 通常已被裁剪到 `port_ret.index`，
    丢失了首行前一日的持仓，参考账本无法复现首行 → 两侧一致地从第 2 行起比对。
    """
    assert_weight_matrix(W_held, name="W_held")
    if set(W_held.columns) != set(close.columns):
        raise AssertionError("仓位对账失败：持仓列集合与价格矩阵不一致")

    ref = reconcile_from_weights(W_held, close, buy_cost, sell_cost)
    idx = ref.index[1:]
    gaps: dict[str, float] = {}

    for name in ("turnover", "port_ret"):
        gaps[name] = _max_abs_gap(res[name], ref[name], idx)

    if "gross_ret" in res and "cost" in res:
        gaps["gross_ret"] = _max_abs_gap(res["gross_ret"], ref["gross_ret"], idx)
        gaps["cost"] = _max_abs_gap(res["cost"], ref["cost"], idx)
        # 盈亏核对：结果内部的逐日闭合（毛 - 费 = 净）
        gaps["close_identity"] = _max_abs_gap(
            res["port_ret"], res["gross_ret"] - res["cost"], res.index)

    # 净值由净收益唯一确定
    gaps["cum"] = _max_abs_gap(res["cum"], (1.0 + res["port_ret"]).cumprod(), res.index)

    bad = {k: v for k, v in gaps.items() if not (v <= tol)}
    if bad:
        detail = "；".join(f"{k}={v:.3e}" for k, v in bad.items())
        raise AssertionError(f"账本对账失败（容差 {tol:.1e}）：{detail}")
    return gaps


def assert_books_equal(
    res_a: pd.DataFrame,
    res_b: pd.DataFrame,
    keys: Sequence[str] = ("port_ret", "turnover"),
    tol: float = 1e-12,
) -> dict[str, float]:
    """两条实现的账本必须逐位一致（跨实现语义一致性）；失败即 raise。"""
    common = res_a.index.intersection(res_b.index)
    if len(common) == 0:
        raise AssertionError("跨实现比对失败：两条结果无公共日期")
    if len(res_a) != len(res_b):
        raise AssertionError(f"跨实现账本长度不同：{len(res_a)} vs {len(res_b)}")

    gaps: dict[str, float] = {}
    for k in keys:
        a = res_a.loc[common, k].to_numpy(dtype=float)
        b = res_b.loc[common, k].to_numpy(dtype=float)
        gap = float(np.abs(a - b).max()) if a.size else 0.0
        gaps[k] = gap
        if not (gap <= tol):
            raise AssertionError(f"跨实现账本不一致[{k}]：最大偏差 {gap:.3e} > {tol:.1e}")
    return gaps
