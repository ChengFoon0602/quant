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

from backtest.invariants import InvariantViolation, assert_weight_matrix
from risk.cost_model import BUY_COST, SELL_COST
from risk.drawdown_control import relevering_cost

# 对账容差：各实现均用同一套 pandas 向量运算，浮点误差应在 1e-12 量级
TOL: float = 1e-10

__all__ = ["TOL", "reconcile_from_weights", "assert_closed", "assert_books_equal",
           "close_overlay_ledger", "assert_overlay_closed"]


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


def close_overlay_ledger(
    raw_ret: pd.Series,
    scale: pd.Series,
    *,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
) -> pd.DataFrame:
    """合成**覆盖层**（仓位缩放 + 现金）的三层账本 —— 覆盖层下的清算独立重算。

    一层账本挡不住覆盖层。加上回撤控制后，账本必须拆成三层：

        仓位账  `position_ret = scale × raw_ret`
                （持仓按 `scale` 等比缩放，毛收益与资产级费用同比例缩放）
        费用账  `relever_cost = |Δscale| × gross × 费率`
                （调杠杆本身要再交易一次：增仓按买入费率、减仓按卖出费率。
                  这一层**不在** `raw_ret` 里，也**不在** `scale × raw_ret` 里）
        现金账  `cash_ret = (1 − scale) × 0`
                （未投入部分按现金处理。**零息是本框架的显式约定，不是事实** ——
                  若要计货基收益，必须单独建模，不能塞进本恒等式）

    `净收益 = 仓位账 − 费用账 + 现金账`

    ⚠️ 本函数的用途是**暴露缺口**：`risk.drawdown_control` 的两个控制函数只产出
    **仓位账**（`scale × raw_ret`），**没有记费用账**。因此直接把它们的
    `controlled_ret` 当作净收益，覆盖层清算就**不闭合**，差额恰好等于 `relever_cost`
    （`assert_overlay_closed` 会把这个差额指出来）。
    实测量级：夏普被高估 0.15~0.18（见 `run_drawdown_control.py` 边界扫描）。

    Returns
    -------
    pd.DataFrame
        columns: `position_ret` / `relever_cost` / `cash_ret` / `net_ret` / `cash_weight`。
    """
    raw = pd.Series(raw_ret)
    sc = pd.Series(scale).reindex(raw.index).fillna(1.0)
    if not bool(((sc >= 0.0) & (sc <= 1.0)).all()):
        raise InvariantViolation("scale（仓位系数）必须落在 [0, 1]")

    position_ret = sc * raw
    relever = relevering_cost(sc, gross=gross, buy_cost=buy_cost, sell_cost=sell_cost)
    cash_ret = (1.0 - sc) * 0.0

    return pd.DataFrame({
        "position_ret": position_ret,
        "relever_cost": relever,
        "cash_ret": cash_ret,
        "net_ret": position_ret - relever + cash_ret,
        "cash_weight": 1.0 - sc,
    })


def assert_overlay_closed(
    res: pd.DataFrame,
    raw_ret: pd.Series,
    scale: pd.Series,
    *,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
    tol: float = TOL,
) -> dict[str, float]:
    """覆盖层清算闭合：`res` 的净收益必须等于三层账本的合成结果；失败即 raise。

    Parameters
    ----------
    res : pd.DataFrame
        待验证的受控账本，须含 `port_ret`（无则取 `net_ret`）。
    raw_ret : pd.Series
        **未加覆盖层**的原始净收益（即 `scale ≡ 1` 时的账本）。
    scale : pd.Series
        逐日仓位系数。
    gross : float
        被缩放账本的总杠杆 `Σ|w|`（LS 约 2.0，纯多约 1.0）。
    tol : float
        容差。

    Returns
    -------
    dict[str, float]
        `net_ret`（**判定项**，超容差即 raise）；`position_only_gap` 与
        `relever_cost_total` 是**诊断项**：前者是「若把仓位账当净收益会差多少」
        （按定义恰等于费用账），**预期非零，不是违规**。
    """
    ledger = close_overlay_ledger(raw_ret, scale, gross=gross,
                                 buy_cost=buy_cost, sell_cost=sell_cost)
    col = "port_ret" if "port_ret" in res.columns else "net_ret"
    if col not in res.columns:
        raise InvariantViolation(f"res 缺少 net_ret/port_ret 列，实际列：{list(res.columns)}")

    gaps = {
        "net_ret": _max_abs_gap(res[col], ledger["net_ret"], ledger.index),
        "position_only_gap": _max_abs_gap(res[col], ledger["position_ret"], ledger.index),
        "relever_cost_total": float(ledger["relever_cost"].sum()),
    }
    if gaps["net_ret"] > tol:
        lever = gaps["relever_cost_total"]
        cash = float(ledger["cash_weight"].mean())
        hint = ""
        if lever > tol:
            hint = (f"\n  费用账（调杠杆成本）合计 {lever:.6f}，平均现金占比 {cash:.3f}。"
                    "\n  ⚠️ 若 res 直接取自 risk.drawdown_control 的 controlled_ret 而该模块"
                    "漏记费用账，差额即此处。"
                    "\n     补法：`close_overlay_ledger(raw_ret, scale)['net_ret']`。")
        raise AssertionError(
            f"覆盖层清算不闭合：net_ret 最大偏差 {gaps['net_ret']:.3e} > {tol:.1e}{hint}")
    return gaps
