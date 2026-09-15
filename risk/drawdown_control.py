"""
risk/drawdown_control.py — 回撤触发的仓位控制（两种响应形态 + 闭环 + **闭合账本**）。

背景
----
全仓 grep `stop_loss|止损|max_drawdown_limit|trailing_stop` **零命中** —— 框架此前
没有任何回撤控制机制（CLAUDE.md TODO 20）。本模块提供两个参数化组件。

两种响应形态（回答「能不能自适应」）
------------------------------------
| 函数 | 响应形态 | 参数 |
|---|---|---|
| `apply_drawdown_control` | **两态开关 + 滞后带** | threshold / cut / recovery |
| `apply_drawdown_scaling` | **连续线性映射**（自适应形态，无台阶） | max_cut_at / floor |

后者即「自适应」形态：仓位随回撤**连续**变化，天然无抖动。但**参数本身仍是固定的** ——
让阈值也随波动率自适应会再引入两个参数，样本内更易拟合出好看曲线（铁律 5）。
「连续 vs 两态」在匹配仓位下的对比见 `run_drawdown_control.py`。

账本是闭合的（三层）
--------------------
`controlled_ret` **就是净收益**，已扣掉调杠杆成本，不是「只记仓位账」的半成品：

    仓位账  `position_ret = scale × raw_ret`        （持仓等比缩放）
    费用账  `relever_cost = |Δscale| × gross × 费率`（调杠杆要再交易一次）
    现金账  `cash_ret = (1 − scale) × 0`             （未投入部分按现金，**零息是显式约定**）

    `controlled_ret = 仓位账 − 费用账 + 现金账`

为何必须记费用账：把账本从 `scale[t−1]` 调到 `scale[t]`，等价于对每个持仓按 `Δscale`
再交易一次 —— 增仓按买入费率、减仓按卖出费率。漏记它会**系统性高估**控制效果
（实测夏普被高估 0.15~0.18）。且**净值与回撤都按净收益路径推进**，所以状态反馈与
账本一致。`close_overlay_ledger` / `assert_overlay_closed` 可独立复核这一点。

⚠️ `gross` 是模型假设（本函数只看到收益序列，看不到权重矩阵）：LS 约 2.0、纯多头约 1.0。
默认 2.0 对齐本仓库主流的多空口径 —— **纯多头策略必须显式传 `gross=1.0`**。

闭环与无未来函数（铁律 1）
--------------------------
回撤在**受控（净）路径**上计算（实盘观察到的就是自己账户的回撤；开环会低估控制效果），
第 t 日决策只依赖截至第 t−1 日收盘的回撤 —— 循环内先决策、后更新净值。

用法
----
    from risk.drawdown_control import apply_drawdown_control, apply_drawdown_scaling

    step = apply_drawdown_control(port_ret, threshold=0.03, cut=0.3, recovery=0.015)
    cont = apply_drawdown_scaling(port_ret, max_cut_at=0.02, floor=0.2)
    net = cont["controlled_ret"]        # 已闭合的净收益
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from risk.cost_model import BUY_COST, SELL_COST

__all__ = ["apply_drawdown_control", "apply_drawdown_scaling", "relevering_cost"]

# 规则签名：(前一日受控回撤, 当前状态) -> (当日仓位系数, 新状态)
_DrawdownRule = Callable[[float, float], "tuple[float, float]"]


def _simulate_closed_loop(
    returns: pd.Series,
    rule: _DrawdownRule,
    *,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
    include_relever_cost: bool = True,
) -> pd.DataFrame:
    """闭环模拟 + 三层账本合成。`rule(drawdown_prev, state) -> (scale, next_state)`。

    无状态规则可忽略 `state`。`include_relever_cost=False` 只用于诊断
    （复现「只记仓位账」的旧口径），**生产必须保持 True**。

    Notes
    -----
    状态机 + 净值反馈是**时序依赖**的，无法向量化 —— 与 `build_weight_portfolio` 的
    `trade_limits` 逐日撮合同理，属合理保留的逐日循环；单序列长度 ~4e3，开销可忽略。
    """
    if not (gross > 0.0):
        raise ValueError(f"gross 必须 > 0，得到 {gross}")

    r = pd.Series(returns).dropna()
    if r.empty:
        raise ValueError("returns 为空，无法施加回撤控制")

    raw = r.to_numpy(dtype=float)
    n = raw.size
    scales = np.empty(n, dtype=float)
    costs = np.empty(n, dtype=float)
    dds = np.empty(n, dtype=float)

    state = 1.0
    prev_scale = 1.0      # 起始为满仓 → 首日若降仓，按卖出费率计费
    equity = 1.0
    peak = 1.0
    prev_dd = 0.0         # 第 0 日无历史回撤 → 满仓

    for i in range(n):
        scale, state = rule(prev_dd, state)                  # ① 先决策
        d_scale = scale - prev_scale
        if include_relever_cost and d_scale != 0.0:
            rate = buy_cost if d_scale > 0.0 else sell_cost
            cost = abs(d_scale) * gross * rate               # ② 费用账
        else:
            cost = 0.0
        net = raw[i] * scale - cost                          # ③ 仓位账 − 费用账
        equity *= (1.0 + net)                                #    净值按净收益推进
        if equity > peak:
            peak = equity
        prev_dd = equity / peak - 1.0                        # ④ 收盘后更新，供次日
        scales[i] = scale
        costs[i] = cost
        dds[i] = prev_dd
        prev_scale = scale

    scale_s = pd.Series(scales, index=r.index)
    cost_s = pd.Series(costs, index=r.index)
    drawdown = pd.Series(dds, index=r.index)

    position_ret = r * scale_s
    cash_ret = (1.0 - scale_s) * 0.0
    net_ret = position_ret - cost_s + cash_ret

    return pd.DataFrame({
        "scale": scale_s,
        "raw_ret": r,
        "position_ret": position_ret,   # 仓位账（未扣调杠杆成本）
        "relever_cost": cost_s,         # 费用账
        "cash_ret": cash_ret,           # 现金账（零息约定）
        "cash_weight": 1.0 - scale_s,
        "net_ret": net_ret,
        "controlled_ret": net_ret,      # = 净收益（闭合）
        "drawdown": drawdown,
        "drawdown_prev": drawdown.shift(1).fillna(0.0),
    })


def apply_drawdown_control(
    returns: pd.Series,
    *,
    threshold: float = 0.10,
    cut: float = 0.5,
    recovery: float = 0.05,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
    include_relever_cost: bool = True,
) -> pd.DataFrame:
    """按回撤深度缩放仓位（**两态开关 + 滞后带**，闭环，**账本闭合**）。

    Parameters
    ----------
    returns : pd.Series
        原始组合日收益序列（会 `dropna`）。
    threshold : float, default 0.10
        触发降仓的回撤深度（正数表示 10%）。
    cut : float, default 0.5
        触发后的仓位系数，须落在 `(0, 1]`；`1.0` 即等于不控制。
    recovery : float, default 0.05
        恢复阈值，须满足 `0 ≤ recovery < threshold`（否则滞后带退化、状态机抖动）。
    gross : float, default 2.0
        被缩放账本的总杠杆 `Σ|w|`。**纯多头须传 1.0**。
    include_relever_cost : bool, default True
        是否计入费用账。`False` 仅用于诊断，生产保持 True。

    Returns
    -------
    pd.DataFrame
        `controlled_ret` 为**闭合净收益**；其余列见模块 docstring 的三层账本。
    """
    if not (0.0 <= recovery < threshold):
        raise ValueError(
            f"要求 0 ≤ recovery < threshold，得到 recovery={recovery}, threshold={threshold}"
            "（否则滞后带退化，状态机将在同一阈值上抖动）")
    if not (0.0 < cut <= 1.0):
        raise ValueError(f"cut 必须落在 (0, 1]，得到 {cut}")

    def rule(dd_prev: float, state: float) -> tuple[float, float]:
        if state == 1.0 and dd_prev <= -threshold:
            return cut, cut
        if state != 1.0 and dd_prev >= -recovery:
            return 1.0, 1.0
        return state, state

    return _simulate_closed_loop(
        returns, rule, gross=gross, buy_cost=buy_cost, sell_cost=sell_cost,
        include_relever_cost=include_relever_cost)


def apply_drawdown_scaling(
    returns: pd.Series,
    *,
    max_cut_at: float = 0.20,
    floor: float = 0.30,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
    include_relever_cost: bool = True,
) -> pd.DataFrame:
    """按回撤深度**连续**缩放仓位（自适应形态，闭环，**账本闭合**）。

    映射：`scale = clip(1 + dd_prev / max_cut_at, floor, 1)`

        dd = 0            → 1.0（满仓）
        dd = −max_cut_at  → floor（降至下限）
        中间              → 线性插值

    连续映射**不需要滞后带**也不会抖动。相对两态开关是「早减、缓减」而非「晚减、猛减」；
    边界扫描（`run_drawdown_control.py`）显示最优在 `max_cut_at≈0.02` 的**内部点**。

    Parameters
    ----------
    max_cut_at : float, default 0.20
        仓位降到下限时的回撤深度（正数），须 > 0。
    floor : float, default 0.30
        仓位下限，须落在 `(0, 1]`；`1.0` 即等于不控制。
    gross, buy_cost, sell_cost, include_relever_cost
        同 `apply_drawdown_control`。

    Returns
    -------
    pd.DataFrame
        同 `apply_drawdown_control`。
    """
    if not (max_cut_at > 0.0):
        raise ValueError(f"max_cut_at 必须 > 0，得到 {max_cut_at}")
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"floor 必须落在 (0, 1]，得到 {floor}")

    def rule(dd_prev: float, _state: float) -> tuple[float, float]:
        s = 1.0 + dd_prev / max_cut_at
        s = min(1.0, max(floor, s))
        return s, s

    return _simulate_closed_loop(
        returns, rule, gross=gross, buy_cost=buy_cost, sell_cost=sell_cost,
        include_relever_cost=include_relever_cost)


def relevering_cost(
    scale: pd.Series,
    *,
    gross: float = 2.0,
    buy_cost: float = BUY_COST,
    sell_cost: float = SELL_COST,
) -> pd.Series:
    """按仓位系数的逐日变化**独立重算**「调杠杆」费用账。

    把整个账本从 `scale[t−1]` 调到 `scale[t]`，等价于对每个持仓按 `Δscale` 再交易一次：
    增仓按买入费率、减仓按卖出费率，成交规模为 `|Δscale| × gross`。

    这是**费用账的独立真源**：`_simulate_closed_loop` 在循环内逐日累计同一口径，
    `close_overlay_ledger` 用本函数重算，`assert_overlay_closed` 比对二者 ——
    与「对账必须独立重算」同一条原则。

    Returns
    -------
    pd.Series
        逐日调杠杆成本（与 `scale` 同索引，首日按「起始满仓」计）。
    """
    d = pd.Series(scale).diff().fillna(0.0)
    up = d.clip(lower=0.0) * gross * buy_cost
    down = (-d).clip(lower=0.0) * gross * sell_cost
    return up + down
