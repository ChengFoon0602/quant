"""
risk/drawdown_control.py — 回撤触发的仓位控制（两种响应形态 + 闭环）。

背景
----
全仓 grep `stop_loss|止损|max_drawdown_limit|trailing_stop` **零命中** —— 框架此前
没有任何回撤控制机制（CLAUDE.md TODO 20）。

本模块提供**两个参数化组件**，**均不接入生产**：是否使用、用哪组参数由每个策略显式决定，
与 `gate` / `position_scale` 同属 opt-in 覆盖层。

两种响应形态（回答「能不能自适应」）
------------------------------------
| 函数 | 响应形态 | 参数 |
|---|---|---|
| `apply_drawdown_control` | **两态开关 + 滞后带**（跌破阈值降仓、回到带内恢复） | threshold / cut / recovery |
| `apply_drawdown_scaling` | **连续线性映射**（回撤越深仓位越低，无台阶） | max_cut_at / floor |

后者即「自适应」形态：仓位随回撤**连续**变化，且天然无抖动（无需滞后带）。
但注意：**参数本身仍是固定的** —— 若让阈值也随波动率自适应（`threshold_t = k·σ_t`），
会再引入 `k` 与 lookback 两个参数，在样本内更容易拟合出好看的结果（铁律 5）。
先比较「连续 vs 两态」在**匹配平均仓位**下的差异，再决定是否值得加这一层。

闭环（关键设计取舍）
--------------------
回撤在**受控后**的净值路径上计算，而不是原始路径。理由：实盘里你观察到的就是自己
账户的回撤，降仓后回撤变浅、恢复会提前 —— 开环（按未受控回撤退仓）会**低估**控制效果。
代价是路径依赖，只能顺序模拟。

无未来函数（铁律 1）
--------------------
第 t 日的缩放系数只依赖**截至第 t−1 日收盘**的受控回撤：循环内先决策、后更新净值。

用法
----
    from risk.drawdown_control import apply_drawdown_control, apply_drawdown_scaling

    step = apply_drawdown_control(port_ret, threshold=0.03, cut=0.3, recovery=0.015)
    cont = apply_drawdown_scaling(port_ret, max_cut_at=0.10, floor=0.4)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["apply_drawdown_control", "apply_drawdown_scaling"]

# 规则签名：(前一日受控回撤, 当前状态) -> (当日仓位系数, 新状态)
_DrawdownRule = Callable[[float, float], "tuple[float, float]"]


def _simulate_closed_loop(returns: pd.Series, rule: _DrawdownRule) -> pd.DataFrame:
    """闭环模拟：先决策（只用截至前一日的回撤），后按系数吃当日收益。

    `rule(drawdown_prev, state) -> (scale, next_state)`。
    无状态规则可忽略 `state`。

    Notes
    -----
    状态机 + 净值反馈是**时序依赖**的，无法向量化 —— 与 `build_weight_portfolio` 的
    `trade_limits` 逐日撮合同理，属合理保留的逐日循环；单序列长度 ~4e3，开销可忽略。
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        raise ValueError("returns 为空，无法施加回撤控制")

    raw = r.to_numpy(dtype=float)
    n = raw.size
    scales = np.empty(n, dtype=float)
    dds = np.empty(n, dtype=float)

    state = 1.0
    equity = 1.0
    peak = 1.0
    prev_dd = 0.0   # 第 0 日无历史回撤 → 满仓

    for i in range(n):
        scale, state = rule(prev_dd, state)          # ① 先决策
        equity *= (1.0 + raw[i] * scale)             # ② 再吃当日收益
        if equity > peak:
            peak = equity
        prev_dd = equity / peak - 1.0                # ③ 收盘后更新，供次日
        scales[i] = scale
        dds[i] = prev_dd

    scale_s = pd.Series(scales, index=r.index)
    drawdown = pd.Series(dds, index=r.index)

    return pd.DataFrame({
        "scale": scale_s,
        "raw_ret": r,
        "controlled_ret": r * scale_s,
        "drawdown": drawdown,
        "drawdown_prev": drawdown.shift(1).fillna(0.0),
    })


def apply_drawdown_control(
    returns: pd.Series,
    *,
    threshold: float = 0.10,
    cut: float = 0.5,
    recovery: float = 0.05,
) -> pd.DataFrame:
    """按回撤深度缩放仓位（**两态开关 + 滞后带**，闭环）。

    Parameters
    ----------
    returns : pd.Series
        原始组合日收益序列（会 `dropna`）。
    threshold : float, default 0.10
        触发降仓的回撤深度（正数表示 10%）。
    cut : float, default 0.5
        触发后的仓位系数，须落在 `(0, 1]`；`1.0` 即等于不控制。
    recovery : float, default 0.05
        恢复阈值（回撤收敛到该深度以内则恢复满仓），须满足 `0 ≤ recovery < threshold`。

    Returns
    -------
    pd.DataFrame
        columns: `scale`、`raw_ret`、`controlled_ret`、`drawdown`（受控路径回撤）、
        `drawdown_prev`（决策所用的前一日回撤）。
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

    return _simulate_closed_loop(returns, rule)


def apply_drawdown_scaling(
    returns: pd.Series,
    *,
    max_cut_at: float = 0.20,
    floor: float = 0.30,
) -> pd.DataFrame:
    """按回撤深度**连续**缩放仓位（自适应形态，闭环）。

    映射：`scale = clip(1 + dd_prev / max_cut_at, floor, 1)`

        dd = 0            → 1.0（满仓）
        dd = −max_cut_at  → floor（降至下限）
        中间              → 线性插值

    因为是连续映射，**不需要滞后带**也不会抖动。相较
    `apply_drawdown_control` 的两态开关，它对浅回撤就会开始减仓，
    但减得更少 —— 是「早减、缓减」而非「晚减、猛减」。

    Parameters
    ----------
    returns : pd.Series
        原始组合日收益序列（会 `dropna`）。
    max_cut_at : float, default 0.20
        仓位降到下限时的回撤深度（正数），须 > 0。
    floor : float, default 0.30
        仓位下限，须落在 `(0, 1]`；`1.0` 即等于不控制。

    Returns
    -------
    pd.DataFrame
        与 `apply_drawdown_control` 同 schema。
    """
    if not (max_cut_at > 0.0):
        raise ValueError(f"max_cut_at 必须 > 0，得到 {max_cut_at}")
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"floor 必须落在 (0, 1]，得到 {floor}")

    def rule(dd_prev: float, _state: float) -> tuple[float, float]:
        s = 1.0 + dd_prev / max_cut_at
        s = min(1.0, max(floor, s))
        return s, s

    return _simulate_closed_loop(returns, rule)
