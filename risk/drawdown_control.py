"""
risk/drawdown_control.py — 回撤触发的降仓控制（参数化 + 滞后带 + 闭环）。

背景
----
全仓 grep `stop_loss|止损|max_drawdown_limit|trailing_stop` **零命中** —— 框架此前
没有任何回撤控制机制（CLAUDE.md TODO 20）。

本模块只提供**参数化组件**，**不接入生产**：是否使用、用哪组参数由每个策略显式决定，
与 `gate` / `position_scale` 同属 opt-in 覆盖层。

闭环（关键设计取舍）
--------------------
回撤在**受控后**的净值路径上计算，而不是原始路径。理由：实盘里你观察到的就是自己
账户的回撤，降仓后回撤变浅、恢复会提前 —— 开环（按未受控回撤退仓）会**低估**控制效果。
代价是路径依赖，只能顺序模拟（见下）。

无未来函数（铁律 1）
--------------------
第 t 日的缩放系数只依赖**截至第 t−1 日收盘**的受控回撤：循环内先决策、后更新净值，
`drawdown` 列即该时序的显式体现。

滞后带（避免抖动）
------------------
单阈值规则会在回撤贴着阈值时反复开关（chattering），徒增换手。故用两个阈值：

    跌破 −threshold      → 降仓到 cut
    回到 −recovery 以内  → 恢复到 1.0

要求 `0 ≤ recovery < threshold`。

用法
----
    from risk.drawdown_control import apply_drawdown_control
    ctl = apply_drawdown_control(port_ret, threshold=0.10, cut=0.5, recovery=0.05)
    controlled = ctl["controlled_ret"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["apply_drawdown_control"]


def apply_drawdown_control(
    returns: pd.Series,
    *,
    threshold: float = 0.10,
    cut: float = 0.5,
    recovery: float = 0.05,
) -> pd.DataFrame:
    """按回撤深度缩放仓位（滞后带状态机，闭环）。

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
        columns: `scale`（逐日仓位系数）、`raw_ret`、`controlled_ret`（= raw × scale）、
        `drawdown`（**受控路径**截至当日的回撤）、`drawdown_prev`（决策所用的前一日回撤）。

    Notes
    -----
    状态机 + 净值反馈是**时序依赖**的，无法向量化 —— 与 `build_weight_portfolio` 的
    `trade_limits` 逐日撮合同理，属合理保留的逐日循环；单序列长度 ~4e3，开销可忽略。
    """
    if not (0.0 <= recovery < threshold):
        raise ValueError(
            f"要求 0 ≤ recovery < threshold，得到 recovery={recovery}, threshold={threshold}"
            "（否则滞后带退化，状态机将在同一阈值上抖动）")
    if not (0.0 < cut <= 1.0):
        raise ValueError(f"cut 必须落在 (0, 1]，得到 {cut}")

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
        # ① 先决策：只用截至「前一日收盘」的受控回撤
        if state == 1.0 and prev_dd <= -threshold:
            state = cut
        elif state != 1.0 and prev_dd >= -recovery:
            state = 1.0
        # ② 再按系数吃当日收益
        controlled = raw[i] * state
        equity *= (1.0 + controlled)
        peak = equity if equity > peak else peak
        # ③ 收盘后更新回撤，供次日决策
        prev_dd = equity / peak - 1.0
        scales[i] = state
        dds[i] = prev_dd

    scale = pd.Series(scales, index=r.index)
    drawdown = pd.Series(dds, index=r.index)

    return pd.DataFrame({
        "scale": scale,
        "raw_ret": r,
        "controlled_ret": r * scale,
        "drawdown": drawdown,
        "drawdown_prev": drawdown.shift(1).fillna(0.0),
    })
