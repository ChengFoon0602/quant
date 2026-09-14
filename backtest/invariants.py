"""
backtest/invariants.py — 回测输入的运行时契约断言。

立场
----
确定性内核里「差不多」等于「错」。错误必须在**使用点**立刻中断，而不是降级成
warn 日志 —— 日志会被淹没，问题会静默通过。铁律 3 的 0.3% 成本 bug 就是在
109 项测试全绿的情况下存活，并把「多空 alpha 不成立」写进了三份报告。

与 tests/ 的分工
----------------
    tests/          守护「代码里的默认值 / 口径是否漂移」 —— 静态、事后
    invariants.py   守护「这一次真实输入是否合法」       —— 运行时、事前

实现约定
--------
1. 用**显式 raise**，不用 `assert` 语句：`python -O` 会剥离 assert 语句，
   而契约检查不允许被优化掉。
2. **不提供全局关闭开关**。任何 `strict=False` 都会退化成第二个 warn 日志。
   个别场景确需容错时，由调用点显式 try/except 并记录，责任留在调用点。
3. 只断言**必然成立**的性质（负价格、r < -1、索引乱序、权重 NaN ……），
   不断言「通常成立」的性质（如 close > 0），以免在真实数据上误报。
4. **零项目内依赖**（只用 numpy / pandas）。否则 risk/* 反向依赖本模块时会
   形成循环导入 —— 因为 `risk/__init__.py` 在最外层就 import 了 risk.portfolio。

用法
----
    from backtest.invariants import assert_backtest_inputs
    assert_backtest_inputs(close_matrix, buy_cost=buy_cost, sell_cost=sell_cost)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "InvariantViolation",
    "MAX_ROUND_TRIP_COST",
    "assert_price_panel",
    "assert_returns_panel",
    "assert_weight_matrix",
    "assert_cost_params",
    "assert_backtest_inputs",
]

# 成本防错上界（双边合计）。这是「防错上界」，不是「费率目标值」：
#   铁律 3 真值 ≈ 0.00102（买 0.00026 / 卖 0.00076）
#   ETF 正当口径  0.0004
#   历史事故值    0.003（铁律 3 倍，见 risk/cost_model.py 模块 docstring）
# 取 0.002 —— 高于任何正当口径，但能拦住 0.003 这类事故值。
MAX_ROUND_TRIP_COST: float = 0.002

_TOL = 1e-9
_REL = 1e-6


class InvariantViolation(AssertionError):
    """回测输入的契约违反。继承 AssertionError 以兼容既有测试风格。"""


def _fail(msg: str) -> None:
    raise InvariantViolation(msg)


def _as_frame(x: Union[pd.DataFrame, pd.Series], name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    _fail(f"{name}: 期望 DataFrame/Series，得到 {type(x).__name__}")


def assert_price_panel(close, *, name: str = "close_matrix") -> None:
    """价格面板契约：索引有序且唯一、价格非负、无全 NaN 列。"""
    df = _as_frame(close, name)
    if df.shape[0] < 2 or df.shape[1] < 1:
        _fail(f"{name}: 形状非法 {df.shape}（需 ≥2 行、≥1 列）")

    idx = df.index
    if not idx.is_unique:
        n_dup = int(idx.duplicated().sum())
        _fail(f"{name}: 索引存在 {n_dup} 个重复日期")
    if not idx.is_monotonic_increasing:
        _fail(f"{name}: 索引非单调递增 —— 存在乱序日期，pct_change/对齐语义不可信")

    vals = df.to_numpy(dtype=float)
    if np.isneginf(vals).any():
        _fail(f"{name}: 存在 -inf 价格")
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        _fail(f"{name}: 全为 NaN/inf，无任何有效价格")
    if (finite < 0).any():
        _fail(f"{name}: 存在 {int((finite < 0).sum())} 个负价格 —— 价格必非负")

    all_nan = df.isna().all(axis=0)
    if all_nan.any():
        bad = [str(c) for c in df.columns[all_nan][:5]]
        _fail(f"{name}: {int(all_nan.sum())} 列全为 NaN（如 {bad}）—— 疑似 join 错误")


def assert_returns_panel(daily_ret, *, name: str = "daily_ret") -> None:
    """收益面板契约：无 ±inf，且 pct_change 必然满足 r >= -1。"""
    df = _as_frame(daily_ret, name)
    vals = df.to_numpy(dtype=float)
    n_inf = int(np.isinf(vals).sum())
    if n_inf:
        _fail(f"{name}: 存在 {n_inf} 个 ±inf 收益（分母为 0 或除零？）")
    finite = vals[np.isfinite(vals)]
    if finite.size and (finite < -1.0 - _TOL).any():
        _fail(
            f"{name}: 存在 {int((finite < -1.0 - _TOL).sum())} 个 r < -1 的收益"
            f"（等价 1+r<0），不可能由价格比值产生"
        )


def assert_weight_matrix(
    W,
    *,
    max_gross: Optional[float] = None,
    max_weight_per_asset: Optional[float] = None,
    name: str = "W",
) -> None:
    """权重矩阵契约：无 NaN；可选校验总杠杆 Σ|w| 与单资产权重上限。"""
    df = _as_frame(W, name)
    n_nan = int(df.isna().to_numpy().sum())
    if n_nan:
        _fail(f"{name}: 存在 {n_nan} 个 NaN 权重 —— 权重必须显式给出，不允许隐式缺失")

    if max_gross is not None and df.shape[0]:
        gross = df.abs().sum(axis=1)
        bad = gross > max_gross * (1 + _REL) + _TOL
        if bad.any():
            _fail(
                f"{name}: {int(bad.sum())} 行总杠杆 Σ|w| 最大 {float(gross[bad].max()):.4f}"
                f" 超出上限 {max_gross}"
            )

    if max_weight_per_asset is not None and df.size:
        mx = float(df.abs().to_numpy().max())
        if mx > max_weight_per_asset * (1 + _REL) + _TOL:
            _fail(f"{name}: 单资产权重 {mx:.4f} 超出上限 {max_weight_per_asset}")


def assert_cost_params(buy_cost: float, sell_cost: float) -> None:
    """费率契约：非负、有限、双边合计未超防错上界。"""
    for nm, c in (("buy_cost", buy_cost), ("sell_cost", sell_cost)):
        if not np.isfinite(c):
            _fail(f"{nm}={c!r} 非有限值")
        if c < 0:
            _fail(f"{nm}={c!r} 为负 —— 费率必非负")

    rt = buy_cost + sell_cost
    if rt > MAX_ROUND_TRIP_COST + _TOL:
        _fail(
            f"双边成本 {rt:.5f}（{rt * 1e4:.1f}bps）超出防错上界 "
            f"{MAX_ROUND_TRIP_COST}（{MAX_ROUND_TRIP_COST * 1e4:.0f}bps）。"
            f"铁律 3 真值 ≈ 0.00102（买 0.00026 / 卖 0.00076）。"
            f"若确为超高摩擦场景，请修正上界常量并写明理由，而不是放行。"
        )


def assert_backtest_inputs(close, *, buy_cost=None, sell_cost=None) -> None:
    """一次校验回测入口的输入：价格面板 +（可选）费率。"""
    assert_price_panel(close)
    if (buy_cost is None) != (sell_cost is None):
        _fail("assert_backtest_inputs: buy_cost / sell_cost 必须同时给出")
    if buy_cost is not None:
        assert_cost_params(buy_cost, sell_cost)
