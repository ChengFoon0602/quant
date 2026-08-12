"""
test_monthly_sampling.py — 月末采样方法论验证（方向2 核心假设）。

验证内容:
  1. month_end_dates: 2010-2025 日频索引 → ~190 个月末截面（独立观测数）
  2. 季频因子日频前向填充 → 日频 IC_IR 系统性高于月末 IC_IR
     （IC 自相关虚高实证 —— 方向2 与价量链路分道的方法论证据）
  3. fwd_return(21): close(t+22)/close(t+1)-1 无未来函数

用合成数据（季度更新因子 + 真实信号 + 噪声）自包含，不依赖已拉取的真实数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "strategies" / "zz500_fundamental_trial"))
sys.path.insert(0, str(PROJECT_ROOT / "strategies" / "feature_selection"))

from purify import month_end_dates, fwd_return, _ic_ir_vec, _fm_vec  # noqa: E402


def _synthetic_panel(seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 2010-2025 合成面板：30 只股票，季频因子前向填充 + 真实信号。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2025-12-31")
    n_stocks = 30
    syms = [f"s{i:03d}" for i in range(n_stocks)]

    # 季度序列（每季度最后一个交易日）
    _s = pd.Series(dates, index=dates)
    quarter_last = _s.groupby(_s.dt.to_period("Q")).last()
    q_days = pd.DatetimeIndex(quarter_last.values)
    n_q = len(q_days)  # 64 个季度
    period_map = _s.dt.to_period("Q")
    p2q = {p: qi for qi, p in enumerate(period_map.unique())}
    q_day_idx = np.array([p2q[p] for p in period_map])

    # 每股基础 alpha（截面差异）+ 时变共同效应（有界 sin，避免信号爆炸）
    alpha_base = rng.normal(0, 1, n_stocks)
    time_effect = 0.3 * np.sin(np.arange(n_q) / 3.0)
    alpha_q = alpha_base[None, :] + time_effect[:, None]   # (n_q, n_stocks)

    # 收盘价：每日收益 = 该季度 alpha × scale + 噪声
    rets = alpha_q[q_day_idx] * 0.02 + rng.normal(0, 0.03, (len(dates), n_stocks))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)

    # 季频因子：每季度 alpha 前向填充到日频（模拟 _forward_fill_to_daily）
    factor_quarterly = pd.DataFrame(alpha_q, index=q_days, columns=syms)
    factor_daily = factor_quarterly.reindex(dates).ffill()
    return close, factor_daily


def test_month_end_count():
    dates = pd.bdate_range("2010-01-01", "2025-12-31")
    med = month_end_dates(dates)
    # 2010-2025 = 16 年 × 12 月 = 192 个月（允许 ±2 边界缺失）
    assert 185 <= len(med) <= 192, f"月末截面数 {len(med)} 不在 [185, 192]"
    print(f"  [PASS] month_end_dates: {len(med)} 个月末截面")


def test_daily_significance_inflated():
    """季频因子日频采样的统计显著性虚高（IC 自相关 → t 值通胀）。

    核心方法论点：同一信号在日频采样下，因子在季度内恒定 → 逐日 IC 序列
    强自相关（lag1 → ~1）→ 有效样本数虚增 → t 值被 sqrt(过采样) 放大。
    月末采样打破季度内自相关（lag1 → ~0），t 值才是真显著。
    因此【月调仓基本面因子必须用月末截面评估】——否则一个日频恰好过阈值的
    因子在月末会显著失败。
    """
    close, factor = _synthetic_panel()
    fwd = fwd_return(close, 21)
    med = month_end_dates(factor.index)

    ic_d = _ic_ir_vec(factor, fwd)
    ic_m = _ic_ir_vec(factor.loc[med], fwd.loc[med])

    # 1. IC 序列自相关：日频 lag1 ≈ 1（季度内冗余），月末 lag1 低（独立观测）
    def _lag1_autocorr(f, r):
        joint = f.notna() & r.notna()
        ics, didx = [], []
        for d in f.index:
            m = joint.loc[d]
            if m.sum() < 30:
                continue
            rp = f.loc[d][m].rank(); rr = r.loc[d][m].rank()
            ics.append(rp.corr(rr)); didx.append(d)
        return pd.Series(ics, index=didx).autocorr(1)

    ac_d = _lag1_autocorr(factor, fwd)
    ac_m = _lag1_autocorr(factor.loc[med], fwd.loc[med])
    assert ac_d > 0.5, f"日频 IC 应强自相关（lag1={ac_d:.2f} < 0.5）"
    assert ac_d > ac_m + 0.2, f"月末应打破自相关（日频 {ac_d:.2f} vs 月末 {ac_m:.2f}）"

    # 2. 显著性通胀：同一信号的 t 值日频 >> 月末（有效样本虚增）
    assert ic_d["t"] > 2 * ic_m["t"], (
        f"日频 t 值应显著虚高（{ic_d['t']:.1f} vs 月末 {ic_m['t']:.1f}）")
    print(f"  [PASS] 日频 t={ic_d['t']:.0f} (lag1={ac_d:.2f}) vs "
          f"月末 t={ic_m['t']:.0f} (lag1={ac_m:.2f}) → 显著性通胀 "
          f"{ic_d['t'] / ic_m['t']:.1f}x")


def test_fwd_return_no_lookahead():
    """fwd_return(21) 对齐：t 时刻的信号必须只用到 ≤t 的信息。"""
    close, _ = _synthetic_panel()
    fwd = fwd_return(close, 21)
    # 检查对齐：fwd.loc[t] = close(t+22)/close(t+1)-1（shift(-1) entry, shift(-22) exit）
    t0 = close.index[100]
    expected = close.loc[close.index[122], "s000"] / close.loc[close.index[101], "s000"] - 1
    got = fwd.loc[t0, "s000"]
    assert abs(got - expected) < 1e-9, f"fwd_return 对齐错误: {got} vs {expected}"
    print("  [PASS] fwd_return(21) = close(t+22)/close(t+1)-1，无未来函数")


if __name__ == "__main__":
    print("test_monthly_sampling: 月末采样方法论验证")
    test_month_end_count()
    test_fwd_return_no_lookahead()
    test_daily_significance_inflated()
    print("\n全部通过 ✓")
