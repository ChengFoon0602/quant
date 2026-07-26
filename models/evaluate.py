"""
models/evaluate.py — 模型无关的 OOF 评估模块。

给定模型对样本的预测分数（1=买入概率，0=低配/做空概率），按日期做截面评估：
  1. 日度 Rank IC：分数与未来 5 日收益的 rank correlation
  2. 五分位组收益：每天按分数把股票分 5 组，每组等权持有 5 日
  3. 多空价差：Top - Bottom 五分位，扣除双边成本
  4. 累计权益曲线、年化收益、夏普、最大回撤

所有评估遵守未来函数铁律：分数日在 t，收益从 t+1 开始算。

用法:
    from models.evaluate import evaluate_oof
    metrics, curves = evaluate_oof(pred_df, close_matrix, fwd_days=5)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# 项目默认费率：买入 0.026%，卖出 0.076%，双边合计 ≈0.1%。滑点默认 0.05%。
COST_LONG_OPEN = 0.00026   # 买入佣金 + 过户费
COST_LONG_CLOSE = 0.00076  # 卖出佣金 + 印花税 + 过户费
COST_SHORT_OPEN = 0.00026  # 做空开仓（这里简化为股票借贷成本≈0）
COST_SHORT_CLOSE = 0.00076
SLIPPAGE = 0.0005


def compute_daily_rank_ic(pred_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """逐日计算分数与未来收益的 Rank IC。"""
    common_dates = pred_df.index.intersection(fwd_ret.index)
    common_cols = pred_df.columns.intersection(fwd_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = fwd_ret.loc[common_dates, common_cols]

    ics = []
    dates = []
    for d in common_dates:
        pv = p.loc[d]
        rv = r.loc[d]
        mask = pv.notna() & rv.notna()
        if mask.sum() < 10:
            continue
        ic = pv[mask].rank().corr(rv[mask].rank())
        if pd.isna(ic):
            continue
        ics.append(ic)
        dates.append(d)
    return pd.Series(ics, index=dates, name="rank_ic")


def compute_quintile_returns(
    pred_df: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """计算每天各分位组的未来收益。

    对每一天，按 pred 分数分 n_quintiles 组，每组等权。
    返回 DataFrame (date × quintile)，其中 1=bottom, n=top。
    """
    common_dates = pred_df.index.intersection(fwd_ret.index)
    common_cols = pred_df.columns.intersection(fwd_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = fwd_ret.loc[common_dates, common_cols]

    records = []
    for d in common_dates:
        pv = p.loc[d]
        rv = r.loc[d]
        mask = pv.notna() & rv.notna()
        if mask.sum() < n_quintiles * 3:
            continue
        try:
            labels = pd.qcut(pv[mask], n_quintiles, labels=False, duplicates="drop")
        except Exception:
            continue
        daily_ret = rv[mask].groupby(labels).mean()
        row = {f"q{i+1}": daily_ret.get(i, np.nan) for i in range(n_quintiles)}
        row["date"] = d
        records.append(row)
    if not records:
        return pd.DataFrame(columns=[f"q{i+1}" for i in range(n_quintiles)])
    return pd.DataFrame(records).set_index("date").sort_index()


def compute_long_short_curve(
    quintile_ret: pd.DataFrame,
    cost: float | None = None,
) -> pd.DataFrame:
    """多空 (Top - Bottom) 日收益曲线。

    默认 cost = 双边交易成本：long 开仓+平仓 + short 开仓+平仓 + 2×滑点。
    """
    if cost is None:
        cost = (
            COST_LONG_OPEN + COST_LONG_CLOSE
            + COST_SHORT_OPEN + COST_SHORT_CLOSE
            + 2 * SLIPPAGE
        )
    q_cols = [c for c in quintile_ret.columns if c.startswith("q")]
    n = len(q_cols)
    top = quintile_ret[f"q{n}"]
    bottom = quintile_ret["q1"]
    ls_ret = top - bottom - cost
    cum = (1 + ls_ret.fillna(0)).cumprod()
    return pd.DataFrame({
        "long_short_ret": ls_ret,
        "long_short_cum": cum,
    })


def _performance_metrics(ret_series: pd.Series) -> dict:
    """给定日收益序列，返回年化收益、夏普、最大回撤。"""
    ret = ret_series.dropna()
    if len(ret) == 0:
        return {"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "n_days": 0}
    mean_ret = ret.mean()
    std_ret = ret.std()
    annual_ret = (1 + mean_ret) ** 252 - 1
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    return {
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": len(ret),
    }


def evaluate_oof(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    label_days: int = 5,
    hold_days: int = 1,
    n_quintiles: int = 5,
    cost: float | None = None,
) -> Tuple[dict, dict]:
    """主入口：评估 OOF 预测。

    Parameters
    ----------
    pred_df : pd.DataFrame
        模型预测分数 (date × stocks)，值越大越看好。
    close_matrix : pd.DataFrame
        收盘价矩阵，用于计算组合收益。
    label_days : int
        标签前瞻期（仅用于 Rank IC 对齐，确认预测目标）。
    hold_days : int
        组合持有期交易日数。默认 1 表示日频再平衡；
        大于 1 时使用 overlapped portfolio 收益（暂未实现，传入 >1 会报错）。
    n_quintiles : int
        分位数组数，默认 5。
    cost : float | None
        多空双边成本，None 用默认约 0.202%。

    Returns
    -------
    metrics : dict
        汇总指标。
    curves : dict
        rank_ic, quintile_ret, long_short 曲线 DataFrame。
    """
    if hold_days != 1:
        raise NotImplementedError("Only hold_days=1 (daily rebalancing) is currently supported")

    # 日频再平衡收益：close(t+2)/close(t+1) - 1
    # 信号日在 t，次日开盘买入，再次日收盘为第一笔可观测收益
    entry = close_matrix.shift(-1)
    exit_ = close_matrix.shift(-2)
    daily_ret = exit_ / entry - 1

    # 1. Rank IC：分数与日收益（label_days 后）的秩相关
    # 为与标签定义一致，Rank IC 仍看 label_days 后的收益
    entry_ic = close_matrix.shift(-1)
    exit_ic = close_matrix.shift(-(label_days + 1))
    fwd_ret_ic = exit_ic / entry_ic - 1
    rank_ic = compute_daily_rank_ic(pred_df, fwd_ret_ic)
    ic_mean = rank_ic.mean()
    ic_std = rank_ic.std(ddof=0)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_t = ic_ir * np.sqrt(len(rank_ic)) if ic_std > 0 else 0.0

    # 2. 五分位（日频再平衡）
    quintile_ret = compute_quintile_returns(pred_df, daily_ret, n_quintiles=n_quintiles)

    # 3. 多空曲线
    ls_df = compute_long_short_curve(quintile_ret, cost=cost)
    ls_metrics = _performance_metrics(ls_df["long_short_ret"])

    metrics = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_t": ic_t,
        "ic_days": len(rank_ic),
        "ls_annual_return": ls_metrics["annual_return"],
        "ls_sharpe": ls_metrics["sharpe"],
        "ls_max_drawdown": ls_metrics["max_drawdown"],
        "ls_n_days": ls_metrics["n_days"],
    }

    # 各分位统计
    for col in quintile_ret.columns:
        m = _performance_metrics(quintile_ret[col])
        metrics[f"{col}_annual_return"] = m["annual_return"]
        metrics[f"{col}_sharpe"] = m["sharpe"]

    curves = {
        "rank_ic": rank_ic,
        "quintile_ret": quintile_ret,
        "long_short": ls_df,
    }
    return metrics, curves


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetcher import load_daily, cache_summary

    cache = cache_summary()
    symbols = sorted(cache["symbol"].tolist())[:300]
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()

    # dummy random prediction for smoke test
    pred = pd.DataFrame(
        np.random.randn(*close_matrix.shape),
        index=close_matrix.index,
        columns=close_matrix.columns,
    )
    metrics, curves = evaluate_oof(pred, close_matrix)
    print(metrics)
