"""
strategies/pairs_trading/pairs.py — 配对交易：协整检验、配对筛选、信号与回测。

## 为什么不用 `risk/portfolio.build_weight_portfolio`

那是**截面选股**引擎（按因子分位等权持有 Top Q 只），而配对交易是**成对建仓**：
一个配对的仓位同时包含多腿与空腿，两者的相对权重由协整系数 β 决定，且必须
在协整破裂时无条件同时平仓。语义不同，故本模块自带状态机式回测。

## 核心风险（已实测）

协整关系在 A 股极不稳定：训练期通过 EG 检验的 125 对配对中，仅 44.7%（2018-2020
→ 2021-2022）与 22.8%（2020-2022 → 2023-2024）在样本外仍然协整。因此本模块
**不做固定训练期**，而是每个重估日对全部候选对重新检验，并允许持仓中途失效平仓。

## 口径

- 价格：后复权（`data.fetcher.load_daily(symbol, adjust="1")`）
- 成本：双边 0.1%（`risk.cost_model.ROUND_TRIP`），每腿各计一次
- 融券：**主口径不计**（方案 A，先验证价差回归本身有无 alpha），
  另设 `short_borrow_rate` 参数用于第二口径敏感性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: ADF 5% 临界值（含常数、无趋势，MacKinnon 大样本）
ADF_CRIT_5: float = -2.86


# ══════════════════════════════════════════════════════════════════
# 计量工具
# ══════════════════════════════════════════════════════════════════
def adf_tstat(x: np.ndarray, maxlag: int = 1) -> float:
    """增广 Dickey-Fuller 检验的 t 统计量（含常数项，无时间趋势）。

    回归式：$\\Delta x_t = a + b \\, x_{t-1} + \\sum_k c_k \\Delta x_{t-k} + e_t$
    返回 $b$ 的 t 值。$b$ 显著为负即拒绝单位根。
    """
    x = np.asarray(x, float)
    dx = np.diff(x)
    n = len(dx)
    if n < 30:
        return 0.0
    if maxlag > 0 and n > maxlag + 20:
        cols = [np.ones(n - maxlag), x[maxlag:-1]]
        for k in range(maxlag):
            cols.append(dx[maxlag - k - 1: n - k - 1])
        X = np.column_stack(cols)
        y = dx[maxlag:]
    else:
        X = np.column_stack([np.ones(n), x[:-1]])
        y = dx

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = X.shape[1]
    s2 = float(resid @ resid) / (len(y) - k)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        return 0.0
    return float(beta[1] / se[1]) if se[1] > 0 else 0.0


def ou_half_life(spread: np.ndarray) -> float:
    """OU 过程半衰期（天）。用 AR(1) 系数 $\\phi$ 估计：$T_{1/2} = -\\ln 2 / \\ln \\phi$。

    返回 `inf` 表示价差不回归（$\\phi \\ge 1$），返回 `nan` 表示样本不足。
    """
    s = np.asarray(spread, float)
    if len(s) < 30:
        return float("nan")
    y = s[1:]
    x = s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(b[1])
    if not np.isfinite(phi) or phi <= 0:
        return float("nan")
    if phi >= 1.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))


@dataclass
class CointResult:
    """一次 Engle-Granger 两步法的结果。"""
    is_coint: bool
    alpha: float
    beta: float
    resid_mu: float
    resid_std: float
    half_life: float
    adf_t: float
    ecm_coef: float


def engle_granger(
    y: np.ndarray,
    x: np.ndarray,
    *,
    crit: float = ADF_CRIT_5,
    require_ecm_sign: bool = True,
    max_half_life: float = 120.0,
) -> CointResult:
    """Engle-Granger 两步法 + ECM 符号检验 + 半衰期约束。

    Step 1：$\\ln y_t = \\alpha + \\beta \\ln x_t + e_t$，对 $\\hat e_t$ 做 ADF。
    Step 2：$\\Delta \\ln y_t = c + \\delta \\hat e_{t-1} + \\varepsilon_t$，要求 $\\delta < 0$
    （误差修正机制存在），否则判定为**伪协整**。

    `require_ecm_sign` 与 `max_half_life` 两项目前 EG 不做、但配对交易必须做的过滤：
    前者排除"残差平稳但无回归动力"，后者排除回归太慢（持有成本吃掉收益）的配对。
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(y) < 100 or len(y) != len(x):
        return CointResult(False, 0.0, 0.0, np.nan, np.nan, np.nan, 0.0, np.nan)

    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(b[0]), float(b[1])
    resid = y - X @ b
    mu, sd = float(np.mean(resid)), float(np.std(resid, ddof=1))

    t_adf = adf_tstat(resid)
    if t_adf >= crit:
        return CointResult(False, alpha, beta, mu, sd, np.nan, t_adf, np.nan)

    hl = ou_half_life(resid)
    if not np.isfinite(hl) or hl > max_half_life:
        return CointResult(False, alpha, beta, mu, sd, hl, t_adf, np.nan)

    ecm_coef = np.nan
    if require_ecm_sign:
        dy = np.diff(y)
        X2 = np.column_stack([np.ones(len(resid) - 1), resid[:-1]])
        b2, *_ = np.linalg.lstsq(X2, dy, rcond=None)
        ecm_coef = float(b2[1])
        if ecm_coef >= 0:
            return CointResult(False, alpha, beta, mu, sd, hl, t_adf, ecm_coef)

    return CointResult(True, alpha, beta, mu, sd, hl, t_adf, ecm_coef)


# ══════════════════════════════════════════════════════════════════
# 配对筛选
# ══════════════════════════════════════════════════════════════════
def build_candidate_pairs(
    close: pd.DataFrame,
    industry: pd.Series,
    *,
    min_corr: float = 0.60,
    min_overlap: int = 750,
    max_pairs_per_symbol: int = 6,
) -> List[Tuple[str, str]]:
    """构造候选配对：同行业 + 收益相关性门槛 + 每只股票限制配对数。

    同行业约束用证监会门类（18 类，较粗）；相关性筛选部分弥补了行业粒度不足
    ——同一门类下业务差异大的两只股票收益相关性通常也低。
    `max_pairs_per_symbol` 防止少数高流动性股票吸走全部配对（组合集中度控制）。
    """
    cols = [c for c in close.columns if isinstance(industry.get(c), str)]
    if len(cols) < 2:
        return []

    ret = close[cols].pct_change(fill_method=None)
    corr = ret.corr(min_periods=min_overlap)

    groups: Dict[str, List[str]] = {}
    for c in cols:
        groups.setdefault(industry[c], []).append(c)

    counts: Dict[str, int] = {c: 0 for c in cols}
    pairs: List[Tuple[str, str]] = []
    for _, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        sub = corr.loc[members, members]
        # 只用上三角，避免 (a,b) 与 (b,a) 重复
        iu = np.triu_indices(len(members), k=1)
        vals = sub.to_numpy()[iu]
        order = np.argsort(-vals)
        for idx in order:
            v = vals[idx]
            if not np.isfinite(v) or v < min_corr:
                break
            a = members[iu[0][idx]]
            b = members[iu[1][idx]]
            if counts[a] >= max_pairs_per_symbol or counts[b] >= max_pairs_per_symbol:
                continue
            pairs.append((a, b))
            counts[a] += 1
            counts[b] += 1
    return pairs


# ══════════════════════════════════════════════════════════════════
# 滚动协整状态
# ══════════════════════════════════════════════════════════════════
def rolling_cointegration(
    close: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    *,
    window: int = 250,
    step: int = 20,
    max_half_life: float = 120.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对每个配对在每个重估日做 EG 检验。

    Returns
    -------
    (ok, alpha, beta, resid_mu, resid_sd)
        均为 `index=重估日, columns=配对键 "a|b"` 的宽表。
        `resid_mu` / `resid_sd` 是**该次估计所用窗口内残差的均值与标准差**，
        回测时用它标准化当前价差 —— 若沿用建仓时点的 μ/σ，β 每次重估更新后
        价差水平会整体平移，旧 μ/σ 失效会导致 z 假突破（实测使协整失效平仓占比虚高至 46%）。
    """
    logp = np.log(close)
    keys = [f"{a}|{b}" for a, b in pairs]
    dates = logp.index
    eval_dates = dates[window::step]

    n_p, n_d = len(pairs), len(eval_dates)
    ok_mat = np.zeros((n_d, n_p), dtype=bool)
    alpha_mat = np.full((n_d, n_p), np.nan)
    beta_mat = np.full((n_d, n_p), np.nan)
    mu_mat = np.full((n_d, n_p), np.nan)
    sd_mat = np.full((n_d, n_p), np.nan)

    col_pos = {c: i for i, c in enumerate(logp.columns)}
    arr = logp.to_numpy()

    for di, dt in enumerate(eval_dates):
        i1 = dates.get_loc(dt)
        i0 = i1 - window
        if i0 < 0:
            continue
        chunk = arr[i0:i1]
        for pi, (a, b) in enumerate(pairs):
            ia, ib = col_pos[a], col_pos[b]
            ya = chunk[:, ia]
            yb = chunk[:, ib]
            m = np.isfinite(ya) & np.isfinite(yb)
            if m.sum() < window * 0.8:
                continue
            res = engle_granger(ya[m], yb[m], max_half_life=max_half_life)
            if res.is_coint:
                ok_mat[di, pi] = True
                alpha_mat[di, pi] = res.alpha
                beta_mat[di, pi] = res.beta
                mu_mat[di, pi] = res.resid_mu
                sd_mat[di, pi] = res.resid_std

    out = [pd.DataFrame(m, index=eval_dates, columns=keys)
           for m in (ok_mat, alpha_mat, beta_mat, mu_mat, sd_mat)]

    if verbose:
        per_date = out[0].sum(axis=1)
        print(f"[协整] {len(pairs)} 对 × {len(eval_dates)} 个重估日  "
              f"平均通过 {per_date.mean():.1f} 对/日  "
              f"（最高 {per_date.max()}，最低 {per_date.min()}）")

    return tuple(out)  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════
# 回测
# ══════════════════════════════════════════════════════════════════
def backtest_pairs(
    close: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    coint_ok: pd.DataFrame,
    beta_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    mu_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.0,
    max_hold: int = 20,
    round_trip_cost: float = 0.001,
    short_borrow_rate: float = 0.0,
    beta_drift_tol: float = 0.30,
    return_trades: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, List[dict]]:
    """配对组合回测（状态机）。

    每日对每个配对：
      1. 用最近一次有效协整估计的 $(\\alpha,\\beta,\\mu,\\sigma)$ 计算
         $z_t = (\\ln P^a_t - \\beta\\ln P^b_t - \\alpha - \\mu)/\\sigma$
      2. $z > z_{entry}$ → 空 a 多 b（pos = −1）；$z < -z_{entry}$ → 多 a 空 b（pos = +1）
      3. **持仓期间锁定建仓时的 $(\\alpha,\\beta,\\mu,\\sigma)$**，保证 z 的时间序列连续
         —— 若每次重估都换参数，价差水平会整体平移，z 出现假跳变
      4. 平仓条件：$z$ 穿过 $z_{exit}$、持有超 `max_hold`、协整状态失效、
         或 $\\beta$ 相对建仓时漂移超 `beta_drift_tol`（结构变化）
      5. 组合收益等权分配给所有活跃配对

    Costs
    -----
    `round_trip_cost` 是**单腿双边**成本。一个完整回合 = 开仓（两腿各一次单边）
    + 平仓（两腿各一次单边）= 2 × `round_trip_cost`。
    默认 0.001 对应铁律 0.1%（买 0.026% + 卖 0.076%）。

    `short_borrow_rate` > 0 时按日计提空腿融券成本（方案 A 主口径为 0）。
    """
    logp = np.log(close)
    ret = close.pct_change(fill_method=None)
    dates = close.index
    keys = [f"{a}|{b}" for a, b in pairs]

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(dates.union(df.index)).ffill().reindex(dates)

    ok_daily = _align(coint_ok).fillna(False)
    beta_daily = _align(beta_df)
    alpha_daily = _align(alpha_df)
    mu_daily = _align(mu_df)
    sd_daily = _align(sd_df)

    pos = {k: 0 for k in keys}
    p_beta = {k: np.nan for k in keys}
    p_alpha = {k: np.nan for k in keys}
    p_mu = {k: np.nan for k in keys}
    p_sd = {k: np.nan for k in keys}
    hold_days = {k: 0 for k in keys}

    daily_gross = np.zeros(len(dates))
    daily_cost = np.zeros(len(dates))
    daily_borrow = np.zeros(len(dates))
    n_active_series = np.zeros(len(dates))
    trades: List[dict] = []

    col = {c: i for i, c in enumerate(close.columns)}
    arr_l = logp.to_numpy()
    arr_r = ret.to_numpy()

    for ti in range(1, len(dates)):
        dt = dates[ti]
        active: List[Tuple[str, int]] = []

        for k in keys:
            a, b = k.split("|")
            ia, ib = col[a], col[b]
            if not (np.isfinite(arr_l[ti, ia]) and np.isfinite(arr_l[ti, ib])):
                continue

            # ── 平仓检查（用建仓时锁定的参数，保证 z 连续）──
            if pos[k] != 0:
                hold_days[k] += 1
                reason = None
                if not bool(ok_daily.at[dt, k]):
                    reason = "coint_break"
                elif hold_days[k] >= max_hold:
                    reason = "max_hold"
                else:
                    nb = beta_daily.at[dt, k]
                    if np.isfinite(nb) and p_beta[k] not in (0, np.nan) \
                            and abs(nb / p_beta[k] - 1.0) > beta_drift_tol:
                        reason = "beta_drift"
                    else:
                        spread = arr_l[ti, ia] - p_beta[k] * arr_l[ti, ib] - p_alpha[k]
                        z = (spread - p_mu[k]) / p_sd[k] if p_sd[k] > 0 else np.nan
                        if np.isfinite(z):
                            if pos[k] == 1 and z >= -z_exit:
                                reason = "z_exit"
                            elif pos[k] == -1 and z <= z_exit:
                                reason = "z_exit"
                if reason:
                    daily_cost[ti] += round_trip_cost
                    trades.append({"key": k, "exit": dt, "days": hold_days[k], "reason": reason})
                    pos[k] = 0
                    hold_days[k] = 0

            # ── 建仓检查 ──
            if pos[k] == 0 and bool(ok_daily.at[dt, k]):
                beta = beta_daily.at[dt, k]
                alpha = alpha_daily.at[dt, k]
                mu = mu_daily.at[dt, k]
                sd = sd_daily.at[dt, k]
                if not all(np.isfinite(v) for v in (beta, alpha, mu, sd)) or sd <= 0:
                    continue
                spread = arr_l[ti, ia] - beta * arr_l[ti, ib] - alpha
                z = (spread - mu) / sd
                if z > z_entry:
                    pos[k] = -1
                elif z < -z_entry:
                    pos[k] = 1
                else:
                    continue
                p_beta[k], p_alpha[k], p_mu[k], p_sd[k] = beta, alpha, mu, sd
                hold_days[k] = 0
                daily_cost[ti] += round_trip_cost
                trades.append({"key": k, "entry": dt, "beta": float(beta), "z": float(z)})

            if pos[k] != 0:
                active.append((k, pos[k]))

        if active:
            w = 1.0 / len(active)
            for k, p in active:
                a, b = k.split("|")
                leg = arr_r[ti, col[a]] - arr_r[ti, col[b]]
                if np.isfinite(leg):
                    daily_gross[ti] += w * p * leg
                    if short_borrow_rate > 0:
                        daily_borrow[ti] += w * short_borrow_rate / 252.0
            n_active_series[ti] = len(active)

    net = daily_gross - daily_cost - daily_borrow
    out = pd.DataFrame({
        "gross_ret": daily_gross,
        "cost": daily_cost,
        "borrow": daily_borrow,
        "port_ret": net,
        "n_active": n_active_series,
        "cum": (1.0 + net).cumprod(),
    }, index=dates)
    return (out, trades) if return_trades else out


def summarize_trades(trades: List[dict]) -> pd.DataFrame:
    """把交易流水配对成完整回合（entry → exit），统计持有天数与平仓原因。"""
    rows = []
    open_rec: Dict[str, dict] = {}
    for t in trades:
        k = t["key"]
        if "entry" in t:
            open_rec[k] = {"key": k, "entry": t["entry"], "beta": t.get("beta"), "z": t.get("z")}
        elif "exit" in t and k in open_rec:
            r = open_rec.pop(k)
            r.update({"exit": t["exit"], "days": t["days"], "reason": t["reason"]})
            rows.append(r)
    return pd.DataFrame(rows)
