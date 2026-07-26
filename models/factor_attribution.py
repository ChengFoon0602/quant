"""
models/factor_attribution.py — 组合收益因子归因。

目标: 对 LightGBM 组合和 alpha001 组合做 CAPM / 风格因子归因，剥离市场 beta，
      估计 pure alpha 及其显著性。

数据限制:
  - 没有沪深300指数、市值、账面价值等标准 FF3 数据。
  - 用个股等权组合作为市场因子 MKT 代理。
  - 无风险利率按 3% 年化近似（日利率 ≈ 3%/252）。
  - 风格因子用已有 alpha 因子的主成分 / 分组收益作为代理，不作为标准 SMB/HML。

用法:
    cd D:/桌面文件/quant
    python models/factor_attribution.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import cache_summary, load_daily
from models.portfolio_backtest import build_portfolio

# ── 配置 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RF_ANNUAL = 0.03  # 无风险利率 3% 年化
RF_DAILY = RF_ANNUAL / 252


def load_market_and_portfolios():
    """加载市场等权收益和各组合收益。"""
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

    # 市场等权 overlapped 5 日收益（与组合构建一致）
    daily_ret = close_matrix.shift(-2) / close_matrix.shift(-1) - 1
    market_signal = daily_ret.mean(axis=1)
    market_ret = market_signal.rolling(5).mean().dropna()

    # 读取组合收益
    lgb_ls = pd.read_csv(MODEL_DIR / "portfolio_backtest_summary.csv")  # placeholder
    # 实际组合收益需要重新构建，这里读取 CSV 不够，直接调用 portfolio_backtest 的逻辑
    from models.portfolio_backtest import load_data as pb_load_data
    pred_lgb, alpha001, close_matrix, _, _, _ = pb_load_data()

    lgb_ls_df = build_portfolio(pred_lgb, close_matrix, long_only=False, cost=0.003, hold_days=5)
    lgb_lo_df = build_portfolio(pred_lgb, close_matrix, long_only=True, cost=0.003, hold_days=5)
    a001_ls_df = build_portfolio(alpha001, close_matrix, long_only=False, cost=0.003, hold_days=5)

    portfolios = {
        "LightGBM LS": lgb_ls_df["port_ret"],
        "LightGBM Long-only": lgb_lo_df["port_ret"],
        "alpha001 LS": a001_ls_df["port_ret"],
    }
    return portfolios, market_ret


def _ols_stats(y: np.ndarray, X: np.ndarray) -> dict:
    """最小二乘回归：y = X @ beta，返回 beta, t, p, r2。"""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    y_pred = X @ beta
    resid = y - y_pred
    ssr = (resid ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ssr / sst if sst > 0 else 0.0
    sigma2 = ssr / max(n - k, 1)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat = beta / (se + 1e-12)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=max(n - k, 1)))
    return {
        "beta": beta,
        "t_stat": t_stat,
        "p_value": p_value,
        "r2": r2,
        "nobs": n,
    }


def capm_attribution(port_ret: pd.Series, market_ret: pd.Series, name: str) -> dict:
    """CAPM 归因：R_p - R_f = alpha + beta * (R_m - R_f) + eps。"""
    common_dates = port_ret.index.intersection(market_ret.index)
    p = (port_ret.loc[common_dates] - RF_DAILY).dropna()
    m = (market_ret.loc[common_dates] - RF_DAILY).dropna()

    aligned = pd.concat([p, m], axis=1).dropna()
    y = aligned.iloc[:, 0].values
    mk = aligned.iloc[:, 1].values
    X = np.column_stack([np.ones(len(y)), mk])

    res = _ols_stats(y, X)
    alpha_daily = res["beta"][0]
    beta = res["beta"][1]
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    alpha_t = res["t_stat"][0]
    beta_t = res["t_stat"][1]
    alpha_p = res["p_value"][0]
    r2 = res["r2"]

    # 累计 pure alpha
    p_aligned = aligned.iloc[:, 0]
    m_aligned = aligned.iloc[:, 1]
    alpha_cum = (1 + (p_aligned - beta * m_aligned).fillna(0)).cumprod()

    return {
        "name": name,
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "beta": beta,
        "beta_t": beta_t,
        "r2": r2,
        "n_obs": res["nobs"],
        "alpha_cum": alpha_cum,
        "excess_ret": p_aligned,
        "market_excess": m_aligned,
    }


def rolling_capm(port_ret: pd.Series, market_ret: pd.Series, window: int = 252) -> pd.DataFrame:
    """滚动 CAPM alpha / beta。"""
    common_dates = port_ret.index.intersection(market_ret.index)
    p = (port_ret.loc[common_dates] - RF_DAILY).dropna()
    m = (market_ret.loc[common_dates] - RF_DAILY).dropna()

    aligned = pd.concat([p, m], axis=1).dropna()
    y_all = aligned.iloc[:, 0].values
    m_all = aligned.iloc[:, 1].values

    alphas = []
    betas = []
    dates = []
    for i in range(window, len(y_all)):
        sub_y = y_all[i - window:i]
        sub_m = m_all[i - window:i]
        X = np.column_stack([np.ones(len(sub_y)), sub_m])
        try:
            res = _ols_stats(sub_y, X)
            alphas.append(res["beta"][0])
            betas.append(res["beta"][1])
            dates.append(aligned.index[i])
        except Exception:
            continue
    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=dates)


def plot_attribution(results: dict):
    """绘制归因结果。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计 pure alpha
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res["alpha_cum"].index, res["alpha_cum"].values, label=name, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("累计 Pure Alpha（剥离市场 Beta 后）")
    ax.set_ylabel("累计净值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Beta 暴露
    ax = axes[0, 1]
    names = list(results.keys())
    betas = [results[n]["beta"] for n in names]
    colors = ["green", "orange", "blue"]
    bars = ax.bar(names, betas, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="beta=1")
    ax.set_title("市场 Beta 暴露")
    ax.set_ylabel("Beta")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.legend()

    # 3. 滚动 alpha
    ax = axes[1, 0]
    for name, res in results.items():
        port_ret = res["excess_ret"] + RF_DAILY
        mkt = res["market_excess"] + RF_DAILY
        rolling = rolling_capm(port_ret, mkt, window=252)
        if len(rolling) > 0:
            ax.plot(rolling.index, rolling["alpha"] * 252, label=name, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("滚动年化 Alpha（252 日窗口）")
    ax.set_ylabel("年化 Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 滚动 beta
    ax = axes[1, 1]
    for name, res in results.items():
        port_ret = res["excess_ret"] + RF_DAILY
        mkt = res["market_excess"] + RF_DAILY
        rolling = rolling_capm(port_ret, mkt, window=252)
        if len(rolling) > 0:
            ax.plot(rolling.index, rolling["beta"], label=name, linewidth=1.0)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("滚动 Beta（252 日窗口）")
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "factor_attribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {fig_path}")


def main():
    print("=" * 70)
    print("因子归因 — CAPM 剥离市场 Beta")
    print("=" * 70)

    portfolios, market_ret = load_market_and_portfolios()

    results = {}
    rows = []
    for name, port_ret in portfolios.items():
        res = capm_attribution(port_ret, market_ret, name)
        results[name] = res
        rows.append({
            "组合": name,
            "Alpha(年化)": f"{res['alpha_annual']:.2%}",
            "Alpha t": f"{res['alpha_t']:.2f}",
            "Alpha p": f"{res['alpha_p']:.4f}",
            "Beta": f"{res['beta']:.3f}",
            "Beta t": f"{res['beta_t']:.2f}",
            "R²": f"{res['r2']:.3f}",
            "样本数": res["n_obs"],
        })
        print(f"\n{name}:")
        print(f"  Alpha(年化) = {res['alpha_annual']:+.2%}  (t={res['alpha_t']:.2f}, p={res['alpha_p']:.4f})")
        print(f"  Beta        = {res['beta']:.3f}  (t={res['beta_t']:.2f})")
        print(f"  R²          = {res['r2']:.3f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("CAPM 归因汇总")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv(MODEL_DIR / "factor_attribution.csv", index=False)
    print(f"\n结果保存: models/factor_attribution.csv")

    plot_attribution(results)


if __name__ == "__main__":
    main()
