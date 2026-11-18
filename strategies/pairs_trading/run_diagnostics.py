"""
strategies/pairs_trading/run_diagnostics.py — 三项前置诊断。

1. 协整关系跨期存活率（训练期通过 → 测试期是否仍通过）
2. 价差事件研究（建仓后 z 是否回归）
3. 合成数据验证（证明实现本身正确）

产出：`figures/03_coint_survival.png`、`figures/04_event_study.png`、`figures/05_synthetic_check.png`
执行：`python strategies/pairs_trading/run_diagnostics.py`（约 2 分钟）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.fetcher import load_daily
from risk.cost_model import ROUND_TRIP
from risk.portfolio import calculate_metrics
from strategies.pairs_trading.pairs import (
    backtest_pairs,
    build_candidate_pairs,
    engle_granger,
    rolling_cointegration,
    summarize_trades,
)
from viz.plotting import plot_stratified_returns

FIG = Path(__file__).parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_sample(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    mem = pd.read_csv(ROOT / "data/cache_meta/zz500_membership.csv")
    syms = [str(c) for c in mem.columns if c != "Unnamed: 0"][:n]
    ind = pd.read_csv(ROOT / "data/cache_meta/industry.csv",
                      dtype={"symbol": str}).set_index("symbol")["industry"]
    px = {}
    for s in syms:
        d = load_daily(s, adjust="1")
        if d is not None and len(d) > 1500:
            px[s] = d["close"]
    return pd.DataFrame(px).sort_index(), ind


# ══════════════════════════════════════════════════════════════════
def diag_survival(close: pd.DataFrame, ind: pd.Series, n_sample: int = 300) -> pd.DataFrame:
    """协整关系的跨期存活率。"""
    print("\n" + "─" * 78)
    print("诊断 1 · 协整关系跨期存活率")
    print("─" * 78)
    rng = np.random.default_rng(7)
    groups: dict[str, list[str]] = {}
    for s in close.columns:
        g = ind.get(s)
        if isinstance(g, str):
            groups.setdefault(g, []).append(s)

    pairs = []
    for g, ss in groups.items():
        if len(ss) < 2:
            continue
        for _ in range(min(40, len(ss) // 2)):
            a, b = rng.choice(ss, 2, replace=False)
            pairs.append((a, b))

    logp = np.log(close)
    rows = []
    windows = [(("2018-01-01", "2020-12-31"), ("2021-01-01", "2022-12-31")),
               (("2020-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31"))]
    for tr, te in windows:
        ok_tr = survived = n_test = 0
        for a, b in pairs:
            s1 = logp[[a, b]].loc[tr[0]:tr[1]].dropna()
            if len(s1) < 400:
                continue
            if not engle_granger(s1[a].to_numpy(), s1[b].to_numpy()).is_coint:
                continue
            ok_tr += 1
            s2 = logp[[a, b]].loc[te[0]:te[1]].dropna()
            if len(s2) < 300:
                continue
            n_test += 1
            if engle_granger(s2[a].to_numpy(), s2[b].to_numpy()).is_coint:
                survived += 1
        rate = ok_tr / len(pairs) if pairs else 0.0
        surv = survived / n_test if n_test else 0.0
        rows.append({"训练期": f"{tr[0][:7]}~{tr[1][:7]}", "测试期": f"{te[0][:7]}~{te[1][:7]}",
                     "通过率": rate, "存活率": surv, "n_pass": ok_tr, "n_survive": survived})
        print(f"  {tr[0][:7]}~{tr[1][:7]} → 通过 {ok_tr}/{len(pairs)} = {rate:.1%}"
              f" | 测试期仍通过 {survived}/{n_test} = {surv:.1%}")

    df = pd.DataFrame(rows)
    plot_stratified_returns(
        {"2018-20 → 21-22": df.loc[0, "存活率"] * 100,
         "2020-22 → 23-24": df.loc[1, "存活率"] * 100},
        title="协整关系跨期存活率（训练期通过 → 测试期仍通过）",
        xlabel="训练期 → 测试期", ylabel="存活率 (%)",
        save_path=FIG / "03_coint_survival.png",
    )
    return df


def diag_event_study(close: pd.DataFrame, ind: pd.Series) -> pd.DataFrame:
    """建仓事件后价差的 z 路径。"""
    print("\n" + "─" * 78)
    print("诊断 2 · 价差事件研究（建仓后是否回归）")
    print("─" * 78)
    pairs = build_candidate_pairs(close, ind, min_corr=0.60, max_pairs_per_symbol=6)
    ok, alpha, beta, mu, sd = rolling_cointegration(close, pairs, verbose=False)
    _, trades = backtest_pairs(close, pairs, ok, beta, alpha, mu, sd,
                               z_entry=2.0, max_hold=40,
                               round_trip_cost=ROUND_TRIP, return_trades=True)

    logp = np.log(close)
    def _align(df):
        return df.reindex(close.index.union(df.index)).ffill().reindex(close.index)
    b_d, a_d, m_d, s_d = (_align(beta), _align(alpha), _align(mu), _align(sd))

    rows = []
    for e in [t for t in trades if "entry" in t]:
        k = e["key"]
        a, b = k.split("|")
        dt = e["entry"]
        if dt not in close.index:
            continue
        i = close.index.get_loc(dt)
        bt, al, m0, s0 = b_d.at[dt, k], a_d.at[dt, k], m_d.at[dt, k], s_d.at[dt, k]
        if not all(np.isfinite(v) for v in (bt, al, m0, s0)) or s0 <= 0:
            continue
        z0 = (logp[a].iloc[i] - bt * logp[b].iloc[i] - al - m0) / s0
        rec = {"z0": z0}
        for h in (1, 3, 5, 10, 20):
            if i + h < len(close):
                rec[f"dz{h}"] = (logp[a].iloc[i + h] - bt * logp[b].iloc[i + h] - al - m0) / s0 - z0
        rows.append(rec)

    ev = pd.DataFrame(rows)
    print(f"  建仓事件 {len(ev)} 次   建仓时 |z| 均值 {ev['z0'].abs().mean():.2f}")
    means, regress = {}, {}
    for h in (1, 3, 5, 10, 20):
        c = f"dz{h}"
        if c not in ev:
            continue
        x = ev[c].dropna()
        means[f"{h}日"] = float(x.mean())
        regress[f"{h}日"] = float((np.sign(x) != np.sign(ev.loc[x.index, "z0"])).mean()) * 100
        print(f"  h={h:2d}: Δz 均值 {x.mean():+.3f}   回归比例 {regress[f'{h}日']:.1f}%")

    plot_stratified_returns(
        regress,
        title="建仓后价差回归比例（50% = 抛硬币，>60% 才算有回归倾向）",
        xlabel="持有天数", ylabel="回归比例 (%)",
        save_path=FIG / "04_event_study.png",
    )
    return ev


def diag_synthetic() -> None:
    """合成数据：y = βx + AR(1) 噪声，验证实现能捕捉已知协整。"""
    print("\n" + "─" * 78)
    print("诊断 3 · 合成数据验证（工具正确性）")
    print("─" * 78)
    rng = np.random.default_rng(42)
    T, N = 2000, 40
    dates = pd.bdate_range("2010-01-01", periods=T)
    x = np.cumsum(rng.normal(0, 0.015, (T, 1)), axis=0)
    data = {}
    for i in range(N):
        beta_true = rng.uniform(0.8, 1.2)
        noise = np.zeros(T)
        phi = 0.93
        for t in range(1, T):
            noise[t] = phi * noise[t - 1] + rng.normal(0, 0.02)
        data[f"S{i:02d}"] = np.exp(beta_true * x[:, 0] + noise + np.log(10))
    close = pd.DataFrame(data, index=dates)
    ind = pd.Series({c: "X" for c in close.columns})

    pairs = build_candidate_pairs(close, ind, min_corr=0.3, max_pairs_per_symbol=4)
    ok, alpha, beta, mu, sd = rolling_cointegration(close, pairs, verbose=False)
    res, trades = backtest_pairs(close, pairs, ok, beta, alpha, mu, sd,
                                 z_entry=2.0, max_hold=20,
                                 round_trip_cost=ROUND_TRIP, return_trades=True)
    td = summarize_trades(trades)
    g, m = calculate_metrics(res["gross_ret"]), calculate_metrics(res["port_ret"])
    print(f"  {len(pairs)} 对   回合 {len(td)}   平均持有 {td['days'].mean():.1f} 天")
    print(f"  毛年化 {g['annual_return']:+.2%}   净年化 {m['annual_return']:+.2%}")
    verdict = "实现正确（能捕捉已知协整）" if g["annual_return"] > 0.05 else "⚠ 实现可疑，需排查"
    print(f"  判决：{verdict}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(["合成数据\n(已知协整)", "A股 zz500\n(实际)"],
           [g["annual_return"] * 100, -36.53],
           color=["#55A868", "#C44E52"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("毛年化收益 (%)")
    ax.set_title("工具正确性检验：合成数据 vs A 股实际")
    for i, v in enumerate([g["annual_return"] * 100, -36.53]):
        ax.text(i, v + (2 if v > 0 else -4), f"{v:+.1f}%", ha="center", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "05_synthetic_check.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("  配对交易前置诊断")
    print("=" * 78)
    close, ind = load_sample(300)
    print(f"[数据] {close.shape[1]} 只  {close.index.min().date()}~{close.index.max().date()}"
          f"  ({time.time()-t0:.0f}s)")

    diag_survival(close, ind)
    diag_event_study(close, ind)
    diag_synthetic()
    print(f"\n全部诊断完成，耗时 {time.time()-t0:.0f}s   图表在 {FIG}")


if __name__ == "__main__":
    main()
