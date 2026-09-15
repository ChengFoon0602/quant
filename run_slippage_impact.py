"""
run_slippage_impact.py — 滑点对跨链路结论的影响量化（离线运行器）。

回答一个问题（CLAUDE.md TODO 23）：

> **滑点（单边万 5）至今未进主口径 —— 加进去，各链路夏普会掉多少？**

为什么需要它
------------
`risk/cost_model.py::SLIPPAGE = 0.0005` 已定义、`CostModel.with_slippage()` 已实现，
但**零生产调用**：`build_weight_portfolio` / `build_portfolio` 的组合级成本只有
BUY_COST / SELL_COST（注释写明「另计」）。因此**所有已发布夏普都不含滑点**。
量级：单边万 5 → 双边 0.1%，与佣金+印花税的双边 0.102% **近乎等量**。

本脚本把 `slippage` 参数接入后（默认 0.0，不改历史数字），对比「无滑点 vs 单边万 5」，
供决策「是否把滑点纳入铁律 3 主口径」。

用法
----
    python run_slippage_impact.py               # 全部链路
    python run_slippage_impact.py --route models

输出: slippage_report.txt（UTF-8）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.metrics import annualize_cagr, max_drawdown, sharpe_ratio  # noqa: E402
from data.fetcher import CACHE_DIR, load_field_panel  # noqa: E402
from risk.cost_model import SLIPPAGE  # noqa: E402
from risk.portfolio import build_weight_portfolio  # noqa: E402

try:
    from models.portfolio_backtest import build_portfolio as _models_build_portfolio
except Exception:  # noqa: BLE001
    _models_build_portfolio = None

REPORT_PATH = PROJECT_ROOT / "slippage_report.txt"

ROUTES = (
    {"key": "models", "name": "models（ML 非线性合成）",
     "pred": "models/oof_predictions.csv", "hold_days": 5, "monthly": False},
    {"key": "zz500", "name": "zz500_pit_trial（量价×中证500）",
     "pred": "strategies/zz500_pit_trial/oof_predictions.csv", "hold_days": 5, "monthly": False},
    {"key": "zz500_wf", "name": "zz500 WF PIT-select",
     "pred": "strategies/zz500_pit_trial/oof_predictions_pit_select.csv",
     "hold_days": 5, "monthly": False},
    {"key": "fundamental", "name": "zz500_fundamental_trial（月度）",
     "pred": "strategies/zz500_fundamental_trial/oof_predictions_monthly.csv",
     "hold_days": 1, "monthly": True},
)


def _read_pred(rel: str) -> pd.DataFrame | None:
    p = PROJECT_ROOT / rel
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _row(name: str, ret: pd.Series) -> dict:
    return {
        "config": name,
        "cagr": annualize_cagr(ret),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1.0 + ret).prod() - 1.0) if (1.0 + ret).prod() > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="滑点影响量化")
    ap.add_argument("--n", type=int, default=None, help="抽样股票数（默认全量）")
    ap.add_argument("--route", default=None, help="只跑指定链路 key")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 88)
    log("滑点影响量化（单边万 5 = 0.0005，按总换手计提）")
    log("=" * 88)
    log(f"参数: n={args.n or '全量'} route={args.route or '全部'} "
        f"start={args.start} end={args.end}")
    log("口径: 同一组合，仅切换 slippage=0.0 vs 0.0005；min_stocks_mult=3")
    log("⚠️ 复现未含各报告 PIT 成员掩码与 bias 剔除 → 绝对值与报告不可比，只看差值。")

    routes = [r for r in ROUTES if args.route in (None, r["key"])]
    preds: dict[str, pd.DataFrame] = {}
    for r in routes:
        df = _read_pred(r["pred"])
        if df is None:
            log(f"  [跳过] {r['name']} —— 缺少 {r['pred']}")
            continue
        preds[r["key"]] = df
    if not preds:
        log("❌ 没有任何可用链路")
        return 1

    cached = {p.stem for p in CACHE_DIR.glob("*.csv")}
    union = sorted({str(c) for df in preds.values() for c in df.columns if str(c) in cached})
    if args.n:
        union = union[:args.n]

    t0 = time.time()
    close = load_field_panel(union, fields=("close",), start=args.start, end=args.end)["close"]
    log(f"\n真实价格载入完成（{time.time() - t0:.1f}s），股票并集 {close.shape[1]} 只")

    summary: list[dict] = []
    for r in routes:
        if r["key"] not in preds:
            continue
        pred = preds[r["key"]]
        syms = [str(c) for c in pred.columns if str(c) in close.columns]
        if not syms:
            continue
        c = close[syms]
        if r["monthly"]:
            lo, hi = pred.index.min(), pred.index.max()
            idx = c.index[(c.index >= lo) & (c.index <= hi)]
            p = pred.reindex(idx).ffill().reindex(columns=syms)
        else:
            idx = pred.index.intersection(c.index)
            p = pred.loc[idx, syms]
        c = c.loc[idx]

        base = build_weight_portfolio(p, c, hold_days=r["hold_days"], min_stocks_mult=3)
        slip = build_weight_portfolio(p, c, hold_days=r["hold_days"], min_stocks_mult=3,
                                      slippage=SLIPPAGE)

        b, s = _row("无滑点", base["port_ret"]), _row("万5滑点", slip["port_ret"])
        log("")
        log("-" * 88)
        log(f"[{r['key']}] {r['name']}   hold_days={r['hold_days']}"
            f"{'（月末前向填充）' if r['monthly'] else ''}")
        log(f"  CAGR     {b['cagr'] * 100:>9,.2f}% → {s['cagr'] * 100:>9,.2f}%"
            f"  ({(s['cagr'] - b['cagr']) * 100:+,.2f} pp)")
        log(f"  夏普     {b['sharpe']:>9,.4f} → {s['sharpe']:>9,.4f}"
            f"  ({(s['sharpe'] - b['sharpe']):+,.4f}, "
            f"相对 {(s['sharpe'] - b['sharpe']) / max(abs(b['sharpe']), 1e-9) * 100:+,.2f}%)")
        log(f"  最大回撤 {b['max_drawdown'] * 100:>8,.2f}% → {s['max_drawdown'] * 100:>8,.2f}%")
        summary.append({
            "route": r["key"],
            "hold_days": r["hold_days"],
            "sharpe_base": b["sharpe"],
            "sharpe_slip": s["sharpe"],
            "sharpe_delta": s["sharpe"] - b["sharpe"],
            "sharpe_rel_%": (s["sharpe"] - b["sharpe"]) / max(abs(b["sharpe"]), 1e-9) * 100,
            "cagr_delta_pp": (s["cagr"] - b["cagr"]) * 100,
            "turnover_sum": float(base["turnover"].sum()),
        })

    log("")
    log("=" * 88)
    log("跨链路汇总")
    log("=" * 88)
    sdf = pd.DataFrame(summary).set_index("route")
    log(sdf.to_string(float_format=lambda v: f"{v:,.4f}"))
    log("")
    log("判读:")
    if len(sdf):
        worst = sdf.loc[sdf["sharpe_delta"].idxmin()]
        log(f"  夏普变化      : {sdf['sharpe_delta'].min():+.4f} ~ {sdf['sharpe_delta'].max():+.4f}"
            f"（最差 {sdf['sharpe_delta'].idxmin()}）")
        log(f"  夏普相对变化  : {sdf['sharpe_rel_%'].min():+.2f}% ~ {sdf['sharpe_rel_%'].max():+.2f}%")
        log(f"  年化变化      : {sdf['cagr_delta_pp'].min():+.2f} ~ "
            f"{sdf['cagr_delta_pp'].max():+.2f} pp")
        log("")
        log("  滑点按**总换手**计提，故换手越高的链路受影响越大。")
        log("  对比铁律成本（双边 0.102%）：滑点（双边 0.1%）与其**近乎等量** ——")
        log("  若纳入，等价于「摩擦成本翻倍」。")
        log("  ⚠️ 是否改铁律 3 由你决定；本脚本只量化，不替你做决定。")

    log("")
    log("结论: ✅ 完成")
    log("=" * 88)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
