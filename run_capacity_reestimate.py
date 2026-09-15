"""
run_capacity_reestimate.py — 容量上限重估（trade_limits + 滑点纳入基准）。

回答（CLAUDE.md TODO 24）：

> **P1 容量上限「< 0.5 亿」是在「无涨跌停约束、无滑点」的基准上算的。
>   把这两者纳入后，容量上限会往哪移？**

方法
----
对 P1 多头（LO）做 AUM 网格冲击成本检验（sqrt 法则），三种基准各跑一遍：

    baseline      无约束、无滑点（历史口径）
    +trade_limits 加一字涨跌停约束
    +limits+slip  约束 + 单边万5 滑点

三种都用 `build_weight_portfolio` 同一路径，delta 干净。

⚠️ 依赖 zz500 PIT 面板与去 bias 预测；数据未缓存时本脚本会明确失败并提示。

用法
----
    python run_capacity_reestimate.py

输出: capacity_reestimate_report.txt（UTF-8）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import load_field_panel  # noqa: E402
from risk.cost_model import SLIPPAGE  # noqa: E402
from risk.tradability import build_trade_limits  # noqa: E402
from strategies.feature_selection.build_pit_matrix import load_pit_panel  # noqa: E402
from strategies.zz500_pit_trial.capacity import run_capacity_sweep  # noqa: E402
from strategies.zz500_pit_trial.neutralize import load_amount_matrix  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "capacity_reestimate_report.txt"
AUM_GRID = [0.5e8, 2e8, 5e8, 20e8, 50e8]   # 0.5亿 ~ 50亿
SIGNIFICANCE_BAR = 0.5


def _ceiling(cap_df: pd.DataFrame, bar: float) -> float | None:
    """夏普跌破 `bar` 的最大 AUM（亿）；全网格未跌破则 None。"""
    below = cap_df.loc[cap_df["sharpe"] <= bar]
    if below.empty:
        return None
    return float(below["aum_yi"].min())


def main() -> int:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 90)
    log("容量上限重估（P1 多头 LO，sqrt 冲击法则）")
    log("=" * 90)
    log("三种基准: baseline / +trade_limits / +limits+slippage（单边万5）")

    ok = True
    try:
        t0 = time.time()
        close, volume, member = load_pit_panel("zz500")
        log(f"PIT 面板: {close.shape[0]} 日 × {close.shape[1]} 票（{time.time() - t0:.1f}s）")

        from strategies.zz500_pit_trial.enhance import load_pred_pit_select
        pred = load_pred_pit_select()
        amount = load_amount_matrix(close)

        idx = pred.index.intersection(close.index)
        cols = pred.columns.intersection(close.columns)
        pred = pred.loc[idx, cols]
        close = close.loc[idx, cols]
        amount = amount.loc[idx, cols]
        log(f"对齐后: {close.shape[0]} 日 × {close.shape[1]} 票")

        # OHLC → 涨跌停掩码
        t1 = time.time()
        ohlc = load_field_panel(list(cols), fields=("open", "high", "low", "close"))
        o, h, l, c = (ohlc[f].loc[idx, cols] for f in ("open", "high", "low", "close"))
        lim_up, lim_down = build_trade_limits(o, h, l, c)
        log(f"涨跌停掩码: 涨停 {float((lim_up & c.notna()).to_numpy().sum()) / max(int(c.notna().to_numpy().sum()), 1):.4%}"
            f" / 跌停 {float((lim_down & c.notna()).to_numpy().sum()) / max(int(c.notna().to_numpy().sum()), 1):.4%}"
            f"（{time.time() - t1:.1f}s）")

        for k in (0.3, 1.0):
            log("")
            log("-" * 90)
            log(f"冲击系数 k={k}（{'温和' if k < 0.5 else '激进'}）")
            base, _ = run_capacity_sweep(pred, close, amount, AUM_GRID, k=k)
            lim, _ = run_capacity_sweep(pred, close, amount, AUM_GRID, k=k,
                                        trade_limits=(lim_up, lim_down))
            lim_slip, _ = run_capacity_sweep(pred, close, amount, AUM_GRID, k=k,
                                             trade_limits=(lim_up, lim_down),
                                             slippage=SLIPPAGE)

            tbl = pd.DataFrame({
                "AUM(亿)": base["aum_yi"],
                "夏普(base)": base["sharpe"],
                "夏普(+限)": lim["sharpe"],
                "夏普(+限+滑)": lim_slip["sharpe"],
                "年化(base)": base["annual"],
                "年化(+限+滑)": lim_slip["annual"],
                "冲击bps(base)": base["avg_impact_bps"],
            })
            log(tbl.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

            cb = _ceiling(base, SIGNIFICANCE_BAR)
            cl = _ceiling(lim, SIGNIFICANCE_BAR)
            cs = _ceiling(lim_slip, SIGNIFICANCE_BAR)

            def fmt(x):
                return f"{x:.2f} 亿" if x is not None else f"> {AUM_GRID[-1] / 1e8:.0f} 亿（未跌破）"

            log(f"  容量上限（夏普跌破 {SIGNIFICANCE_BAR}）:")
            log(f"    baseline      → {fmt(cb)}")
            log(f"    +trade_limits → {fmt(cl)}")
            log(f"    +限+滑点      → {fmt(cs)}")

    except AssertionError as e:
        ok = False
        log("")
        log(f"❌ 失败: {e}")
    except FileNotFoundError as e:
        ok = False
        log("")
        log(f"❌ 数据缺失: {e}")

    log("")
    log(f"结论: {'✅ 完成' if ok else '❌ 未完成，见上'}")
    log("=" * 90)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
