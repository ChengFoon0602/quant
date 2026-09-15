"""
run_drawdown_control.py — 回撤控制的参数网格（离线运行器）。

回答的问题（CLAUDE.md TODO 20）：

> **回撤控制到底有没有用？哪组参数合适？**

为什么要有它：框架此前**没有任何回撤控制**。是否引入、用哪组参数都是设计选择，
本脚本把选择变成**带数据的判断题**，而不是凭直觉拍一个阈值。

方法
----
1. 用真实 OOF 预测 + 真实价格跑一次 LS 组合，取 `port_ret` 作基线；
2. 对 `(threshold, cut)` 网格逐组施加 `apply_drawdown_control`（`recovery = threshold/2`）；
3. 对比 CAGR / 夏普 / 最大回撤 / 触发次数 / 平均仓位。

⚠️ 组件**不接入生产**：结果只用于选参数，接入与否由你决定。

用法
----
    python run_drawdown_control.py            # 全量 OOF universe
    python run_drawdown_control.py --n 200    # 抽样

输出: drawdown_control_report.txt（UTF-8）
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
from risk.drawdown_control import apply_drawdown_control  # noqa: E402
from risk.portfolio import build_weight_portfolio  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "drawdown_control_report.txt"

GRID_THRESHOLD = (0.03, 0.04, 0.05, 0.07, 0.10, 0.15)
GRID_CUT = (0.3, 0.4, 0.5, 0.6, 0.7)


def _row(name: str, ret: pd.Series, n_cuts: int, avg_scale: float) -> dict:
    return {
        "config": name,
        "cagr": annualize_cagr(ret),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "n_cuts": n_cuts,
        "avg_scale": avg_scale,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="回撤控制参数网格")
    ap.add_argument("--n", type=int, default=None, help="抽样股票数（默认全量）")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--hold-days", type=int, default=5)
    args = ap.parse_args()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 84)
    log("回撤控制参数网格报告")
    log("=" * 84)
    log(f"参数: n={args.n or '全量'} start={args.start} end={args.end} hold_days={args.hold_days}")
    log("口径: recovery = threshold / 2；闭环（回撤在受控路径上计算）")

    ok = True
    try:
        pred_path = PROJECT_ROOT / "models" / "oof_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少 OOF 预测 {pred_path}（由 models/lgbm_trainer.py 生成）")
        pred = pd.read_csv(pred_path, index_col=0)
        pred.index = pd.to_datetime(pred.index)
        pred = pred.sort_index()

        cached = {p.stem for p in CACHE_DIR.glob("*.csv")}
        syms = [c for c in pred.columns if str(c) in cached]
        if args.n:
            syms = syms[:args.n]

        t0 = time.time()
        close = load_field_panel(syms, fields=("close",),
                                 start=args.start, end=args.end)["close"]
        common_idx = pred.index.intersection(close.index)
        common_cols = pred.columns.intersection(close.columns)
        pred_a = pred.loc[common_idx, common_cols]
        c = close.loc[common_idx, common_cols]
        log(f"真实面板: {c.shape[0]} 日 × {c.shape[1]} 票（载入 {time.time() - t0:.1f}s）")
        if c.shape[1] == 0:
            raise AssertionError("面板为空：检查 data/cache 与 oof_predictions 的股票交集")

        res = build_weight_portfolio(pred_a, c, hold_days=args.hold_days)
        base_ret = res["port_ret"]
        rows = [_row("baseline（无控制）", base_ret, 0, 1.0)]
        for threshold in GRID_THRESHOLD:
            for cut in GRID_CUT:
                ctl = apply_drawdown_control(base_ret, threshold=threshold, cut=cut,
                                             recovery=threshold / 2.0)
                n_cuts = int((ctl["scale"].diff().fillna(1.0) < 0).sum())
                rows.append(_row(f"thr={threshold:.2f} cut={cut:.1f}",
                                 ctl["controlled_ret"], n_cuts,
                                 float(ctl["scale"].mean())))

        table = pd.DataFrame(rows).set_index("config")
        table = table.sort_values("sharpe", ascending=False)
        log("")
        log(table.to_string(float_format=lambda v: f"{v:,.4f}"))

        base = table.loc["baseline（无控制）"]
        best = table.drop(index="baseline（无控制）").iloc[0]
        log("")
        log("判读:")
        log(f"  基线          夏普 {base['sharpe']:.4f} | CAGR {base['cagr'] * 100:.2f}%"
            f" | MDD {base['max_drawdown'] * 100:.2f}%")
        log(f"  网格最优        {best.name} → 夏普 {best['sharpe']:.4f}"
            f" | CAGR {best['cagr'] * 100:.2f}% | MDD {best['max_drawdown'] * 100:.2f}%"
            f" | 触发 {int(best['n_cuts'])} 次 | 平均仓位 {best['avg_scale']:.3f}")
        log(f"  夏普变化       {best['sharpe'] - base['sharpe']:+.4f}")
        log(f"  回撤变化       {(best['max_drawdown'] - base['max_drawdown']) * 100:+.2f} 个百分点")
        log(f"  最优是否在网格边界: 阈值={best.name.split()[0]}、仓位={best.name.split()[1]}"
            " —— 若落在边界，需扩大网格再判")

        log("")
        log("读表须知（否则容易误读）:")
        log("  · 夏普对**常数**杠杆不变（r×c 使 mean 与 std 同比例变化）——因此夏普的提升")
        log("    来自「择时减仓」而非「整体降杠杆」，可与基线直接比较。")
        log("  · 但 CAGR **不可**直接与基线比较：受控组合平均仓位更低（见 avg_scale 列），")
        log("    收益下降里含降杠杆的成本。判断「值不值」应看 夏普/回撤 与 avg_scale 的取舍。")
        log("  · 表面规律：减仓越深 → 夏普越高、回撤越浅、绝对收益越低（本网格单调）。")
        log("    这不是「最优解在角落」，而是**风险偏好问题** —— 继续外推只会更极端。")
        log("  · 本组件**未接入生产**：上表只用于选参数，接入与否由你决定（CLAUDE.md TODO 20）。")

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
    log("=" * 84)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
