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
from backtest.reconciliation import assert_overlay_closed, close_overlay_ledger  # noqa: E402
from data.fetcher import CACHE_DIR, load_field_panel  # noqa: E402
from risk.drawdown_control import (  # noqa: E402
    apply_drawdown_control,
    apply_drawdown_scaling,
    relevering_cost,
)
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
        log("固定（两态滞后带）vs 自适应（连续映射）—— 匹配平均仓位后的**前沿**对比")
        step_family = table.drop(index="baseline（无控制）")

        cont_rows = []
        for mca in (0.03, 0.05, 0.08, 0.12, 0.20, 0.30):
            for fl in (0.3, 0.4, 0.5, 0.7):
                ctl = apply_drawdown_scaling(base_ret, max_cut_at=mca, floor=fl)
                cont_rows.append({
                    "config": f"cont mca={mca:.2f} floor={fl:.1f}",
                    "cagr": annualize_cagr(ctl["controlled_ret"]),
                    "sharpe": sharpe_ratio(ctl["controlled_ret"]),
                    "max_drawdown": max_drawdown(ctl["controlled_ret"]),
                    "avg_scale": float(ctl["scale"].mean()),
                })
        cont = pd.DataFrame(cont_rows).set_index("config").sort_values(
            "sharpe", ascending=False)
        log("")
        log(cont.head(6).to_string(float_format=lambda v: f"{v:,.4f}"))

        # 在若干「平均仓位档」上各取两族最接近的方案，比较夏普 —— 比单点比较稳健
        cmp_rows = []
        for target in (0.55, 0.65, 0.75, 0.85):
            s_hit = step_family.iloc[(step_family["avg_scale"] - target).abs().argsort()[:1]]
            c_hit = cont.iloc[(cont["avg_scale"] - target).abs().argsort()[:1]]
            s_sr, c_sr = float(s_hit["sharpe"].iloc[0]), float(c_hit["sharpe"].iloc[0])
            cmp_rows.append({
                "target_avg": target,
                "step_cfg": str(s_hit.index[0]),
                "step_avg": float(s_hit["avg_scale"].iloc[0]),
                "step_sharpe": s_sr,
                "cont_cfg": str(c_hit.index[0]),
                "cont_avg": float(c_hit["avg_scale"].iloc[0]),
                "cont_sharpe": c_sr,
                "sharpe_diff": c_sr - s_sr,
            })
        cmp_df = pd.DataFrame(cmp_rows).set_index("target_avg")
        log("")
        log("匹配平均仓位后的前沿对比:")
        log(cmp_df.to_string(float_format=lambda v: f"{v:,.4f}"))

        n_win = int((cmp_df["sharpe_diff"] > 0).sum())
        avg_diff = float(cmp_df["sharpe_diff"].mean())
        log("")
        log(f"  连续在 {n_win}/{len(cmp_df)} 个仓位档上更优；平均夏普差 {avg_diff:+.4f}")
        if n_win >= 3:
            log("  判定: 匹配仓位下**连续映射整体占优** —— 响应形态本身带信息量，")
            log("        而非仅是「同一条前沿上的另一个点」。")
            log("        机理: 两态开关在浅回撤时仍满仓（dd 从 0 掉到 −threshold 之间不动作），")
            log("        连续映射则「早减、缓减」，能吃到回撤初段 —— 而回撤初段的减仓最有效。")
        else:
            log("  判定: 匹配仓位下两族无一致优劣 —— 差异主要来自降杠杆而非响应形态。")
        log("")
        log("  ⚠️ 两点提醒:")
        log("    · 两族的参数都是在**同一段样本内**挑的，夏普都含样本内选择效应；")
        log("      若要据此定稿，应做 walk-forward 或 out-of-sample 复核（铁律 5）。")
        log("    · 「连续」只说明**响应**连续；**参数本身仍是固定的**。若让阈值随波动率")
        log("      自适应（threshold_t = k·σ_t），会再引入 k 与 lookback 两个参数，")
        log("      在样本内更容易拟合出好看的曲线 —— 前述提醒会更强。")

        log("")
        log("=" * 84)
        log("边界扫描：连续族最优是否真在 mca≈0.02，还是网格没包住？")
        log("=" * 84)
        log("⚠️ 含「调杠杆成本」：受控收益是按 scale 缩放净收益，默认假设**调杠杆免费**。")
        log("   mca 越小 → scale 在 0.2↔1.0 间切换越频繁 → 该假设越不成立。")
        scan_rows = []
        for mca in (0.002, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05):
            for fl in (0.05, 0.1, 0.2, 0.3, 0.5):
                ctl = apply_drawdown_scaling(base_ret, max_cut_at=mca, floor=fl)
                rlc = relevering_cost(ctl["scale"], gross=2.0)
                adj = ctl["controlled_ret"] - rlc
                scan_rows.append({
                    "mca": mca,
                    "floor": fl,
                    "config": f"mca={mca:.3f} fl={fl:.2f}",
                    "sharpe": sharpe_ratio(ctl["controlled_ret"]),
                    "sharpe_adj": sharpe_ratio(adj),
                    "cagr": annualize_cagr(ctl["controlled_ret"]),
                    "cagr_adj": annualize_cagr(adj),
                    "avg_scale": float(ctl["scale"].mean()),
                    "sum_dscale": float(ctl["scale"].diff().abs().sum()),
                    "relever_cost": float(rlc.sum()),
                })
        scan = pd.DataFrame(scan_rows).set_index("config")

        # 每个 mca 下取最优 floor，看 mca 的边际趋势
        marg = scan.loc[scan.groupby("mca")["sharpe"].idxmax()]
        log("")
        log("按 mca 分组取最优 floor（观察边际趋势）:")
        log(marg[["mca", "floor", "sharpe", "sharpe_adj", "cagr", "cagr_adj",
                  "avg_scale", "sum_dscale", "relever_cost"]].to_string(
            float_format=lambda v: f"{v:,.4f}"))

        mca_min = float(scan["mca"].min())
        best_raw = scan.loc[scan["sharpe"].idxmax()]
        best_adj = scan.loc[scan["sharpe_adj"].idxmax()]
        log("")
        log(f"  未计调杠杆成本  最优: {best_raw.name}  夏普 {best_raw['sharpe']:.4f}"
            f"  (mca={best_raw['mca']:.3f})")
        log(f"  计入调杠杆成本  最优: {best_adj.name}  夏普 {best_adj['sharpe_adj']:.4f}"
            f"  (mca={best_adj['mca']:.3f})")
        log(f"  原始序列（无控制）夏普: {sharpe_ratio(base_ret):.4f}")
        log("")
        if float(best_raw["mca"]) <= mca_min * 1.01:
            log("  ① 未计成本时最优仍贴在 mca 下界 → **网格仍未包住**。")
        else:
            log(f"  ① 未计成本时最优在 mca={best_raw['mca']:.3f}（**内部点**）"
                " → 先前「贴边界」是网格粒度问题，非退化。")
        if float(best_adj["mca"]) > float(best_raw["mca"]) * 1.05:
            log(f"  ② 计入调杠杆成本后最优**移向更大的 mca"
                f"（{best_raw['mca']:.3f} → {best_adj['mca']:.3f}）**"
                " → 极端区确实在吃「调杠杆免费」这个假设。")
        else:
            log("  ② 计入成本后最优位置基本不动 → 结论不依赖该假设。")
        log(f"  ③ mca 越小，Σ|Δscale| 从 {marg['sum_dscale'].min():.1f} 升到 "
            f"{marg['sum_dscale'].max():.1f}（{marg['sum_dscale'].max() / max(marg['sum_dscale'].min(), 1e-9):.0f}×）；"
            "调杠杆成本随之放大。")

        # ── 清算演示：受控账本目前只记了「仓位账」 ──
        log("")
        log("清算演示（覆盖层必须三层账本才闭合）:")
        probe = apply_drawdown_scaling(base_ret, max_cut_at=0.02, floor=0.20)
        ledger = close_overlay_ledger(base_ret, probe["scale"], gross=2.0)
        try:
            assert_overlay_closed(
                pd.DataFrame({"port_ret": probe["controlled_ret"]}),
                base_ret, probe["scale"], gross=2.0)
            log("  ? 仓位账直接作净收益竟然通过了 —— 说明费用账为 0（不应发生）")
        except AssertionError as exc:
            log("  ✗ 把 controlled_ret（= 仓位账）直接当净收益 → 清算**不闭合**：")
            log(f"     {str(exc).splitlines()[0]}")
        assert_overlay_closed(
            pd.DataFrame({"port_ret": ledger["net_ret"]}),
            base_ret, probe["scale"], gross=2.0)
        log("  ✓ 补上费用账与现金账（`close_overlay_ledger`）→ 闭合")
        log(f"     费用账（调杠杆成本）合计 {float(ledger['relever_cost'].sum()):.6f}"
            f" | 平均现金占比 {float(ledger['cash_weight'].mean()):.4f}")

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
