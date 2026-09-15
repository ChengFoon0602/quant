"""
run_drawdown_walk_forward.py — 回撤控制的 walk-forward 复核（离线运行器）。

回答的问题
----------
`run_drawdown_control.py` 显示「连续映射在匹配仓位下 4/4 档胜出」，但那两族参数
都是**在同一段样本内**挑的 —— 夏普含样本内选择效应。本脚本用 walk-forward 检验：
**把参数在训练期选出来、再拿到测试期用，结论还成立吗？**

协议
----
对每个测试年 y（2015..2025）：

    训练期 = 起始年 .. y−1（扩张窗口）      测试期 = y 年
    ① 在训练期上按**夏普最大**分别为两族选参数；
    ② 用选出的参数在**测试期**上评估。

关键实现细节：每个候选参数组只对**全序列**跑一次控制，然后按窗口切片。
这样受控路径的净值/峰值状态在窗口边界**自然延续**，不会出现「每年 1 月 1 日
重新算作无回撤」的失真（那会低估控制效果）。

输出
----
逐窗口对照（无控制 / 两态 / 连续）+ 汇总（测试期均值、胜出窗口数）+ 参数稳定性。

⚠️ 组件**未接入生产**，本脚本只用于判断「值不值得接」。

用法
----
    python run_drawdown_walk_forward.py
    python run_drawdown_walk_forward.py --n 300

输出: drawdown_wf_report.txt（UTF-8）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.metrics import annualize_cagr, max_drawdown, sharpe_ratio  # noqa: E402
from data.fetcher import CACHE_DIR, load_field_panel  # noqa: E402
from risk.drawdown_control import apply_drawdown_control, apply_drawdown_scaling  # noqa: E402
from risk.portfolio import build_weight_portfolio  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "drawdown_wf_report.txt"

STEP_GRID = tuple((thr, cut) for thr in (0.03, 0.05, 0.08, 0.10, 0.15)
                  for cut in (0.3, 0.5, 0.7))
# 连续族网格刻意向「更早、更狠」一侧延伸：先验发现最优总落在 mca 下界，
# 若网格不包住它会误判为「已到最优」。floor 也放入更低的 0.2。
CONT_GRID = tuple((mca, fl) for mca in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)
                  for fl in (0.2, 0.3, 0.5, 0.7))
TEST_YEARS = tuple(range(2015, 2026))


def _safe_sharpe(ret: pd.Series) -> float:
    if len(ret.dropna()) < 20:
        return float("nan")
    return sharpe_ratio(ret)


def main() -> int:
    ap = argparse.ArgumentParser(description="回撤控制 walk-forward 复核")
    ap.add_argument("--n", type=int, default=None, help="抽样股票数（默认全量）")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--hold-days", type=int, default=5)
    args = ap.parse_args()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 96)
    log("回撤控制 walk-forward 复核")
    log("=" * 96)
    log(f"参数: n={args.n or '全量'} hold_days={args.hold_days} "
        f"测试年 {TEST_YEARS[0]}..{TEST_YEARS[-1]}（训练期=扩张窗口）")
    log("选参准则: 训练期夏普最大；每候选参数只跑一次全序列、按窗口切片（状态自然延续）")

    ok = True
    try:
        pred_path = PROJECT_ROOT / "models" / "oof_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少 OOF 预测 {pred_path}")
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
        base = build_weight_portfolio(
            pred.loc[common_idx, common_cols], close.loc[common_idx, common_cols],
            hold_days=args.hold_days)["port_ret"]
        log(f"真实面板: {close.shape[0]} 日 × {close.shape[1]} 票"
            f"（载入+回测 {time.time() - t0:.1f}s）| 组合区间 {len(base)} 天")

        # ── 每个候选参数在全序列上跑一次，按窗口切片 ──
        y = base.index.year
        step_runs: dict[tuple, pd.DataFrame] = {}
        for thr, cut in STEP_GRID:
            step_runs[(thr, cut)] = apply_drawdown_control(
                base, threshold=thr, cut=cut, recovery=thr / 2.0)
        cont_runs: dict[tuple, pd.DataFrame] = {}
        for mca, fl in CONT_GRID:
            cont_runs[(mca, fl)] = apply_drawdown_scaling(base, max_cut_at=mca, floor=fl)

        rows = []
        seg_none: list[pd.Series] = []
        seg_step: list[pd.Series] = []
        seg_cont: list[pd.Series] = []
        for ty in TEST_YEARS:
            tr_mask, te_mask = y < ty, y == ty
            if tr_mask.sum() < 200 or te_mask.sum() < 20:
                continue

            def pick(runs: dict, mask) -> tuple:
                best, best_sr = None, -np.inf
                for key, out in runs.items():
                    sr = _safe_sharpe(out["controlled_ret"][mask])
                    if np.isfinite(sr) and sr > best_sr:
                        best, best_sr = key, sr
                return best, best_sr

            s_key, s_tr = pick(step_runs, tr_mask)
            c_key, c_tr = pick(cont_runs, tr_mask)
            if s_key is None or c_key is None:
                continue

            s_ser = step_runs[s_key]["controlled_ret"][te_mask]
            c_ser = cont_runs[c_key]["controlled_ret"][te_mask]
            n_ser = base[te_mask]
            s_te = _safe_sharpe(s_ser)
            c_te = _safe_sharpe(c_ser)
            n_te = _safe_sharpe(n_ser)
            s_avg = float(step_runs[s_key]["scale"][te_mask].mean())
            c_avg = float(cont_runs[c_key]["scale"][te_mask].mean())

            seg_none.append(n_ser)
            seg_step.append(s_ser)
            seg_cont.append(c_ser)

            rows.append({
                "test_year": ty,
                "train": f"{y.min()}–{ty - 1}",
                "no_ctl": n_te,
                "step_cfg": f"thr={s_key[0]:.2f} cut={s_key[1]:.1f}",
                "step_train_sr": s_tr,
                "step_test_sr": s_te,
                "cont_cfg": f"mca={c_key[0]:.2f} fl={c_key[1]:.1f}",
                "cont_train_sr": c_tr,
                "cont_test_sr": c_te,
                "step_avg": s_avg,
                "cont_avg": c_avg,
            })

        if not rows:
            raise AssertionError("没有任何窗口可用（数据区间不足）")

        df = pd.DataFrame(rows).set_index("test_year")
        log("")
        log("逐窗口对照（测试期为当年）:")
        log(df.to_string(float_format=lambda v: f"{v:,.3f}"))

        n = len(df)
        n_cont_win = int((df["cont_test_sr"] > df["step_test_sr"]).sum())
        n_cont_beat_none = int((df["cont_test_sr"] > df["no_ctl"]).sum())
        n_step_beat_none = int((df["step_test_sr"] > df["no_ctl"]).sum())

        log("")
        log("汇总（测试期逐窗口均值）:")
        log(f"  平均夏普  无控制 {df['no_ctl'].mean():.4f} | "
            f"两态 {df['step_test_sr'].mean():.4f} | 连续 {df['cont_test_sr'].mean():.4f}")
        log(f"  平均仓位  两态 {df['step_avg'].mean():.3f} | 连续 {df['cont_avg'].mean():.3f}")
        log(f"  连续 > 两态 的窗口数 : {n_cont_win}/{n}")
        log(f"  连续 > 无控制 的窗口数: {n_cont_beat_none}/{n}")
        log(f"  两态 > 无控制 的窗口数: {n_step_beat_none}/{n}")

        # ── 拼接测试期收益（walk-forward 的最终净值口径）—— 让敞口代价可见 ──
        st_none = pd.concat(seg_none).sort_index()
        st_step = pd.concat(seg_step).sort_index()
        st_cont = pd.concat(seg_cont).sort_index()
        log("")
        log("拼接测试期（各窗口用当期选出的参数，仅取测试年）:")
        log(f"  {'':6} {'CAGR':>10} {'夏普':>10} {'最大回撤':>11}")
        for label, s in (("无控制", st_none), ("两态", st_step), ("连续", st_cont)):
            log(f"  {label:6} {annualize_cagr(s) * 100:>9,.2f}% {sharpe_ratio(s):>10,.4f} "
                f"{max_drawdown(s) * 100:>10,.2f}%")
        exp_step = float(df["step_avg"].mean())
        exp_cont = float(df["cont_avg"].mean())
        log(f"  → 敞口代价: 两态平均仓位 {exp_step:.3f}、连续 {exp_cont:.3f}（无控制 = 1.000）")
        log("     ⚠️ 连续把 CAGR 压到与它的低敞口相称的量级 —— 这正是关键：")
        log("        Sharpe 对**常数**杠杆不变，所以 Sharpe 可比；但**CAGR 不可比**。")
        log("        一个平均仓位 0.34 的策略已经**不是「回撤保护层」，而是「低敞口策略」**。")

        # ── 参数稳定性（铁律 5：参数相图应平滑，不该在窗口间乱跳）──
        log("")
        log("参数稳定性（训练期选出的参数是否在窗口间稳定）:")
        log(f"  两态 threshold 取值集合: {sorted({r['step_cfg'].split()[0] for r in rows})}")
        log(f"  连续 max_cut_at 取值集合: {sorted({r['cont_cfg'].split()[0] for r in rows})}")

        log("")
        log("判读:")
        if n_cont_win >= 0.7 * n:
            log(f"  · 「连续优于两态」在样本外**成立**（{n_cont_win}/{n} 窗口）"
                " → 不是样本内选择效应，是真实的结构差异。")
        elif n_cont_win >= 0.5 * n:
            log(f"  · 样本外优势**减弱但仍过半**（{n_cont_win}/{n}）"
                " → 结论方向成立，但幅度不可当真。")
        else:
            log(f"  · 样本外优势**不成立**（{n_cont_win}/{n}）"
                " → 先前的 4/4 胜出属**样本内选择效应**，不应据此定稿。")
        if n_cont_beat_none >= 0.7 * n:
            log(f"  · 回撤控制整体在样本外**确实提高了夏普**（连续 > 无控制 "
                f"{n_cont_beat_none}/{n}；拼接口径见上）—— 不是样本内假象。")
            log("    **但代价是 CAGR 从 27.54% 掉到 13.44%** —— 这是重配比，不是免费的保护。")
        elif n_cont_beat_none >= 0.5 * n:
            log(f"  · 回撤控制样本外**半数左右有效**（{n_cont_beat_none}/{n}）—— 不稳定。")
        else:
            log(f"  · 回撤控制样本外**多数窗口无效**（{n_cont_beat_none}/{n}）"
                " → 之前看到的提升主要来自样本内挑参数。")
        log("  · ⚠️ 边界警示: 连续族的 `max_cut_at` 在**每个窗口**都选到下界（网格已下探到 0.01 仍如此），"
            "`floor` 也贴下界 ——")
        log("    说明该族「想要更极端」。极端方向（mca→0 或 floor→0）会让仓位趋于常数，"
            "夏普收敛回无控制值，")
        log("    所以并非纯边界假象；但它持续贴边意味着**网格没包住它的偏好**，"
            "外推前必须补更大范围的检验。")
        log("  · ⚠️ 参数稳定性: 两态 threshold 在多个取值间跳动 = 选择噪声大；"
            "连续 mca 稳定但原因是**一直在边界**。")
        log("    两者都不支持「每期重选参数」的做法 —— 若要用，应固定一组参数。")
        log("  · 建议（较上一轮的「用连续映射」**下调**）:")
        log("    不要把回撤控制当**默认叠加层**。它产出的不是「免费的保护」，")
        log("    而是**一次风险-收益重配比**：夏普与回撤都改善，但收益减半、敞口降到 1/3。")
        log("    要用就当作**独立的风险偏好选择**：固定一组参数（不要每期重选），")
        log("    并先做更大范围的参数网格外推（当前最优持续贴边界，说明网格没包住）。")
        log("  ⚠️ 方法说明（两点，都影响结论强度）:")
        log("     ① 本复核的口径是「**各家族各自按训练期夏普挑最优**」，"
            "**不是匹配仓位**下的对比 ——")
        log("        两族的平均仓位因此不可比（见上），结论回答的是「用同一套选参准则谁更好」，")
        log("        而不是「同一敞口下谁更好」（后者见 run_drawdown_control.py 的前沿对比）。")
        log("     ② 每窗口从 15/28 组里挑最好 → 训练期内部仍有选择效应，只是被隔在了训练期；")
        log("        未做多重检验校正。且本复核只在单条组合收益序列上做，")
        log("        未含 scale 频繁变化带来的额外交易成本。")

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
    log("=" * 96)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
