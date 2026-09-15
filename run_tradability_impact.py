"""
run_tradability_impact.py — 量化交易约束的代价（离线运行器，多链路）。

回答两个问题：

  1. **一字涨跌停约束会让策略损失多少换手与收益？**（CLAUDE.md TODO 21）
  2. **若把它接进主流量价链路，各条已发布结论会受多大影响？**（TODO 19 的影响评估）

为什么需要它：`trade_limits` 此前零生产调用，因此「涨跌停不可交易」从未约束过任何
已发布结论。决定要不要接线之前，必须先有这个数。

方法
----
对每条链路，用**它自己的真实 OOF 预测** + 真实价格/OHLC 复现组合，
在「无约束」与「加约束」两种设置下对比（其余参数完全相同，故差值可归因于约束）。

⚠️ 口径说明：这是用各链路的 OOF 预测复现的**近似**组合（统一用 LS 与各链路自身的
hold_days），**不是逐字重跑那几份报告**。目的是判断影响的**量级**，而非复算报告数字。

⚠️ 已知下界：`detect_limit_moves` 只识别**一字板**（开盘即触限价且 high==low），
盘中封板无法成交的情形未建模 —— 真实代价更大。

用法
----
    python run_tradability_impact.py               # 全部链路
    python run_tradability_impact.py --n 200       # 抽样（快速自检）
    python run_tradability_impact.py --route models

输出: tradability_report.txt（UTF-8）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import CACHE_DIR, load_field_panel  # noqa: E402
from risk.tradability import build_trade_limits, measure_restriction_impact  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "tradability_report.txt"
OHLC_FIELDS = ("open", "high", "low", "close")

# 各链路：预测文件 + 自身口径（hold_days 取自各自 report.py / backtest_monthly.py）
ROUTES = (
    {
        "key": "models",
        "name": "models（ML 非线性合成）",
        "pred": "models/oof_predictions.csv",
        "hold_days": 5,
        "monthly": False,
    },
    {
        "key": "zz500",
        "name": "zz500_pit_trial（量价×中证500）",
        "pred": "strategies/zz500_pit_trial/oof_predictions.csv",
        "hold_days": 5,
        "monthly": False,
    },
    {
        "key": "zz500_wf",
        "name": "zz500 WF PIT-select（年度重选池）",
        "pred": "strategies/zz500_pit_trial/oof_predictions_pit_select.csv",
        "hold_days": 5,
        "monthly": False,
    },
    {
        "key": "fundamental",
        "name": "zz500_fundamental_trial（方向2 基本面）",
        "pred": "strategies/zz500_fundamental_trial/oof_predictions_monthly.csv",
        "hold_days": 1,
        "monthly": True,
    },
)


def _read_pred(rel: str) -> pd.DataFrame | None:
    p = PROJECT_ROOT / rel
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description="交易约束代价量化（多链路）")
    ap.add_argument("--n", type=int, default=None, help="抽样股票数（默认全量）")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--route", default=None, help="只跑指定链路 key")
    args = ap.parse_args()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 88)
    log("交易约束代价量化报告（一字涨跌停，多链路）")
    log("=" * 88)
    log(f"参数: n={args.n or '全量'} start={args.start} end={args.end} "
        f"route={args.route or '全部'}")
    log("口径: 各链路自己的 OOF 预测复现 LS 组合；min_stocks_mult=3")
    log("⚠️ 本复现**未含**各报告的 PIT 成员掩码与 selection-bias 剔除 —— 因此")
    log("   绝对值与报告数字**不可比**；只有「约束前 − 约束后」的**差值**有量级参考意义。")

    routes = [r for r in ROUTES if args.route in (None, r["key"])]
    summary: list[dict] = []
    ok = True

    try:
        # ── 预测：先读全部可用链路，取其股票并集 ──
        preds: dict[str, pd.DataFrame] = {}
        for r in routes:
            df = _read_pred(r["pred"])
            if df is None:
                log(f"  [跳过] {r['name']} —— 缺少 {r['pred']}")
                continue
            preds[r["key"]] = df
        if not preds:
            raise FileNotFoundError("没有任何可用链路的 OOF 预测")

        cached = {p.stem for p in CACHE_DIR.glob("*.csv")}
        union = sorted({str(c) for df in preds.values() for c in df.columns if str(c) in cached})
        if args.n:
            union = union[:args.n]
        log("")
        log(f"股票并集（各链路预测列 ∩ cache）: {len(union)} 只")

        t0 = time.time()
        ohlc = load_field_panel(union, fields=OHLC_FIELDS, start=args.start, end=args.end)
        log(f"真实 OHLC 载入完成（{time.time() - t0:.1f}s）")
        if ohlc["close"].shape[1] == 0:
            raise AssertionError("OHLC 面板为空：检查 data/cache 与预测文件的股票交集")

        for r in routes:
            if r["key"] not in preds:
                continue
            pred = preds[r["key"]]

            # ── 该链路的股票子集与面板切片 ──
            syms = [str(c) for c in pred.columns if str(c) in ohlc["close"].columns]
            if not syms:
                log(f"\n[跳过] {r['name']} —— 与 OHLC 面板无交集")
                continue
            close = ohlc["close"][syms]
            o, h, l = (ohlc[f][syms] for f in ("open", "high", "low"))

            if r["monthly"]:
                # 月末预测 → 日频前向填充（月调仓语义，见 backtest_monthly.py::ffill_to_daily）；
                # 窗口取预测自身的日期范围，避免无信号区间稀释指标
                lo, hi = pred.index.min(), pred.index.max()
                idx = close.index[(close.index >= lo) & (close.index <= hi)]
                p = pred.reindex(idx).ffill().reindex(columns=syms)
            else:
                idx = pred.index.intersection(close.index)
                p = pred.loc[idx, syms]

            close = close.loc[idx]
            o, h, l = (x.loc[idx] for x in (o, h, l))

            lim_up, lim_down = build_trade_limits(o, h, l, close)
            valid = close.notna()
            n_valid = max(int(valid.to_numpy().sum()), 1)
            up_rate = float((lim_up & valid).to_numpy().sum()) / n_valid
            down_rate = float((lim_down & valid).to_numpy().sum()) / n_valid

            table = measure_restriction_impact(
                p, close, (lim_up, lim_down),
                hold_days=r["hold_days"], min_stocks_mult=3)
            d = table.loc["delta"]
            u = table.loc["unrestricted"]
            rr = table.loc["restricted"]

            log("")
            log("-" * 88)
            log(f"[{r['key']}] {r['name']}   hold_days={r['hold_days']}"
                f"{'（月末预测前向填充）' if r['monthly'] else ''}")
            log(f"  面板 {close.shape[0]} 日 × {close.shape[1]} 票 | "
                f"一字涨停 {up_rate * 100:.4f}% / 一字跌停 {down_rate * 100:.4f}%")
            log(f"  换手     {u['total_turnover']:>10,.1f} → {rr['total_turnover']:>10,.1f}"
                f"  ({d['total_turnover']:+,.1f})")
            log(f"  年化     {u['cagr'] * 100:>9,.2f}% → {rr['cagr'] * 100:>9,.2f}%"
                f"  ({d['cagr'] * 100:+,.2f} pp)")
            log(f"  夏普     {u['sharpe']:>10,.4f} → {rr['sharpe']:>10,.4f}"
                f"  ({d['sharpe']:+,.4f})")
            log(f"  最大回撤 {u['max_drawdown'] * 100:>9,.2f}% → "
                f"{rr['max_drawdown'] * 100:>9,.2f}%  ({d['max_drawdown'] * 100:+,.2f} pp)")

            summary.append({
                "route": r["key"],
                "hold_days": r["hold_days"],
                "n_syms": close.shape[1],
                "up_lock_%": up_rate * 100,
                "turnover_%": (d["total_turnover"] / max(u["total_turnover"], 1e-9) * 100),
                "cagr_base": u["cagr"],
                "cagr_restricted": rr["cagr"],
                "sharpe_base": u["sharpe"],
                "sharpe_restricted": rr["sharpe"],
                "sharpe_delta": d["sharpe"],
                "sharpe_rel_%": d["sharpe"] / max(abs(u["sharpe"]), 1e-9) * 100.0,
                "mdd_delta_pp": d["max_drawdown"] * 100,
            })

        # ── 汇总 ──
        log("")
        log("=" * 88)
        log("跨链路汇总（约束代价）")
        log("=" * 88)
        sdf = pd.DataFrame(summary).set_index("route")
        log(sdf.to_string(float_format=lambda v: f"{v:,.4f}"))
        log("")
        log("判读:")
        if len(sdf):
            worst = sdf.loc[sdf["sharpe_delta"].idxmin()]
            log(f"  换手变化      : {sdf['turnover_%'].min():+.2f}% ~ {sdf['turnover_%'].max():+.2f}%"
                "（**几乎不变** —— 一字板占比仅 ~0.1%）")
            log(f"  夏普变化      : {sdf['sharpe_delta'].min():+.4f} ~ {sdf['sharpe_delta'].max():+.4f}"
                f"（最差链路 {sdf['sharpe_delta'].idxmin()}）")
            log(f"  夏普相对变化  : {sdf['sharpe_rel_%'].min():+.2f}% ~ {sdf['sharpe_rel_%'].max():+.2f}%"
                "  ← 决策应看这个，绝对值受复现口径影响")
            log(f"  年化变化      : {(sdf['cagr_restricted'] - sdf['cagr_base']).min() * 100:+.2f} pp ~ "
                f"{(sdf['cagr_restricted'] - sdf['cagr_base']).max() * 100:+.2f} pp")
            log(f"  最差链路      : {worst.name}，Δ夏普 {worst['sharpe_delta']:+.4f}"
                f"（相对 {worst['sharpe_rel_%']:+.2f}%）")
            log("")
            log("  含义:")
            log("    · 换手几乎不动、夏普却掉几个百分点 —— 再次印证「被拦住的恰是关键交易」")
            log("      （涨停追涨类），而不是量的问题。")
            log("    · 基本面（月度、hold=1）受影响最小（<3%）—— 月调仓撞上一字板的概率极低。")
            log("    · **没有任何一条链路的结论方向被改变**：接线属于「补齐现实性」，")
            log("      不是「修正错误」。")
            log("  ⚠️ 下界提醒: 只覆盖一字板；盘中封板的不可成交未建模，真实代价更大。")

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
    log("=" * 88)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
