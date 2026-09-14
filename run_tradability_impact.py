"""
run_tradability_impact.py — 量化交易约束的代价（离线运行器）。

回答一个问题（CLAUDE.md TODO 21）：

> **一字涨跌停约束会让策略损失多少换手与收益？**

为什么需要它：`trade_limits` 此前零生产调用，因此「涨跌停不可交易」从未约束过任何
已发布结论。容量与可交易性研究必须先有这个数，再决定要不要接线（TODO 19）。

数据来源
--------
    pred  ← models/oof_predictions.csv    真实 OOF 预测（790 票，gitignore 生成物）
    OHLC  ← data/cache/*.csv              真实开高低收（用于识别一字板）

用法
----
    python run_tradability_impact.py            # 全量 OOF universe
    python run_tradability_impact.py --n 200    # 抽样（快速自检）

输出: tradability_report.txt（UTF-8，同时打印到终端）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import CACHE_DIR, load_daily  # noqa: E402
from risk.tradability import build_trade_limits, measure_restriction_impact  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "tradability_report.txt"
OHLC_FIELDS = ("open", "high", "low", "close")


def load_real_ohlc(symbols, start: str = "2010-01-01", end: str = "2025-12-31"):
    """从本地缓存读真实 OHLC 面板（四字段各一个矩阵）。"""
    frames: dict[str, dict[str, pd.Series]] = {k: {} for k in OHLC_FIELDS}
    for i, s in enumerate(symbols):
        df = load_daily(str(s))
        if df is None or not set(OHLC_FIELDS).issubset(df.columns):
            continue
        sub = df.loc[(df.index >= start) & (df.index <= end)]
        if len(sub) < 100:
            continue
        for k in OHLC_FIELDS:
            frames[k][str(s)] = sub[k]
        if (i + 1) % 200 == 0:
            print(f"  ...已加载 {i + 1}/{len(symbols)} 只")
    return {k: pd.DataFrame(v).sort_index() for k, v in frames.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="交易约束代价量化")
    ap.add_argument("--n", type=int, default=None, help="抽样股票数（默认全量）")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--hold-days", type=int, default=5)
    args = ap.parse_args()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(str(msg))

    log("=" * 72)
    log("交易约束代价量化报告（一字涨跌停）")
    log("=" * 72)
    log(f"参数: n={args.n or '全量'} start={args.start} end={args.end} "
        f"hold_days={args.hold_days}")

    ok = True
    try:
        pred_path = PROJECT_ROOT / "models" / "oof_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"缺少 OOF 预测 {pred_path}（由 models/lgbm_trainer.py 生成）")
        pred = pd.read_csv(pred_path, index_col=0)
        pred.index = pd.to_datetime(pred.index)
        pred = pred.sort_index()

        cached = {p.stem for p in CACHE_DIR.glob("*.csv")}
        syms = [c for c in pred.columns if str(c) in cached]
        if args.n:
            syms = syms[:args.n]

        t0 = time.time()
        ohlc = load_real_ohlc(syms, args.start, args.end)
        log(f"真实 OHLC 载入完成（{time.time() - t0:.1f}s）")

        close = ohlc["close"]
        common_idx = pred.index.intersection(close.index)
        common_cols = pred.columns.intersection(close.columns)
        pred_a = pred.loc[common_idx, common_cols]
        o = ohlc["open"].loc[common_idx, common_cols]
        h = ohlc["high"].loc[common_idx, common_cols]
        l = ohlc["low"].loc[common_idx, common_cols]
        c = close.loc[common_idx, common_cols]
        log(f"对齐后面板: {c.shape[0]} 日 × {c.shape[1]} 票")
        if c.shape[1] == 0:
            raise AssertionError("面板为空：检查 data/cache 与 oof_predictions 的股票交集")

        lim_up, lim_down = build_trade_limits(o, h, l, c)
        valid = c.notna()
        up_rate = float((lim_up & valid).to_numpy().sum()) / max(int(valid.to_numpy().sum()), 1)
        down_rate = float((lim_down & valid).to_numpy().sum()) / max(int(valid.to_numpy().sum()), 1)
        log("")
        log("一字板发生率（占有效格数）:")
        log(f"  一字涨停锁: {up_rate * 100:.4f}%   一字跌停锁: {down_rate * 100:.4f}%")

        table = measure_restriction_impact(
            pred_a, c, (lim_up, lim_down), hold_days=args.hold_days)
        log("")
        log("对账对比（除 trade_limits 外参数完全相同）:")
        log(table.to_string(float_format=lambda v: f"{v:,.4f}"))

        u = table.loc["unrestricted"]
        r = table.loc["restricted"]
        d = table.loc["delta"]
        log("")
        log("约束的代价（restricted − unrestricted）:")
        log(f"  换手合计  {u['total_turnover']:>12,.2f} → {r['total_turnover']:>12,.2f}"
            f"   ({d['total_turnover']:+,.2f}, "
            f"{d['total_turnover'] / max(u['total_turnover'], 1e-9) * 100:+.2f}%)")
        log(f"  成交日数  {u['n_trade_days']:>12,.0f} → {r['n_trade_days']:>12,.0f}"
            f"   ({d['n_trade_days']:+,.0f} 天)")
        log(f"  累计收益  {u['total_return'] * 100:>11,.2f}% → {r['total_return'] * 100:>11,.2f}%"
            f"   ({d['total_return'] * 100:+,.2f} 个百分点)")
        log(f"  年化 CAGR {u['cagr'] * 100:>11,.2f}% → {r['cagr'] * 100:>11,.2f}%"
            f"   ({d['cagr'] * 100:+,.2f} 个百分点)")
        log(f"  夏普      {u['sharpe']:>12,.4f} → {r['sharpe']:>12,.4f}"
            f"   ({d['sharpe']:+,.4f})")
        log(f"  最大回撤  {u['max_drawdown'] * 100:>11,.2f}% → {r['max_drawdown'] * 100:>11,.2f}%"
            f"   ({d['max_drawdown'] * 100:+,.2f} 个百分点)")

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
    log("=" * 72)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
