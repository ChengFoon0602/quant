"""
run_reconciliation.py — 对真实报告数据跑账本自洽性对账（离线运行器）。

分工（与 `tests/test_reconciliation.py`）
----------------------------------------
    tests/test_reconciliation.py   合成数据：快（~0.3s）、可复现，随测试套件每次运行
    run_reconciliation.py          真实数据：慢（约 10–30s）、离线运行，按需执行

数据来源
--------
    pred  ← models/oof_predictions.csv    真实 OOF 预测（790 票，由 lgbm_trainer.py 生成）
    close ← data/cache/*.csv              真实价格

    ⚠️ 刻意**不依赖** `models/portfolio_backtest.py::load_data()` —— 它引用的
       `strategies/feature_selection/X_matrix.csv` 是被 `.gitignore`「特征矩阵」段
       有意排除的大文件生成物，当前工作副本中不存在。

附带输出
--------
    `min_stocks_mult` 2 vs 3 在真实 universe 上的影响量化（持仓日 / 换手 / 夏普），
    供「是否统一开仓门槛」的决策使用。

用法
----
    python run_reconciliation.py              # 全量
    python run_reconciliation.py --n 100      # 抽样 100 票（快速自检）
    python run_reconciliation.py --start 2015-01-01

输出: reconciliation_report.txt（UTF-8，同时打印到终端）
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

from backtest.reconciliation import assert_books_equal, assert_closed  # noqa: E402
from data.fetcher import CACHE_DIR, load_daily  # noqa: E402
from risk.portfolio import build_weight_portfolio  # noqa: E402

try:
    from models.portfolio_backtest import build_portfolio as _models_build_portfolio
except Exception as _e:  # noqa: BLE001
    _models_build_portfolio = None
    _MODELS_IMPORT_ERR = _e

REPORT_PATH = PROJECT_ROOT / "reconciliation_report.txt"
TRADING_DAYS = 252


def load_real_panel(n_syms: int | None = None,
                    start: str = "2010-01-01",
                    end: str = "2025-12-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """真实价格面板 + 真实 OOF 预测（对齐到公共日期与列）。"""
    pred_path = PROJECT_ROOT / "models" / "oof_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"缺少 OOF 预测 {pred_path}（由 models/lgbm_trainer.py 生成，属 gitignore 生成物）")

    pred = pd.read_csv(pred_path, index_col=0)
    pred.index = pd.to_datetime(pred.index)
    pred = pred.sort_index()

    cached = {p.stem for p in CACHE_DIR.glob("*.csv")}
    syms = [c for c in pred.columns if str(c) in cached]
    if n_syms:
        syms = syms[:n_syms]

    data: dict[str, pd.Series] = {}
    for i, s in enumerate(syms):
        df = load_daily(str(s))
        if df is None or "close" not in df.columns:
            continue
        s_close = df.loc[(df.index >= start) & (df.index <= end), "close"]
        if len(s_close) >= 100:
            data[str(s)] = s_close
        if (i + 1) % 200 == 0:
            print(f"  ...已加载 {i + 1}/{len(syms)} 只")

    close = pd.DataFrame(data).sort_index()
    common_idx = pred.index.intersection(close.index)
    common_cols = pred.columns.intersection(close.columns)
    return pred.loc[common_idx, common_cols], close.loc[common_idx, common_cols]


def _sharpe(ret: pd.Series) -> float:
    s = ret.dropna()
    if len(s) < 2 or s.std(ddof=1) < 1e-12:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS))


def main() -> int:
    ap = argparse.ArgumentParser(description="真实数据账本对账")
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
    log("真实数据账本自洽性对账报告")
    log("=" * 72)
    log(f"参数: n={args.n or '全量'} start={args.start} end={args.end} hold_days={args.hold_days}")

    ok = True
    try:
        t0 = time.time()
        pred, close = load_real_panel(args.n, args.start, args.end)
        log(f"真实面板: pred {pred.shape} / close {close.shape}  （载入 {time.time() - t0:.1f}s）")
        if close.shape[1] == 0:
            raise AssertionError("真实面板为空：检查 data/cache 与 oof_predictions 的股票交集")

        # ── ① risk 实现对账 ──
        hd = args.hold_days
        res_r, W_r = build_weight_portfolio(pred, close, hold_days=hd, return_weights=True)
        gaps = assert_closed(res_r, W_r, close)
        log("")
        log("① risk/portfolio.py::build_weight_portfolio")
        log(f"   对账通过；区间 {len(res_r)} 天，换手合计 {float(res_r['turnover'].sum()):.1f}")
        for k, v in gaps.items():
            log(f"     {k:<14} 最大绝对偏差 {v:.3e}")

        # ── ② models 实现对账 ──
        res_m = None
        W_m = None
        if _models_build_portfolio is not None:
            res_m, W_m = _models_build_portfolio(pred, close, hold_days=hd, return_weights=True)
            gaps_m = assert_closed(res_m, W_m, close)
            log("")
            log("② models/portfolio_backtest.py::build_portfolio")
            log(f"   对账通过；区间 {len(res_m)} 天，换手合计 {float(res_m['turnover'].sum()):.1f}")
            for k, v in gaps_m.items():
                log(f"     {k:<14} 最大绝对偏差 {v:.3e}")
        else:
            log("")
            log(f"② models 实现不可导入，跳过：{_MODELS_IMPORT_ERR}")

        # ── ③ 跨实现语义一致 ──
        log("")
        log("③ 跨实现逐位比对")
        if res_m is not None:
            try:
                assert_books_equal(res_r, res_m)
                log("   默认门槛下：逐位一致 ✓")
            except AssertionError as e:
                log(f"   默认门槛下不一致（预期：min_stocks_mult 2 vs 3）")
                log(f"     {e}")
                res_r3, _ = build_weight_portfolio(
                    pred, close, hold_days=hd, min_stocks_mult=3, return_weights=True)
                assert_books_equal(res_r3, res_m)
                log("   对齐 min_stocks_mult=3 后：逐位一致 ✓（证明差异只来自该参数）")
        else:
            log("   已跳过（models 实现不可用）")

        # ── ④ min_stocks 影响量化 ──
        log("")
        log("④ min_stocks_mult 2 vs 3 影响量化（真实 universe）")
        res_2, _ = build_weight_portfolio(
            pred, close, hold_days=hd, min_stocks_mult=2, return_weights=True)
        res_3, _ = build_weight_portfolio(
            pred, close, hold_days=hd, min_stocks_mult=3, return_weights=True)
        d2 = int((res_2["turnover"] > 0).sum())
        d3 = int((res_3["turnover"] > 0).sum())
        gap_days = (d2 - d3) / max(d2, 1) * 100
        sr2, sr3 = _sharpe(res_2["port_ret"]), _sharpe(res_3["port_ret"])
        log(f"   开仓日数:  mult=2 → {d2} 天 | mult=3 → {d3} 天 | 差 {d2 - d3} 天（{gap_days:.2f}%）")
        log(f"   换手合计:  mult=2 → {float(res_2['turnover'].sum()):.1f} | "
            f"mult=3 → {float(res_3['turnover'].sum()):.1f}")
        log(f"   LS 夏普  :  mult=2 → {sr2:.3f} | mult=3 → {sr3:.3f} | 差 {abs(sr2 - sr3):.4f}")
        log("")
        log("   注: 门槛只在「当日有效股数落在 [base*2, base*3) 区间」时才起作用。")
        log("       真实全量 universe（每日数百只有效）几乎不触发；")
        log("       稀疏截面（每日有效股数 < 15）才显著 —— 抽样运行可复现后者。")
        if gap_days < 1.0 and abs(sr2 - sr3) < 0.01:
            log("   判定: 对真实 universe 的结论**无影响**（持仓日差 < 1% 且夏普差 < 0.01）")
            log("   处置: 本次仅参数化、**不统一**，默认行为保持不变（决策记录见 docs/ 实施计划）")
        else:
            log("   判定: 差异不可忽略 → 维持现状，统一需单独评估对已发布报告的影响")

    except AssertionError as e:
        ok = False
        log("")
        log(f"❌ 对账失败: {e}")
    except FileNotFoundError as e:
        ok = False
        log("")
        log(f"❌ 数据缺失: {e}")

    log("")
    log(f"结论: {'✅ 全部通过' if ok else '❌ 存在问题，见上'}")
    log("=" * 72)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
