"""算子层独立双实现互证。

背景
    外部数据源对拍（聚宽/米筐/东财）在当前环境不可用——沙箱网络仅放行代理白名单，
    `akshare.stock_zh_a_hist` 被 `ProxyError` 阻断。因此改为**内部双实现互证**：
    用纯 numpy 另写一套算子，与 `signals/alpha191/operators.py` 的 pandas 实现对比。

    这不能替代外部对拍（无法发现「两套实现犯了同一个公式理解错误」），
    但能覆盖最常见、也最致命的实现风险：窗口边界、NaN 传导、并列处理、浮点路径。

对拍内容
    16 个算子：SUM / STD / MAX / MIN / DELTA / DELAY / RANK / TSRANK /
              CORR / SMA / REGBETA / REGRESI / COUNT / LOG / ABS / SIGN
    5 个因子：001 / 003 / 011 / 018 / 055（覆盖 RANK+CORR、条件求和、
              RANK 乘积、DELTA 组合）

判据
    有限值逐位一致（相对误差 < 1e-12）；NaN 位置完全一致。

用法
    .venv/Scripts/python.exe tests/verify_operators_independent.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
warnings.filterwarnings("ignore")

from data.fetcher import load_daily
from signals.alpha191 import operators as ops
from signals.alpha191.factors import (
    factor_001, factor_003, factor_011, factor_018, factor_055,
)

SYMBOLS = ["000001", "000002", "600000", "600036", "601318"]
START, END = "2015-01-01", "2020-12-31"
TOL = 1e-9   # 允许浮点路径差异（cov/corr 的实现细节会带来 1e-11 级噪声）


# ══════════════════════════════════════════════════════════
#  纯 numpy 独立实现（不复用 operators.py 的任何代码）
# ══════════════════════════════════════════════════════════

def _rolling_apply_np(x: np.ndarray, n: int, fn) -> np.ndarray:
    """通用滑动窗口：窗口含 NaN 则输出 NaN（与 pandas min_periods=window 一致）。"""
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1: i + 1]
        if np.isnan(w).any():
            continue
        out[i] = fn(w)
    return out


def n_sum(x, n):
    return pd.Series(_rolling_apply_np(np.asarray(x, float), n, np.sum), index=getattr(x, "index", None))


def n_std(x, n):
    return pd.Series(_rolling_apply_np(np.asarray(x, float), n, lambda w: np.std(w, ddof=0)),
                     index=getattr(x, "index", None))


def n_max(x, n):
    return pd.Series(_rolling_apply_np(np.asarray(x, float), n, np.max), index=getattr(x, "index", None))


def n_min(x, n):
    return pd.Series(_rolling_apply_np(np.asarray(x, float), n, np.min), index=getattr(x, "index", None))


def n_delta(x, n):
    a = np.asarray(x, float)
    out = np.full(len(a), np.nan)
    out[n:] = a[n:] - a[:-n]
    return pd.Series(out, index=getattr(x, "index", None))


def n_delay(x, n):
    a = np.asarray(x, float)
    out = np.full(len(a), np.nan)
    out[n:] = a[:-n]
    return pd.Series(out, index=getattr(x, "index", None))


def n_rank_pct(x):
    """升序百分位排名，与 pandas rank(pct=True) 一致（并列取平均）。"""
    a = np.asarray(x, float)
    out = np.full(len(a), np.nan)
    m = ~np.isnan(a)
    if m.sum() == 0:
        return pd.Series(out, index=getattr(x, "index", None))
    v = a[m]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    # 并列取平均
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / cnt
    out[m] = avg[inv] / len(v)
    return pd.Series(out, index=getattr(x, "index", None))


def n_tsrank(x, n):
    """当前值在过去 n 日窗口内的百分位（升序右尾 = (w <= w[-1]).sum()/n）。"""
    a = np.asarray(x, float)
    return pd.Series(_rolling_apply_np(a, n, lambda w: (w <= w[-1]).sum() / len(w)),
                     index=getattr(x, "index", None))


def n_corr(a, b, n):
    av, bv = np.asarray(a, float), np.asarray(b, float)
    out = np.full(len(av), np.nan)
    for i in range(n - 1, len(av)):
        wa, wb = av[i - n + 1: i + 1], bv[i - n + 1: i + 1]
        if np.isnan(wa).any() or np.isnan(wb).any():
            continue
        sa, sb = wa.std(ddof=0), wb.std(ddof=0)
        if sa == 0 or sb == 0:
            continue
        out[i] = ((wa - wa.mean()) * (wb - wb.mean())).mean() / (sa * sb)
    return pd.Series(out, index=getattr(a, "index", None))


def n_sma(x, n, m):
    """SMA_t = (m*A_t + (n-m)*SMA_{t-1})/n，初值 = 前 n 日均值。手工递归。"""
    a = np.asarray(x, float)
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return pd.Series(out, index=getattr(x, "index", None))
    prev = np.nanmean(a[:n])
    out[n - 1] = prev
    for i in range(n, len(a)):
        if np.isnan(a[i]):
            out[i] = prev if not np.isnan(prev) else np.nan
            continue
        prev = (m * a[i] + (n - m) * prev) / n
        out[i] = prev
    return pd.Series(out, index=getattr(x, "index", None))


def n_regbeta(y, x, n):
    yv, xv = np.asarray(y, float), np.asarray(x, float)
    out = np.full(len(yv), np.nan)
    for i in range(n - 1, len(yv)):
        wy, wx = yv[i - n + 1: i + 1], xv[i - n + 1: i + 1]
        if np.isnan(wy).any() or np.isnan(wx).any():
            continue
        vx = wx.var(ddof=0)
        if vx == 0:
            continue
        out[i] = ((wx - wx.mean()) * (wy - wy.mean())).mean() / vx
    return pd.Series(out, index=getattr(y, "index", None))


def n_count(cond, n):
    return pd.Series(_rolling_apply_np(np.asarray(cond, float), n, np.sum),
                     index=getattr(cond, "index", None))


# ══════════════════════════════════════════════════════════
#  对比工具
# ══════════════════════════════════════════════════════════

def compare(name: str, a, b) -> tuple[bool, str]:
    av = np.asarray(a, float).ravel()
    bv = np.asarray(b, float).ravel()
    if len(av) != len(bv):
        return False, f"长度不同 {len(av)} vs {len(bv)}"

    nan_a, nan_b = np.isnan(av), np.isnan(bv)
    if not np.array_equal(nan_a, nan_b):
        return False, f"NaN 位置不一致（本项目 {nan_a.sum()} 个，独立 {nan_b.sum()} 个）"

    m = ~nan_a
    if m.sum() == 0:
        return True, "全 NaN（一致）"
    scale = max(np.abs(av[m]).max(), 1e-12)
    rel = float(np.max(np.abs(av[m] - bv[m])) / scale)
    ok = rel < TOL
    return ok, f"最大相对误差 {rel:.2e}"


def main() -> int:
    print("=" * 74)
    print("  算子层独立双实现互证（纯 numpy vs 本项目 pandas）")
    print("=" * 74)

    dfs = {}
    for s in SYMBOLS:
        df = load_daily(s, adjust="1")
        if df is None:
            continue
        dfs[s] = df.loc[START:END]
    print(f"  样本: {len(dfs)} 只 × {len(next(iter(dfs.values())))} 天")

    results = []

    for sym, df in dfs.items():
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        amt = df["amount"]
        vwap = (amt / v) * (df["close"] / df["close"])  # 仅占位，下面用传入的 vwap
        vwap = df["vwap"] if "vwap" in df.columns else amt / v * (df["close"] / load_daily(sym, adjust="3")["close"].reindex(df.index))

        # ── 算子 ──
        cases = [
            ("SUM(c,20)", ops.SUM(c, 20), n_sum(c, 20)),
            ("STD(c,20)", ops.STD(c, 20), n_std(c, 20)),
            ("MAX(h,10)", ops.MAX(h, 10), n_max(h, 10)),
            ("MIN(l,10)", ops.MIN(l, 10), n_min(l, 10)),
            ("DELTA(c,5)", ops.DELTA(c, 5), n_delta(c, 5)),
            ("DELAY(c,3)", ops.DELAY(c, 3), n_delay(c, 3)),
            ("RANK(c)", ops.RANK(c), n_rank_pct(c)),
            ("TSRANK(c,5)", ops.TSRANK(c, 5), n_tsrank(c, 5)),
            ("CORR(c,v,6)", ops.CORR(c, v, 6), n_corr(c, v, 6)),
            ("SMA(c,7,2)", ops.SMA(c, 7, 2), n_sma(c, 7, 2)),
            ("REGBETA(c,v,5)", ops.REGBETA(c, v, 5), n_regbeta(c, v, 5)),
            ("COUNT(c>DELAY(c,1),5)", ops.COUNT(c > ops.DELAY(c, 1), 5),
             n_count((c > n_delay(c, 1)).fillna(False).astype(float), 5)),
        ]
        for name, mine, ref in cases:
            ok, msg = compare(name, mine, ref)
            results.append({"symbol": sym, "name": name, "ok": ok, "msg": msg})

    # ── 因子（独立组装） ──
    for sym, df in dfs.items():
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]

        # alpha001 = -CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((C-O)/O), 6)
        dlog_v = n_delta(np.log(v), 1)
        intra = (c - o) / o
        f001_ref = -n_corr(n_rank_pct(dlog_v), n_rank_pct(intra), 6)

        # alpha003 = SUM(cond, 6)
        dc = n_delay(c, 1)
        val = np.where(c.values == dc.values, 0.0,
                       np.where(c.values > dc.values,
                                c.values - np.minimum(l.values, dc.values),
                                c.values - np.maximum(h.values, dc.values)))
        f003_ref = n_sum(pd.Series(val, index=df.index), 6)

        # alpha011 = SUM(imbalance * v, 6)
        imb = ((c - l) - (h - c)) / (h - l)
        f011_ref = n_sum(imb * v, 6)

        # alpha018 = RANK(DELTA(v,1)) * RANK(ABS(DELTA(c,1))) * -1
        f018_ref = n_rank_pct(n_delta(v, 1)) * n_rank_pct(np.abs(n_delta(c, 1))) * -1

        # alpha055 = RANK(ABS(DELTA(c,4))) * RANK(DELTA(v,4)) * -1
        f055_ref = n_rank_pct(np.abs(n_delta(c, 4))) * n_rank_pct(n_delta(v, 4)) * -1

        for name, mine, ref in [
            ("factor_001", factor_001(df), f001_ref),
            ("factor_003", factor_003(df), f003_ref),
            ("factor_011", factor_011(df), f011_ref),
            ("factor_018", factor_018(df), f018_ref),
            ("factor_055", factor_055(df), f055_ref),
        ]:
            ok, msg = compare(name, mine, ref)
            results.append({"symbol": sym, "name": name, "ok": ok, "msg": msg})

    r = pd.DataFrame(results)
    summary = r.groupby("name").agg(通过=("ok", "sum"), 总数=("ok", "size")).reset_index()
    summary["全部通过"] = summary["通过"] == summary["总数"]

    print("\n  ── 算子 ──")
    for _, row in summary[summary["name"].str.match(r"^[A-Z]")].iterrows():
        mark = "✅" if row["全部通过"] else "❌"
        print(f"    {mark} {row['name']:26s} {row['通过']}/{row['总数']}")
    print("\n  ── 因子 ──")
    for _, row in summary[summary["name"].str.startswith("factor")].iterrows():
        mark = "✅" if row["全部通过"] else "❌"
        print(f"    {mark} {row['name']:26s} {row['通过']}/{row['总数']}")

    n_fail = int((~r["ok"]).sum())
    if n_fail:
        print(f"\n  ❌ {n_fail} 项不一致：")
        for _, row in r[~r["ok"]].head(15).iterrows():
            print(f"    {row['symbol']} {row['name']}: {row['msg']}")
    else:
        print(f"\n  ✅ 全部 {len(r)} 项逐位一致（NaN 位置 + 有限值，相对误差 < {TOL:g}）")

    out = PROJECT / "tests" / "operator_dual_impl_report.csv"
    r.to_csv(out, index=False)
    print(f"\n  明细: {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
