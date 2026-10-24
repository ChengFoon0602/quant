"""生成 alpha001 的 Rank IC 快照，作为口径修复的对照组基线。

alpha001 = -CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)
全部分量都是比率/对数差分 → 对复权口径、VWAP 定义都不敏感。
修复前后它必须几乎不动，因此适合当「修复未引入新 bug」的见证因子。

用法:
    python tests/make_alpha001_snapshot.py
    python tests/make_alpha001_snapshot.py --force   # 覆盖已有快照
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from data.fetcher import load_daily
from signals.alpha191.calculator import compute_factor_matrix

CACHE = PROJECT / "data" / "cache"
OUT = PROJECT / "tests" / "golden" / "alpha001_ic_snapshot.csv"
SYMS_OUT = PROJECT / "tests" / "golden" / "alpha001_snapshot_symbols.txt"
N_SYMS = 30
START, END = "2018-01-01", "2020-12-31"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"快照已存在: {OUT}（--force 覆盖）")
        return 0

    syms = sorted(p.replace("\\", "/").split("/")[-1].replace("_adj1.csv", "")
                  for p in glob.glob(str(CACHE / "*_adj1.csv")))
    syms = [s for s in syms if (CACHE / f"{s}.csv").exists()][:N_SYMS]
    if len(syms) < N_SYMS:
        print(f"adj1 缓存不足（{len(syms)} 只），先运行 data/download_adjust_cache.py")
        return 1

    _, ft = compute_factor_matrix(syms, ["alpha001"], start=START, end=END,
                                  verbose=False, strict=True)
    fac = ft["alpha001"]

    close = pd.DataFrame({s: load_daily(s, adjust="2")["close"] for s in syms}).sort_index()
    close = close.loc[(close.index >= START) & (close.index <= END)]
    ret = close.pct_change().shift(-1)

    ics = {}
    for d in fac.index.intersection(ret.index):
        f, r = fac.loc[d], ret.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if pd.notna(ic):
            ics[d] = ic

    s = pd.Series(ics).sort_index().rename("ic")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(OUT)

    print(f"股票 {len(syms)} 只，日期 {START}~{END}")
    print(f"Rank IC: 均值={s.mean():+.4f}  IC_IR={s.mean()/s.std():+.3f}  有效天数={len(s)}")
    SYMS_OUT.write_text(",".join(syms), encoding="utf-8")
    print(f"已写入 {OUT}")
    print(f"股票池已固定到 {SYMS_OUT}（{len(syms)} 只）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
