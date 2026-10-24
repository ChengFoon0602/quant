"""Alpha191 修复后的健康度守卫测试。

对应用户 2026-10-24 审计发现的问题，四条断言各守一类缺陷：

1. test_no_all_nan        —— 防止「因子静默失效」（原 9 个全 NaN）
2. test_cs_effective      —— 防止「因子无截面分散度」（原 30 个 CS_eff=0）
3. test_vwap_scale        —— 防止 VWAP 量纲再次错配（原 vwap/close 中位 1.31）
4. test_alpha001_invariant—— 对照组：全比率构成的因子，口径修复后应几乎不动

运行:
    pytest tests/test_alpha191_health.py -v
用例需要 adj1/adj3 缓存，缺失时整体 skip。
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from signals.alpha191 import list_factors
from signals.alpha191.calculator import compute_factor_matrix, _build_panel
from signals.alpha191.factors import DEGENERATE_FACTORS
from data.fetcher import load_daily
from data.price_adjust import vwap_scale_stats

CACHE = PROJECT / "data" / "cache"
SNAPSHOT = PROJECT / "tests" / "golden" / "alpha001_ic_snapshot.csv"
SNAPSHOT_SYMS = PROJECT / "tests" / "golden" / "alpha001_snapshot_symbols.txt"

START, END = "2018-01-01", "2020-12-31"
MIN_STOCKS = 30  # TSRANK 只有 5 档离散值，池太窄时截面必然并列


def _available_symbols(limit: int | None = None) -> list[str]:
    got = sorted(p.split("_adj1.csv")[0].replace("\\", "/").split("/")[-1]
                 for p in glob.glob(str(CACHE / "*_adj1.csv")))
    out = [s for s in got if (CACHE / f"{s}.csv").exists()]
    return out if limit is None else out[:limit]


SYMBOLS = _available_symbols(limit=100)  # 全量太慢；100 只足以暴露截面/量纲问题

pytestmark = pytest.mark.skipif(
    len(SYMBOLS) < MIN_STOCKS,
    reason=f"adj1 缓存不足（{len(SYMBOLS)}/{MIN_STOCKS} 只），先运行 data/download_adjust_cache.py",
)


@pytest.fixture(scope="module")
def factor_tensor():
    _, ft = compute_factor_matrix(
        SYMBOLS, list_factors(), start=START, end=END, verbose=False, strict=True
    )
    return ft


# ── 1. 无静默失效 ────────────────────────────────────────────
def test_no_all_nan(factor_tensor):
    bad = [fid for fid, df in factor_tensor.items()
           if fid not in DEGENERATE_FACTORS and df.isna().all().all()]
    assert not bad, f"这些因子全 NaN（未被标记为退化）: {bad}"


# ── 2. 截面分散度 ────────────────────────────────────────────
def test_cs_effective(factor_tensor):
    zero_disp = []
    for fid, df in factor_tensor.items():
        if fid in DEGENERATE_FACTORS:
            continue
        # 每日截面唯一值的中位数；<2 说明多数交易日全部股票取值相同，无法分组
        med = df.nunique(axis=1).median()
        if pd.notna(med) and med < 2:
            zero_disp.append(fid)
    assert not zero_disp, f"这些因子无截面分散度: {zero_disp}"


# ── 3. VWAP 量纲 ────────────────────────────────────────────
def test_vwap_scale():
    syms = SYMBOLS[:5]
    dfs = {}
    for s in syms:
        df = load_daily(s, adjust="1")
        if df is None:
            continue
        dfs[s] = df.loc[(df.index >= START) & (df.index <= END),
                        ["open", "high", "low", "close", "volume", "amount"]]
    assert len(dfs) >= 3, "样本不足"
    panel = _build_panel(dfs, vwap_mode="true")
    stats = vwap_scale_stats(panel["vwap"], panel["close"])

    assert 0.97 <= stats["median"] <= 1.03, (
        f"vwap/close 中位数 {stats['median']:.4f} 超出 [0.97, 1.03]——"
        f"量纲可能再次错配（修复前为 1.31）"
    )
    assert stats["max"] <= 1.15, f"vwap/close 最大值 {stats['max']:.4f} > 1.15"


# ── 4. 对照组不变量 ──────────────────────────────────────────
def _rank_ic_series(fac: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    ret = close.pct_change().shift(-1)
    out = {}
    for d in fac.index.intersection(ret.index):
        f, r = fac.loc[d], ret.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if pd.notna(ic):
            out[d] = ic
    return pd.Series(out).sort_index()


def test_alpha001_invariant():
    """alpha001 全部由比率构成（(close-open)/open、DELTA(LOG(volume))），
    不受复权口径与 VWAP 修复影响。若它变了，说明修复引入了新 bug。

    必须用快照时固定的股票池重算，否则池子变了 IC 序列自然变。
    """
    if not SNAPSHOT.exists() or not SNAPSHOT_SYMS.exists():
        pytest.skip("基线缺失，先运行 tests/make_alpha001_snapshot.py")

    syms = SNAPSHOT_SYMS.read_text(encoding="utf-8").strip().split(",")
    _, ft = compute_factor_matrix(syms, ["alpha001"], start=START, end=END,
                                  verbose=False, strict=True)
    close = pd.DataFrame({s: load_daily(s, adjust="2")["close"] for s in syms}).sort_index()
    close = close.loc[(close.index >= START) & (close.index <= END)]
    current = _rank_ic_series(ft["alpha001"], close)

    base = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)["ic"].sort_index()
    common = current.index.intersection(base.index)
    assert len(common) > 100, "快照与当前重叠不足"

    rho = current.loc[common].corr(base.loc[common], method="spearman")
    assert rho > 0.999, f"alpha001 Rank IC 序列漂移（Spearman={rho:.6f}）"
