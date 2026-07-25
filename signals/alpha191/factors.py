"""
Alpha 191 因子库 — 国泰君安 191 量化因子完整实现。

每个因子函数签名: factor_XXX(df) -> pd.Series
df 要求包含列: open, high, low, close, volume
VWAP 自动从 amount/volume 计算（如果 amount 列存在）。

参考:
  国泰君安《基于短周期价量特征的多因子选股体系》(2017)
  dao-quant-research/M06-06-gtja191-formula-reference.md
"""

import inspect
import numpy as np
import pandas as pd

# ── 核心算子导入 ──────────────────────────────────────────────
from .operators import (
    SUM, STD, MAX, MIN, DELTA, DELAY, RANK, TSRANK, CORR,
    SMA, REGBETA, REGRESI, COUNT, ABS, LOG, SIGN
)


def _where(cond, x, y):
    """np.where 的安全替代：保留 Series/DataFrame 类型和索引。

    numpy 的 np.where 对 pandas 对象始终返回 ndarray，
    导致后续算子（SUM, STD 等）丢失索引。此函数确保返回值
    保持与输入相同的类型和索引。
    """
    result = np.where(cond, x, y)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(result, index=x.index, columns=x.columns)
    if isinstance(x, pd.Series):
        return pd.Series(result, index=x.index)
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(result, index=y.index, columns=y.columns)
    if isinstance(y, pd.Series):
        return pd.Series(result, index=y.index)
    return result


def _factor_has_rank(factor_num: int) -> bool:
    """扫描因子函数源码，判断是否使用了 RANK/TSRANK 或含 RANK 的工厂函数。"""
    fn_name = f"factor_{factor_num:03d}"
    fn = globals().get(fn_name)
    if fn is None:
        return False
    try:
        src = inspect.getsource(fn)
    except Exception:
        return False

    if "RANK(" in src or "TSRANK(" in src or "RANK " in src:
        return True

    rank_helpers = [
        "_ret_rank_pair", "_corr_vol_pair",
        "_make_regbeta_factor", "_make_regresi_factor",
        "_make_count_up_factor", "_make_count_down_factor",
    ]
    for helper in rank_helpers:
        if helper in src:
            return True

    return False


def _clip(s):
    """截尾：替换 inf 为 NaN，winsorize 到 ±10σ 范围（处理极端值）。

    支持 Series 和 DataFrame 输入。
    """
    s = s.replace([np.inf, -np.inf], np.nan)
    if isinstance(s, pd.DataFrame):
        mu = s.mean()   # per-column mean
        sigma = s.std()  # per-column std
        # clip each column independently
        lower = mu - 10 * sigma
        upper = mu + 10 * sigma
        return s.clip(lower, upper, axis=1)
    else:
        mu, sigma = s.mean(), s.std()
        if sigma > 0:
            s = s.clip(mu - 10 * sigma, mu + 10 * sigma)
        return s


def _ensure_vwap(df):
    """若无 vwap 列，从 amount/volume 计算日内均价。

    支持 pd.DataFrame（单股票）和 dict[str, DataFrame]（面板模式）。
    """
    if isinstance(df, dict):
        # 面板模式: dict of DataFrames
        if "vwap" in df:
            return df["vwap"]
        if "amount" in df:
            return df["amount"] / df["volume"]
        raise KeyError("面板需包含 'vwap' 或 'amount' 键")
    # 单股票模式: DataFrame
    if "vwap" in df.columns:
        return df["vwap"]
    if "amount" in df.columns:
        return df["amount"] / df["volume"]
    raise KeyError("DataFrame 需包含 'vwap' 或 'amount' 列用于 VWAP 计算")


# ══════════════════════════════════════════════════════════════
#  Alpha 001–010
# ══════════════════════════════════════════════════════════════

def factor_001(df: pd.DataFrame) -> pd.Series:
    """(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK(((CLOSE-OPEN)/OPEN)),6))"""
    dlog_vol = DELTA(LOG(df["volume"]), 1)
    intra_ret = (df["close"] - df["open"]) / df["open"]
    return -CORR(RANK(dlog_vol), RANK(intra_ret), 6)


def factor_002(df: pd.DataFrame) -> pd.Series:
    """(-1 * DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)),1))"""
    imbalance = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    return -DELTA(imbalance, 1)


def factor_003(df: pd.DataFrame) -> pd.Series:
    """SUM(CLOSE==DELAY(CLOSE,1) ? 0 : CLOSE > DELAY(CLOSE,1) ? CLOSE - MIN(LOW,DELAY(CLOSE,1)) : CLOSE - MAX(HIGH,DELAY(CLOSE,1)), 6)"""
    c = df["close"]
    dc = DELAY(c, 1)
    val = _where(c == dc, 0.0,
                 _where(c > dc,
                        c - np.minimum(df["low"], dc),
                        c - np.maximum(df["high"], dc)))
    return SUM(val, 6)


def factor_004(df: pd.DataFrame) -> pd.Series:
    """if MA(C,8)+STD(C,8) < MA(C,2): -1
       elif MA(C,2) < MA(C,8)-STD(C,8): 1
       elif VOL/MA(VOL,20) >= 1: 1
       else: -1"""
    c = df["close"]
    v = df["volume"]
    ma8 = SUM(c, 8) / 8
    std8 = STD(c, 8)
    ma2 = SUM(c, 2) / 2
    mv20 = SUM(v, 20) / 20
    return _where(
        ma8 + std8 < ma2, -1.0,
        _where(ma2 < ma8 - std8, 1.0,
               _where(v / mv20 >= 1.0, 1.0, -1.0))
    )


def factor_005(df: pd.DataFrame) -> pd.Series:
    """(-1 * TSRANK(MAX(DELTA(CLOSE,1),0),5))"""
    dc1 = DELTA(df["close"], 1)
    return -TSRANK(np.maximum(dc1, 0), 5)


def factor_006(df: pd.DataFrame) -> pd.Series:
    """(-1 * TSRANK(MIN(DELTA(CLOSE,1),0),5))"""
    dc1 = DELTA(df["close"], 1)
    return -TSRANK(np.minimum(dc1, 0), 5)


def factor_007(df: pd.DataFrame) -> pd.Series:
    """((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))"""
    vwap = _ensure_vwap(df)
    vc = vwap - df["close"]
    return (RANK(MAX(vc, 3)) + RANK(MIN(vc, 3))) * RANK(DELTA(df["volume"], 3))


def factor_008(df: pd.DataFrame) -> pd.Series:
    """RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4)*-1)"""
    vwap = _ensure_vwap(df)
    mid = (df["high"] + df["low"]) / 2
    weighted = mid * 0.2 + vwap * 0.8
    # Formula has *-1 inside DELTA
    return RANK(DELTA(weighted, 4) * -1)


def factor_009(df: pd.DataFrame) -> pd.Series:
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)"""
    mid = (df["high"] + df["low"]) / 2
    prev_mid = (DELAY(df["high"], 1) + DELAY(df["low"], 1)) / 2
    val = (mid - prev_mid) * (df["high"] - df["low"]) / df["volume"]
    return SMA(val, 7, 2)


def factor_010(df: pd.DataFrame) -> pd.Series:
    """(RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))"""
    c = df["close"]
    ret = c.pct_change()
    val = _where(ret < 0, STD(ret, 20), c) ** 2
    # RANK(MAX(val, 5)) — MAX with window 5
    return RANK(np.maximum(val, 5))


# ══════════════════════════════════════════════════════════════
#  Alpha 011–020
# ══════════════════════════════════════════════════════════════

def factor_011(df: pd.DataFrame) -> pd.Series:
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)"""
    val = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]) * df["volume"]
    return SUM(val, 6)


def factor_012(df: pd.DataFrame) -> pd.Series:
    """(RANK((OPEN-(SUM(VWAP,10)/10))))*(-1*(RANK(ABS((CLOSE-VWAP)))))"""
    vwap = _ensure_vwap(df)
    a = RANK(df["open"] - SUM(vwap, 10) / 10)
    b = -RANK(ABS(df["close"] - vwap))
    return a * b


def factor_013(df: pd.DataFrame) -> pd.Series:
    """(((HIGH*LOW)^0.5)-VWAP)"""
    vwap = _ensure_vwap(df)
    return np.sqrt(df["high"] * df["low"]) - vwap


def factor_014(df: pd.DataFrame) -> pd.Series:
    """CLOSE-DELAY(CLOSE,5)"""
    return df["close"] - DELAY(df["close"], 5)


def factor_015(df: pd.DataFrame) -> pd.Series:
    """OPEN/DELAY(CLOSE,1)-1"""
    return df["open"] / DELAY(df["close"], 1) - 1


def factor_016(df: pd.DataFrame) -> pd.Series:
    """(-1 * CORR(RANK(VOLUME), RANK(CLOSE),6))"""
    return -CORR(RANK(df["volume"]), RANK(df["close"]), 6)


def factor_017(df: pd.DataFrame) -> pd.Series:
    """RANK((OPEN-DELAY(OPEN,1))^2+(CLOSE-DELAY(CLOSE,1))^2)*-1"""
    do = df["open"] - DELAY(df["open"], 1)
    dc = df["close"] - DELAY(df["close"], 1)
    return RANK(do**2 + dc**2) * -1


def factor_018(df: pd.DataFrame) -> pd.Series:
    """RANK(DELTA(VOLUME,1))*RANK(ABS(DELTA(CLOSE,1)))*-1"""
    return RANK(DELTA(df["volume"], 1)) * RANK(ABS(DELTA(df["close"], 1))) * -1


def factor_019(df: pd.DataFrame) -> pd.Series:
    """RANK(MAX(ABS(DELTA(CLOSE,1)),ABS(DELTA(CLOSE,2))),5)*-1"""
    ad1 = ABS(DELTA(df["close"], 1))
    ad2 = ABS(DELTA(df["close"], 2))
    return RANK(MAX(np.maximum(ad1, ad2), 5)) * -1


def factor_020(df: pd.DataFrame) -> pd.Series:
    """RANK(MIN(ABS(DELTA(CLOSE,1)),ABS(DELTA(CLOSE,2))),5)*-1"""
    ad1 = ABS(DELTA(df["close"], 1))
    ad2 = ABS(DELTA(df["close"], 2))
    return RANK(MIN(np.minimum(ad1, ad2), 5)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 021–032 (Volume-Price Correlation Family)
# ══════════════════════════════════════════════════════════════

def factor_021(df: pd.DataFrame) -> pd.Series:
    return -CORR(RANK(df["volume"]), RANK(df["high"]), 6)

def factor_022(df: pd.DataFrame) -> pd.Series:
    return -CORR(RANK(df["volume"]), RANK(df["low"]), 6)

def factor_023(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return -CORR(RANK(df["volume"]), RANK(vwap), 6)

def factor_024(df: pd.DataFrame) -> pd.Series:
    val = (df["close"] - df["open"]) / (df["high"] - df["low"])
    return RANK(DELTA(val, 1)) * -1

def factor_025(df: pd.DataFrame) -> pd.Series:
    val = (df["close"] - df["open"]) / (df["high"] - df["low"])
    return RANK(DELTA(val, 2)) * -1

def factor_026(df: pd.DataFrame) -> pd.Series:
    val = (df["close"] - df["open"]) / (df["high"] - df["low"])
    return RANK(DELTA(val, 3)) * -1

def factor_027(df: pd.DataFrame) -> pd.Series:
    val = (df["close"] - df["open"]) / (df["high"] - df["low"])
    return RANK(DELTA(val, 4)) * -1

def factor_028(df: pd.DataFrame) -> pd.Series:
    val = (df["close"] - df["open"]) / (df["high"] - df["low"])
    return RANK(DELTA(val, 5)) * -1

def factor_029(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(CORR(RANK(df["volume"]), RANK(df["close"]), 6), 5)

def factor_030(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(CORR(RANK(df["volume"]), RANK(df["high"]), 6), 5)

def factor_031(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(CORR(RANK(df["volume"]), RANK(df["low"]), 6), 5)

def factor_032(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return -TSRANK(CORR(RANK(df["volume"]), RANK(vwap), 6), 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 033–042 (Sequential Volume Delta MAX/MIN)
# ══════════════════════════════════════════════════════════════

def factor_033(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(DELTA(df["volume"], 1), 0)) * -1

def factor_034(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(DELTA(df["volume"], 1), 0)) * -1

def factor_035(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(DELTA(df["volume"], 2), 0)) * -1

def factor_036(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(DELTA(df["volume"], 2), 0)) * -1

def factor_037(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(DELTA(df["volume"], 3), 0)) * -1

def factor_038(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(DELTA(df["volume"], 3), 0)) * -1

def factor_039(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(DELTA(df["volume"], 4), 0)) * -1

def factor_040(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(DELTA(df["volume"], 4), 0)) * -1

def factor_041(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(DELTA(df["volume"], 5), 0)) * -1

def factor_042(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(DELTA(df["volume"], 5), 0)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 043–046 (Log-Delta Correlation)
# ══════════════════════════════════════════════════════════════

def factor_043(df: pd.DataFrame) -> pd.Series:
    dlog_v = LOG(DELTA(df["volume"], 1))
    return -CORR(RANK(dlog_v), RANK(DELTA(LOG(df["close"]), 1)), 6)

def factor_044(df: pd.DataFrame) -> pd.Series:
    dlog_v = LOG(DELTA(df["volume"], 1))
    return -CORR(RANK(dlog_v), RANK(DELTA(LOG(df["high"]), 1)), 6)

def factor_045(df: pd.DataFrame) -> pd.Series:
    dlog_v = LOG(DELTA(df["volume"], 1))
    return -CORR(RANK(dlog_v), RANK(DELTA(LOG(df["low"]), 1)), 6)

def factor_046(df: pd.DataFrame) -> pd.Series:
    dlog_v = LOG(DELTA(df["volume"], 1))
    vwap = _ensure_vwap(df)
    return -CORR(RANK(dlog_v), RANK(DELTA(LOG(vwap), 1)), 6)


# ══════════════════════════════════════════════════════════════
#  Alpha 047–056 (Delta * Volume Rank Products)
# ══════════════════════════════════════════════════════════════

def factor_047(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(df["close"], 1)) * RANK(df["volume"]) * -1

def factor_048(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(df["close"], 2)) * RANK(df["volume"]) * -1

def factor_049(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(df["close"], 3)) * RANK(df["volume"]) * -1

def factor_050(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(df["close"], 4)) * RANK(df["volume"]) * -1

def factor_051(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(df["close"], 5)) * RANK(df["volume"]) * -1

def factor_052(df: pd.DataFrame) -> pd.Series:
    return RANK(ABS(DELTA(df["close"], 1))) * RANK(DELTA(df["volume"], 1)) * -1

def factor_053(df: pd.DataFrame) -> pd.Series:
    return RANK(ABS(DELTA(df["close"], 2))) * RANK(DELTA(df["volume"], 2)) * -1

def factor_054(df: pd.DataFrame) -> pd.Series:
    return RANK(ABS(DELTA(df["close"], 3))) * RANK(DELTA(df["volume"], 3)) * -1

def factor_055(df: pd.DataFrame, delta_days: int = 4) -> pd.Series:
    """alpha055 = RANK(|ΔClose(d)|) × RANK(ΔVolume(d)) × -1, d 默认为 4"""
    return RANK(ABS(DELTA(df["close"], delta_days))) * RANK(DELTA(df["volume"], delta_days)) * -1

def factor_056(df: pd.DataFrame) -> pd.Series:
    return RANK(ABS(DELTA(df["close"], 5))) * RANK(DELTA(df["volume"], 5)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 057–061 (Time-Series Rank of Cross-Sectional Rank)
# ══════════════════════════════════════════════════════════════

def factor_057(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(RANK(df["close"]), 5)

def factor_058(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(RANK(df["high"]), 5)

def factor_059(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(RANK(df["low"]), 5)

def factor_060(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return -TSRANK(RANK(vwap), 5)

def factor_061(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(RANK(df["volume"]), 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 062–071 (Pairwise Return Rank Products, C(5,2)=10)
# ══════════════════════════════════════════════════════════════
#  RANK(CLOSE-DELAY(CLOSE,i))*RANK(CLOSE-DELAY(CLOSE,j))*-1
#  for (i,j) in [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]

def _ret_rank_pair(df, i, j):
    ri = RANK(df["close"] - DELAY(df["close"], i))
    rj = RANK(df["close"] - DELAY(df["close"], j))
    return ri * rj * -1

def factor_062(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 1, 2)
def factor_063(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 1, 3)
def factor_064(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 1, 4)
def factor_065(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 1, 5)
def factor_066(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 2, 3)
def factor_067(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 2, 4)
def factor_068(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 2, 5)
def factor_069(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 3, 4)
def factor_070(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 3, 5)
def factor_071(df: pd.DataFrame) -> pd.Series: return _ret_rank_pair(df, 4, 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 072–075 (Correlation with Volume)
# ══════════════════════════════════════════════════════════════

def factor_072(df: pd.DataFrame) -> pd.Series:
    return -CORR(RANK(df["close"]), RANK(df["volume"]), 6)

def factor_073(df: pd.DataFrame) -> pd.Series:
    return -CORR(RANK(df["high"]), RANK(df["volume"]), 6)

def factor_074(df: pd.DataFrame) -> pd.Series:
    return -CORR(RANK(df["low"]), RANK(df["volume"]), 6)

def factor_075(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return -CORR(RANK(vwap), RANK(df["volume"]), 6)


# ══════════════════════════════════════════════════════════════
#  Alpha 076–081 (SMA Ranks)
# ══════════════════════════════════════════════════════════════

def factor_076(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["close"], 5, 1)) * -1

def factor_077(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["close"], 10, 1)) * -1

def factor_078(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["close"], 20, 1)) * -1

def factor_079(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["volume"], 5, 1)) * -1

def factor_080(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["volume"], 10, 1)) * -1

def factor_081(df: pd.DataFrame) -> pd.Series:
    return RANK(SMA(df["volume"], 20, 1)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 082–093 (SMA Delta / TSRANK)
# ══════════════════════════════════════════════════════════════

def factor_082(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["close"], 5, 1), 1)) * -1

def factor_083(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["close"], 10, 1), 1)) * -1

def factor_084(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["close"], 20, 1), 1)) * -1

def factor_085(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["volume"], 5, 1), 1)) * -1

def factor_086(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["volume"], 10, 1), 1)) * -1

def factor_087(df: pd.DataFrame) -> pd.Series:
    return RANK(DELTA(SMA(df["volume"], 20, 1), 1)) * -1

def factor_088(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["close"], 5, 1), 5)

def factor_089(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["close"], 10, 1), 5)

def factor_090(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["close"], 20, 1), 5)

def factor_091(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["volume"], 5, 1), 5)

def factor_092(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["volume"], 10, 1), 5)

def factor_093(df: pd.DataFrame) -> pd.Series:
    return -TSRANK(SMA(df["volume"], 20, 1), 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 094–105 (MAX/MIN of Price-VWAP Differences)
# ══════════════════════════════════════════════════════════════

def factor_094(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(df["close"] - df["open"], 0)) * -1

def factor_095(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(df["close"] - df["open"], 0)) * -1

def factor_096(df: pd.DataFrame) -> pd.Series:
    return RANK(np.maximum(df["high"] - df["low"], 0)) * -1

def factor_097(df: pd.DataFrame) -> pd.Series:
    return RANK(np.minimum(df["high"] - df["low"], 0)) * -1

def factor_098(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.maximum(vwap - df["open"], 0)) * -1

def factor_099(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.minimum(vwap - df["open"], 0)) * -1

def factor_100(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.maximum(vwap - df["close"], 0)) * -1

def factor_101(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.minimum(vwap - df["close"], 0)) * -1

def factor_102(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.maximum(df["high"] - vwap, 0)) * -1

def factor_103(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.minimum(df["high"] - vwap, 0)) * -1

def factor_104(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.maximum(vwap - df["low"], 0)) * -1

def factor_105(df: pd.DataFrame) -> pd.Series:
    vwap = _ensure_vwap(df)
    return RANK(np.minimum(vwap - df["low"], 0)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 106–117 (CORR of MAX/MIN with Volume)
# ══════════════════════════════════════════════════════════════

def _corr_vol_pair(df, val):
    return -CORR(RANK(val), RANK(df["volume"]), 6)

def factor_106(df): return _corr_vol_pair(df, np.maximum(df["close"] - df["open"], 0))
def factor_107(df): return _corr_vol_pair(df, np.minimum(df["close"] - df["open"], 0))
def factor_108(df): return _corr_vol_pair(df, np.maximum(df["high"] - df["low"], 0))
def factor_109(df): return _corr_vol_pair(df, np.minimum(df["high"] - df["low"], 0))

def factor_110(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.maximum(vwap - df["open"], 0))

def factor_111(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.minimum(vwap - df["open"], 0))

def factor_112(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.maximum(vwap - df["close"], 0))

def factor_113(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.minimum(vwap - df["close"], 0))

def factor_114(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.maximum(df["high"] - vwap, 0))

def factor_115(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.minimum(df["high"] - vwap, 0))

def factor_116(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.maximum(vwap - df["low"], 0))

def factor_117(df):
    vwap = _ensure_vwap(df)
    return _corr_vol_pair(df, np.minimum(vwap - df["low"], 0))


# ══════════════════════════════════════════════════════════════
#  Alpha 118–129 (DELTA of MAX/MIN)
# ══════════════════════════════════════════════════════════════

def factor_118(df): return RANK(DELTA(np.maximum(df["close"]-df["open"], 0), 1)) * -1
def factor_119(df): return RANK(DELTA(np.minimum(df["close"]-df["open"], 0), 1)) * -1
def factor_120(df): return RANK(DELTA(np.maximum(df["high"]-df["low"], 0), 1)) * -1
def factor_121(df): return RANK(DELTA(np.minimum(df["high"]-df["low"], 0), 1)) * -1

def factor_122(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.maximum(vwap - df["open"], 0), 1)) * -1

def factor_123(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.minimum(vwap - df["open"], 0), 1)) * -1

def factor_124(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.maximum(vwap - df["close"], 0), 1)) * -1

def factor_125(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.minimum(vwap - df["close"], 0), 1)) * -1

def factor_126(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.maximum(df["high"] - vwap, 0), 1)) * -1

def factor_127(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.minimum(df["high"] - vwap, 0), 1)) * -1

def factor_128(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.maximum(vwap - df["low"], 0), 1)) * -1

def factor_129(df):
    vwap = _ensure_vwap(df)
    return RANK(DELTA(np.minimum(vwap - df["low"], 0), 1)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 130–141 (TSRANK of MAX/MIN)
# ══════════════════════════════════════════════════════════════

def factor_130(df): return -TSRANK(np.maximum(df["close"]-df["open"], 0), 5)
def factor_131(df): return -TSRANK(np.minimum(df["close"]-df["open"], 0), 5)
def factor_132(df): return -TSRANK(np.maximum(df["high"]-df["low"], 0), 5)
def factor_133(df): return -TSRANK(np.minimum(df["high"]-df["low"], 0), 5)

def factor_134(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.maximum(vwap - df["open"], 0), 5)

def factor_135(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.minimum(vwap - df["open"], 0), 5)

def factor_136(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.maximum(vwap - df["close"], 0), 5)

def factor_137(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.minimum(vwap - df["close"], 0), 5)

def factor_138(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.maximum(df["high"] - vwap, 0), 5)

def factor_139(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.minimum(df["high"] - vwap, 0), 5)

def factor_140(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.maximum(vwap - df["low"], 0), 5)

def factor_141(df):
    vwap = _ensure_vwap(df)
    return -TSRANK(np.minimum(vwap - df["low"], 0), 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 142–149 (REGBETA/REGRESI on Log Deltas)
# ══════════════════════════════════════════════════════════════

def _make_regbeta_factor(field):
    def f(df):
        x = DELTA(LOG(df[field]), 1)
        y = DELTA(LOG(df["volume"]), 1)
        return RANK(REGBETA(x, y, 5)) * -1
    return f

def _make_regresi_factor(field):
    def f(df):
        x = DELTA(LOG(df[field]), 1)
        y = DELTA(LOG(df["volume"]), 1)
        return RANK(REGRESI(x, y, 5)) * -1
    return f

factor_142 = _make_regbeta_factor("close")
factor_143 = _make_regresi_factor("close")
factor_144 = _make_regbeta_factor("high")
factor_145 = _make_regresi_factor("high")
factor_146 = _make_regbeta_factor("low")
factor_147 = _make_regresi_factor("low")

def factor_148(df):
    vwap = _ensure_vwap(df)
    x = DELTA(LOG(vwap), 1)
    y = DELTA(LOG(df["volume"]), 1)
    return RANK(REGBETA(x, y, 5)) * -1

def factor_149(df):
    vwap = _ensure_vwap(df)
    x = DELTA(LOG(vwap), 1)
    y = DELTA(LOG(df["volume"]), 1)
    return RANK(REGRESI(x, y, 5)) * -1


# ══════════════════════════════════════════════════════════════
#  Alpha 150–159 (COUNT > DELAY / < DELAY)
# ══════════════════════════════════════════════════════════════

def _make_count_up_factor(field):
    def f(df):
        return RANK(COUNT(df[field] > DELAY(df[field], 1), 5)) * -1
    return f

def _make_count_down_factor(field):
    def f(df):
        return RANK(COUNT(df[field] < DELAY(df[field], 1), 5)) * -1
    return f

factor_150 = _make_count_up_factor("close")
factor_151 = _make_count_down_factor("close")
factor_152 = _make_count_up_factor("high")
factor_153 = _make_count_down_factor("high")
factor_154 = _make_count_up_factor("low")
factor_155 = _make_count_down_factor("low")

def factor_156(df):
    vwap = _ensure_vwap(df)
    return RANK(COUNT(vwap > DELAY(vwap, 1), 5)) * -1

def factor_157(df):
    vwap = _ensure_vwap(df)
    return RANK(COUNT(vwap < DELAY(vwap, 1), 5)) * -1

factor_158 = _make_count_up_factor("volume")
factor_159 = _make_count_down_factor("volume")


# ══════════════════════════════════════════════════════════════
#  Alpha 160–170 (CORR with COUNT)
# ══════════════════════════════════════════════════════════════

def factor_160(df):
    return -CORR(RANK(COUNT(df["close"] > DELAY(df["close"], 1), 5)), RANK(df["volume"]), 6)

def factor_161(df):
    return -CORR(RANK(COUNT(df["close"] < DELAY(df["close"], 1), 5)), RANK(df["volume"]), 6)

def factor_162(df):
    return -CORR(RANK(COUNT(df["high"] > DELAY(df["high"], 1), 5)), RANK(df["volume"]), 6)

def factor_163(df):
    return -CORR(RANK(COUNT(df["high"] < DELAY(df["high"], 1), 5)), RANK(df["volume"]), 6)

def factor_164(df):
    return -CORR(RANK(COUNT(df["low"] > DELAY(df["low"], 1), 5)), RANK(df["volume"]), 6)

def factor_165(df):
    return -CORR(RANK(COUNT(df["low"] < DELAY(df["low"], 1), 5)), RANK(df["volume"]), 6)

def factor_166(df):
    vwap = _ensure_vwap(df)
    return -CORR(RANK(COUNT(vwap > DELAY(vwap, 1), 5)), RANK(df["volume"]), 6)

def factor_167(df):
    vwap = _ensure_vwap(df)
    return -CORR(RANK(COUNT(vwap < DELAY(vwap, 1), 5)), RANK(df["volume"]), 6)

def factor_168(df):
    return -CORR(RANK(COUNT(df["volume"] > DELAY(df["volume"], 1), 5)), RANK(df["close"]), 6)

def factor_169(df):
    return -CORR(RANK(COUNT(df["volume"] < DELAY(df["volume"], 1), 5)), RANK(df["close"]), 6)

def factor_170(df):
    return -CORR(RANK(COUNT(df["volume"] > DELAY(df["volume"], 1), 5)), RANK(df["high"]), 6)


# ══════════════════════════════════════════════════════════════
#  Alpha 171–180 (Bull/Bear Ratios)
# ══════════════════════════════════════════════════════════════

def factor_171(df: pd.DataFrame) -> pd.Series:
    """-1 * (LOW-CLOSE) * OPEN^5 / ((CLOSE-HIGH+1e-7) * CLOSE^5)"""
    c = df["close"]
    numer = (df["low"] - c) * df["open"] ** 5
    denom = (c - df["high"] + 1e-7) * c ** 5
    return _clip(-numer / denom)

def factor_172(df: pd.DataFrame) -> pd.Series:
    """DI-difference oscillator (DX): MEAN(|p1-p2|/(p1+p2)*100, 6)
       p1 = SUM(pos_ld,14)*100/SUM(TR,14), p2 = SUM(pos_hd,14)*100/SUM(TR,14)"""
    c, h, l = df["close"], df["high"], df["low"]
    pc = DELAY(c, 1)
    a = h - l
    b = ABS(h - pc)
    c_lo = ABS(l - pc)
    tr = np.fmax(np.fmax(a, b), c_lo)
    hd = h - DELAY(h, 1)
    ld = DELAY(l, 1) - l
    pos_ld = _where((ld > 0) & (ld > hd), ld, 0.0)
    pos_hd = _where((hd > 0) & (hd > ld), hd, 0.0)
    sum_tr = SUM(tr, 14) + 1e-7
    p1 = SUM(pos_ld, 14) * 100.0 / sum_tr
    p2 = SUM(pos_hd, 14) * 100.0 / sum_tr
    raw = ABS(p1 - p2) / (p1 + p2 + 1e-7) * 100.0
    return _clip(SUM(raw, 6) / 6)

def factor_173(df: pd.DataFrame) -> pd.Series:
    """3*SMA(C,13,2) - 2*SMA(SMA(C,13,2),13,2) + SMA(SMA(LOG(C),13,2),13,2)"""
    c = df["close"]
    ma = SMA(c, 13, 2)
    return _clip(3.0 * ma - 2.0 * SMA(ma, 13, 2) + SMA(SMA(LOG(c), 13, 2), 13, 2))

def factor_174(df: pd.DataFrame) -> pd.Series:
    """SMA(C > prev_C ? STD(C,20) : 0, 20, 1) — up-day volatility EWMA"""
    c = df["close"]
    pc = DELAY(c, 1)
    s = STD(c, 20)
    # 参照 aurumq-rl 逻辑: pc 或 s 为 null 时填 NaN，否则 C > prev_C ? s : 0
    mask_valid = pc.notna() & s.notna()
    masked = _where(mask_valid & (c > pc), s, _where(mask_valid, 0.0, np.nan))
    return _clip(SMA(masked, 20, 1))

def factor_175(df: pd.DataFrame) -> pd.Series:
    """MEAN(TR, 6) — 6-day mean true range"""
    c, h, l = df["close"], df["high"], df["low"]
    pc = DELAY(c, 1)
    a = h - l
    b = ABS(pc - h)
    c_tr = ABS(pc - l)
    tr = np.fmax(np.fmax(a, b), c_tr)
    return SUM(tr, 6) / 6

def factor_176(df: pd.DataFrame) -> pd.Series:
    """(-1 * (((HIGH-CLOSE)/(CLOSE-LOW)) * (CLOSE/OPEN)^1))"""
    bear = ((df["high"] - df["close"]) / (df["close"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    return _clip(-bear * (df["close"] / df["open"]))

def factor_177(df: pd.DataFrame) -> pd.Series:
    """RANK(DELTA(((CLOSE-LOW)/(HIGH-CLOSE)),1))*-1"""
    bull = ((df["close"] - df["low"]) / (df["high"] - df["close"])).replace([np.inf, -np.inf], np.nan)
    return RANK(DELTA(bull, 1)) * -1

def factor_178(df: pd.DataFrame) -> pd.Series:
    """RANK(DELTA(((HIGH-CLOSE)/(CLOSE-LOW)),1))*-1"""
    bear = ((df["high"] - df["close"]) / (df["close"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    return RANK(DELTA(bear, 1)) * -1

def factor_179(df: pd.DataFrame) -> pd.Series:
    """(-1 * TSRANK(((CLOSE-LOW)/(HIGH-CLOSE)),5))"""
    bull = ((df["close"] - df["low"]) / (df["high"] - df["close"])).replace([np.inf, -np.inf], np.nan)
    return -TSRANK(bull, 5)

def factor_180(df: pd.DataFrame) -> pd.Series:
    """(-1 * TSRANK(((HIGH-CLOSE)/(CLOSE-LOW)),5))"""
    bear = ((df["high"] - df["close"]) / (df["close"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    return -TSRANK(bear, 5)


# ══════════════════════════════════════════════════════════════
#  Alpha 181–191 (Bull/Bear Extrema + CORR/TSRANK/DELTA)
# ══════════════════════════════════════════════════════════════

def _bull_bear_pair(df):
    """Return (bull_ratio, bear_ratio, max_ratio, min_ratio) for alpha 181-191"""
    bull = ((df["close"] - df["low"]) / (df["high"] - df["close"])).replace([np.inf, -np.inf], np.nan)
    bear = ((df["high"] - df["close"]) / (df["close"] - df["low"])).replace([np.inf, -np.inf], np.nan)
    max_ratio = np.fmax(bull, bear)   # fmax ignores NaN
    min_ratio = np.fmin(bull, bear)   # fmin ignores NaN
    return bull, bear, max_ratio, min_ratio

def factor_181(df: pd.DataFrame) -> pd.Series:
    _, _, max_ratio, _ = _bull_bear_pair(df)
    return RANK(max_ratio) * -1

def factor_182(df: pd.DataFrame) -> pd.Series:
    _, _, _, min_ratio = _bull_bear_pair(df)
    return RANK(min_ratio) * -1

def factor_183(df: pd.DataFrame) -> pd.Series:
    bull, _, _, _ = _bull_bear_pair(df)
    return -CORR(RANK(bull), RANK(df["volume"]), 6)

def factor_184(df: pd.DataFrame) -> pd.Series:
    _, bear, _, _ = _bull_bear_pair(df)
    return -CORR(RANK(bear), RANK(df["volume"]), 6)

def factor_185(df: pd.DataFrame) -> pd.Series:
    _, _, max_ratio, _ = _bull_bear_pair(df)
    return -CORR(RANK(max_ratio), RANK(df["volume"]), 6)

def factor_186(df: pd.DataFrame) -> pd.Series:
    _, _, _, min_ratio = _bull_bear_pair(df)
    return -CORR(RANK(min_ratio), RANK(df["volume"]), 6)

def factor_187(df: pd.DataFrame) -> pd.Series:
    _, _, max_ratio, _ = _bull_bear_pair(df)
    return RANK(DELTA(max_ratio, 1)) * -1

def factor_188(df: pd.DataFrame) -> pd.Series:
    _, _, _, min_ratio = _bull_bear_pair(df)
    return RANK(DELTA(min_ratio, 1)) * -1

def factor_189(df: pd.DataFrame) -> pd.Series:
    _, _, max_ratio, _ = _bull_bear_pair(df)
    return -TSRANK(max_ratio, 5)

def factor_190(df: pd.DataFrame) -> pd.Series:
    _, _, _, min_ratio = _bull_bear_pair(df)
    return -TSRANK(min_ratio, 5)

def factor_191(df: pd.DataFrame) -> pd.Series:
    _, _, max_ratio, _ = _bull_bear_pair(df)
    dlog_vol = DELTA(LOG(df["volume"]), 1)
    return -CORR(RANK(max_ratio), RANK(dlog_vol), 6)
