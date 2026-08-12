"""
industry.py — 行业分类拉取（baostock query_stock_industry）。

P1 多头增强需要行业中性化，但全仓库无任何行业映射。本模块拉 baostock 行业分类，
缓存 data/cache_meta/industry.csv（symbol → 申万一级行业）。

⚠️ 局限（报告必须标注）:
  baostock 只返回股票**当前**行业归属快照，非历史 PIT。回看历史用今天的行业归属
  有轻微 look-ahead。业界增强回测常用此近似，可接受，但需明示。
  备选 PIT 行业源（akshare 申万历史行业分类）本期不做。

用法:
    python data/industry.py                          # 拉全市场 + 筛 zz500 成员 + 缓存
    python -c "from data.industry import load_industry; print(load_industry().head())"
"""

from pathlib import Path

import pandas as pd
import baostock as bs

CACHE_DIR = Path(__file__).parent / "cache_meta"
CACHE_PATH = CACHE_DIR / "industry.csv"

_logged_in = False


def _ensure_login():
    global _logged_in
    if _logged_in:
        return
    bs.login()
    _logged_in = True


def _zz500_symbols() -> set[str]:
    """从 zz500_membership.csv 读全部历史成员 symbol 集合。"""
    mpath = CACHE_DIR / "zz500_membership.csv"
    if not mpath.exists():
        return set()
    df = pd.read_csv(mpath, index_col=0)
    return set(df.columns)


def fetch_industry() -> pd.DataFrame:
    """拉全市场行业分类，返回 DataFrame(symbol, industry) 申万一级。"""
    _ensure_login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        rows.append(row)

    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        raise RuntimeError("query_stock_industry 返回空")

    # baostock 只提供「证监会行业分类」，industry 形如 'J66货币金融服务' / 'C36汽车制造业'。
    # 取门类字母（A-S，共 19 个）作为行业桶 —— 粒度与申万一级（28 个）相当。
    df["symbol"] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False)
    df = df[df["industry"].str.strip().ne("")]
    df["industry"] = df["industry"].str[0]
    df = df[["symbol", "industry"]].drop_duplicates(subset=["symbol"], keep="last")
    return df


def build_and_cache() -> pd.DataFrame:
    """拉全市场行业，筛 zz500 成员，缓存到 cache_meta/industry.csv。"""
    df = fetch_industry()
    zz500 = _zz500_symbols()
    df_zz = df[df["symbol"].isin(zz500)].copy()
    df_zz = df_zz.sort_values("symbol")
    df_zz["industry"] = df_zz["industry"].astype(str)

    df_zz.to_csv(CACHE_PATH, index=False)
    print(f"行业缓存: {len(df_zz)} 只 zz500 成员 / {df['symbol'].nunique()} 只全市场")
    print("行业分布:")
    print(df_zz["industry"].value_counts().head(15))
    return df_zz


def load_industry() -> pd.Series | None:
    """读取缓存，返回 Series(index=symbol, value=行业门类)。未缓存返回 None。

    symbol 必须读为字符串（dtype=str 保留前导零，如 '000006'），否则 pandas
    会推断为 int → 与 pred_matrix 的字符串列无法对齐。
    """
    if not CACHE_PATH.exists():
        return None
    df = pd.read_csv(CACHE_PATH, dtype={"symbol": str})
    return df.set_index("symbol")["industry"]


if __name__ == "__main__":
    build_and_cache()
