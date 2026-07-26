"""
index_membership.py — CSI 300 历史成分股 PIT 成员矩阵。

解决幸存者偏差的根本方案：不用"当前成分股回看历史"，而是从 baostock
query_hs300_stocks(date=...) 逐月查询历史时点的真实成分股名单
（已验证含后来退市的股票，如武钢股份 600005），构建 date × symbol
的布尔成员矩阵。

方法与假设:
  - 按月末采样查询（CSI 300 每年 6/12 月定期调整 + 临时调整，
    月频采样最多滞后一个月捕捉临时调整，定期调整不会漏）
  - 查询结果前向填充到日频：某月名单在下一次查询前保持不变
  - baostock 返回的是"查询日 <= date 的最近一次名单快照"，
    天然 PIT，无未来函数
  - 缓存到 data/cache_meta/hs300_membership.csv，不重复查询

用法:
    from data.index_membership import load_membership
    member = load_membership()          # DataFrame[bool], index=月末日期
    daily = expand_to_daily(member, trading_dates)  # 日频前向填充
"""

from pathlib import Path

import pandas as pd

META_DIR = Path(__file__).parent / "cache_meta"
META_DIR.mkdir(parents=True, exist_ok=True)
MEMBERSHIP_PATH = META_DIR / "hs300_membership.csv"


def fetch_membership(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """逐月查询 baostock 历史成分股，返回月末采样的成员矩阵。

    Returns
        DataFrame[bool]: index=查询采样日（月末）, columns=symbols，
        True=该日该股在 CSI 300 内
    """
    import baostock as bs

    bs.login()
    month_ends = pd.date_range(start, end, freq="ME")
    records: dict[pd.Timestamp, set[str]] = {}
    for d in month_ends:
        ds = d.strftime("%Y-%m-%d")
        rs = bs.query_hs300_stocks(date=ds)
        members: set[str] = set()
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            members.add(row[1].replace("sh.", "").replace("sz.", ""))
        if members:
            records[d] = members
        else:
            print(f"[WARN] {ds} 成分股查询为空，跳过")
    bs.logout()

    all_symbols = sorted(set().union(*records.values()))
    matrix = pd.DataFrame(False, index=sorted(records), columns=all_symbols)
    for d, members in records.items():
        matrix.loc[d, sorted(members)] = True
    print(f"成员矩阵: {len(matrix)} 个月度快照 × {len(all_symbols)} 只历史成员")
    return matrix


def load_membership(refresh: bool = False) -> pd.DataFrame:
    """读取缓存的成员矩阵，无缓存（或 refresh=True）时从 baostock 拉取。"""
    if MEMBERSHIP_PATH.exists() and not refresh:
        df = pd.read_csv(MEMBERSHIP_PATH, index_col=0, parse_dates=True)
        df.columns = [str(c).zfill(6) for c in df.columns]
        return df.astype(bool)
    matrix = fetch_membership()
    matrix.to_csv(MEMBERSHIP_PATH)
    return matrix


def expand_to_daily(
    membership: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """月末快照前向填充到交易日频。

    每个交易日使用"最近一次月末快照"的名单——快照日期 <= 交易日，
    保证 PIT（不用未来的名单）。首个快照之前的日期沿用首个快照
    （2010 年初的近似，误差仅头一个月）。
    """
    daily = membership.reindex(
        membership.index.union(trading_dates)
    ).ffill().bfill()
    return daily.loc[trading_dates].astype(bool)
