"""补拉中证 500 历史成员中本地缺失的日线数据（PIT 路线②）。

从 zz500 成员矩阵取历史成员全集，与 data/cache 现有缓存对比，
只下载缺失部分，逐只 download_daily（自动缓存、重试、增量合并）。
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.index_membership import load_membership
from data.fetcher import download_daily, _ensure_login
import baostock as bs


def main():
    m = load_membership("zz500")
    universe = set(m.columns)
    cached = {p.stem for p in pathlib.Path("data/cache").glob("*.csv")}
    missing = sorted(universe - cached)
    print(f"历史成员全集 {len(universe)}  已缓存 {len(cached & universe)}  待补 {len(missing)}")

    _ensure_login()
    ok, fail = [], []
    for i, sym in enumerate(missing):
        try:
            df = download_daily(sym, start="2010-01-01", end="2025-12-31")
            ok.append(sym)
            if (i + 1) % 25 == 0 or i == 0:
                print(f"[{i+1}/{len(missing)}] {sym} OK {len(df)} 条")
        except Exception as e:
            fail.append(sym)
            print(f"[{i+1}/{len(missing)}] {sym} FAIL {e}")
        time.sleep(0.15)
    bs.logout()
    print(f"\n完成: {len(ok)} 成功, {len(fail)} 失败")
    if fail:
        print("失败列表:", fail)


if __name__ == "__main__":
    main()
