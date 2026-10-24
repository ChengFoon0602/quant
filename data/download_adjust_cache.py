"""批量补齐后复权 / 不复权日线缓存。

背景（2026-10-24 alpha191 修复 P0）：
  截面因子必须用后复权（CLAUDE.md 铁律 3.5）。原 `compute_factor_matrix`
  读的是前复权缓存，且 `vwap = amount / volume` 用的是未复权成交额，
  与前复权 close 量纲不一致（实测 vwap/close 中位 1.31），导致 15 个因子退化。

需要两套数据：
  adj="3" 不复权 — 全字段，作为还原基准
  adj="1" 后复权 — 因子计算口径

性能设计（实测 baostock，2026-10-24）：
  - adjustflag="3" 全 7 字段：约 0.9s
  - adjustflag="1" 全 7 字段：约 26s（服务端逐字段现算复权）
  - adjustflag="1" 只要 close+volume：约 5.5s
  - 多进程并发会被服务端限流，实测 3 进程反而比串行慢 18 倍 → 必须串行

因此 adj1 只查 close+volume 两个字段，OHLC 由 adj3 乘 f(t) 合成：
    f(t)     = close_adj1 / close_adj3        （同一天 OHLC 共用一个价格复权因子）
    OHLC_adj1 = OHLC_adj3 × f(t)
    volume_adj1 = 直接取 adj1 查询值（送转股影响股数，不能用 f(t) 推）
    amount_adj1 = adj3 的 amount（成交额不受复权影响）
  790 只约 1.4 小时；串行下不会触发服务端限流。

缓存落盘规则见 `data/fetcher.py::_cache_path`：
  adj="1" → {symbol}_adj1.csv
  adj="3" → {symbol}_adj3.csv

用法:
    python data/download_adjust_cache.py --index hs300
    python data/download_adjust_cache.py --index hs300 --limit 20   # 试跑
    python data/download_adjust_cache.py --index zz500
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import baostock as bs

from data.fetcher import (
    _add_prefix,
    _assert_daily_frame,
    _cache_path,
    _ensure_login,
    download_daily,
    load_daily,
)

START, END = "2010-01-01", "2025-12-31"

# 连续失败达到该次数即判定「服务端限流/断连」，重连 + 退避。
CONSEC_FAIL_RELOGIN = 15


def _reset_login() -> None:
    """强制下次查询重新 login（fetcher 的 _logged_in 是模块级幂等标志）。"""
    import data.fetcher as fetcher

    fetcher._logged_in = False


def _covered(symbol: str, flag: str) -> bool:
    """缓存是否已覆盖 [START, END] 全区间。

    留 15 天容差：START=2010-01-01 是元旦，首个交易日是 01-04，
    严格比较会让所有缓存永远判定「未覆盖」→ 每次全量重下。
    """
    p = _cache_path(symbol, adjust=flag)
    if not p.exists():
        return False
    try:
        idx = pd.read_csv(p, usecols=["date"], parse_dates=["date"])["date"]
    except Exception:
        return False
    if len(idx) == 0:
        return False
    tol = pd.Timedelta(days=15)
    return idx.min() <= pd.Timestamp(START) + tol and idx.max() >= pd.Timestamp(END) - tol


def _query_adj1_light(symbol: str) -> pd.DataFrame:
    """只取后复权的 close 与 volume（避开 26s 的全字段复权计算）。"""
    _ensure_login()
    rs = bs.query_history_k_data_plus(
        _add_prefix(symbol),
        "date,close,volume",
        start_date=START,
        end_date=END,
        frequency="d",
        adjustflag="1",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        # 与 fetcher 同口径：服务端错误 ≠ 该股无数据
        raise RuntimeError(f"{symbol} adj1 baostock 错误 code={rs.error_code} msg={rs.error_msg}")
    if not rows:
        raise RuntimeError(f"{symbol} adj1 返回空数据")
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def _build_adj1(symbol: str, a3: pd.DataFrame) -> pd.DataFrame:
    """用 adj1 的 close/volume + adj3 的 OHLC/amount 合成完整后复权帧。"""
    light = _query_adj1_light(symbol)
    common = a3.index.intersection(light.index)
    if len(common) < 100:
        raise RuntimeError(f"{symbol} adj1/adj3 交集不足 100 行")

    a3c = a3.loc[common]
    lt = light.loc[common]
    f_price = lt["close"] / a3c["close"].replace(0, np.nan)

    out = pd.DataFrame({
        "open": a3c["open"] * f_price,
        "high": a3c["high"] * f_price,
        "low": a3c["low"] * f_price,
        "close": lt["close"],
        "volume": lt["volume"],
        "amount": a3c["amount"],          # 成交额不受复权影响
    }, index=common)

    # 停牌日 close_adj3 = 0 → f(t) 无效。剔除这些日期而不是让整只股票失败。
    bad = f_price.isna() | (f_price <= 0)
    if bad.any():
        out = out.loc[~bad]

    # 数值自检：缩放关系必须保序
    ok = (out["high"] >= out["low"] - 1e-6).all() and (f_price > 0).all()
    if not ok:
        raise RuntimeError(f"{symbol} adj1 合成后 OHLC 关系异常")
    return out


def _sync_one(symbol: str) -> str:
    """确保某只股票的 adj3 + adj1 缓存齐备。返回 status。"""
    need3 = not _covered(symbol, "3")
    need1 = not _covered(symbol, "1")
    if not need3 and not need1:
        return "skip"

    try:
        if need3:
            a3 = download_daily(symbol, start=START, end=END, adjust="3")
        else:
            a3 = load_daily(symbol, adjust="3")
        if a3 is None or len(a3) < 100:
            return "fail:adj3 数据不足"

        if need1:
            out = _build_adj1(symbol, a3)
            _assert_daily_frame(out, symbol)
            out.to_csv(_cache_path(symbol, adjust="1"))
        return "done"
    except Exception as e:
        return f"fail:{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="hs300", choices=["hs300", "zz500"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（试跑）")
    ap.add_argument("--retry-file", default="",
                    help="只重试该清单内的代码（每行 `代码` 或 `代码<TAB>原因`）")
    args = ap.parse_args()

    from data.index_membership import load_membership

    membership = load_membership(args.index)
    symbols = sorted(membership.columns)
    if args.retry_file:
        retry = {ln.split("\t")[0].strip()
                 for ln in Path(args.retry_file).read_text(encoding="utf-8").splitlines()
                 if ln.strip()}
        symbols = [s for s in symbols if s in retry]
        print(f"[retry] 清单 {len(retry)} 只，与 {args.index} 成员交集 {len(symbols)} 只", flush=True)
    if args.limit:
        symbols = symbols[: args.limit]

    # baostock 的查询不带超时，网络卡住会永久阻塞。全局 socket 超时兜底。
    socket.setdefaulttimeout(60)

    print(f"[{args.index}] {len(symbols)} 只，串行补齐 adj3 + adj1", flush=True)

    done = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    consec_fail = 0
    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        status = _sync_one(sym)
        if status == "done":
            done += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            failures.append((sym, status[5:]))

        # 连续失败说明服务端已限流/断连（不是这些股票真没数据）——重连并退避。
        if status.startswith("fail"):
            consec_fail += 1
            if consec_fail >= CONSEC_FAIL_RELOGIN:
                print(f"  ⚠ 连续 {consec_fail} 次失败 → 重连 baostock 并退避 60s", flush=True)
                try:
                    bs.logout()
                except Exception:
                    pass
                _reset_login()
                _ensure_login()
                time.sleep(60)
                consec_fail = 0
        else:
            consec_fail = 0

        if status != "skip" or i % 20 == 0 or i == len(symbols):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(symbols) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i}/{len(symbols)}] {sym}: {status[:40]}  新建={done} 跳过={skipped} "
                  f"失败={failed}  {rate:.2f} 只/秒  剩余 {eta:.0f} 分", flush=True)

    print(f"\n完成: 新建 {done} / 跳过 {skipped} / 失败 {failed}  "
          f"耗时 {(time.time()-t0)/60:.1f} 分", flush=True)
    if failures:
        print("失败明细（前 20 条）:")
        for sym, msg in failures[:20]:
            print(f"  {sym}: {msg}")
        log = Path(__file__).parent / f"download_failures_{args.index}.txt"
        log.write_text("\n".join(f"{s}\t{m}" for s, m in failures), encoding="utf-8")
        print(f"完整明细: {log}")
    bs.logout()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
