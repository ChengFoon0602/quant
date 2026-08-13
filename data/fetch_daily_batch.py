"""
fetch_daily_batch.py — baostock 跨天分批拉取调度器（按天配额安全）。

背景（2026-08-13 实测）：baostock 对单 IP 有【按天累计查询配额】——
即使单进程 2.9 q/s 串行，累计 ~5 万次查询后仍触发 error_code 10001011。
全量需求 6 财报接口 × 1625 股 × 76 季 ≈ 74 万次 → 必须跨天分批：
每批 ~500 股 ≈ 3.8 万次（留 ~25% 配额余量），每天只拉 1 批。

批次 = (接口, 股票段)。股票池按固定顺序切段（500/500/500/125），
6 接口 × 4 段 = 24 批。批内 checkpoint 每 1 万条增量写盘（fetch_interface），
批间状态 JSON（fetch_batch_state.json）断点续传，crash 只重跑当前批。

IPO 裁剪：每只股票用 query_stock_basic 的 IPO 年份只查上市后的季度
（省 ~30-40% 查询配额；2015+ 上市的创业板/科创板省最多）。IPO 映射
一次拉取缓存 data/cache_meta/ipo_years.json 永久复用。

配额安全哨兵：批结束后校验命中量。若整批命中异常低（< 50 条/股），
判定「可能再次触发黑名单」，不标记 done，等待下一冷却周期。

用法:
    python data/fetch_daily_batch.py              # 探测解封 → 拉第一未完成批
    python data/fetch_daily_batch.py --status     # 查看批次进度/剩余天数
    python data/fetch_daily_batch.py --batch N    # 强制拉第 N 批（补拉/调试）
    python data/fetch_daily_batch.py --force      # 跳过解封探测（危险，仅调试）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baostock as bs
from data.fundamental_fetcher import (
    INTERFACE_GROUPS, get_zz500_pit_symbols, get_ipo_year_map, fetch_interface,
)

STATE_PATH = Path(__file__).parent / "fetch_batch_state.json"
BATCH_SIZE = 500          # 每批股票数 → 500×76 ≈ 38,000 查询（安全配额内）
QPS = 3.0
YEARS = range(2007, 2026)
MIN_HIT_RATIO = 0.12      # 配额安全哨兵：命中/查询比低于此 → 疑似黑名单
ERR_BLACKLIST = ("10001011", "黑名单")


def _build_batches() -> list[dict]:
    """构建全部批次（接口 × 股票段），股票池固定顺序（sorted）。"""
    symbols = get_zz500_pit_symbols()
    n = len(symbols)
    batches = []
    for iface in INTERFACE_GROUPS:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batches.append({
                "idx": len(batches), "interface": iface,
                "start": start, "end": end, "done": False, "done_at": None,
            })
    return batches


def _load_state() -> list[dict]:
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)["batches"]
    return _build_batches()


def _save_state(batches: list[dict]):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "batches": batches},
                  f, ensure_ascii=False, indent=1)


def unblocked() -> bool:
    """baostock 是否解封（login 探测，不算查询）。"""
    lg = bs.login()
    ok = lg.error_code == "0"
    if ok:
        bs.logout()
    return ok


def _run_batch(batch: dict, qps: float = QPS) -> bool:
    """拉取一个批次。返回 True = 拉取完成且命中量正常（标记 done）。"""
    symbols = get_zz500_pit_symbols()
    sym_sub = symbols[batch["start"]:batch["end"]]
    api, fields = INTERFACE_GROUPS[batch["interface"]]

    # IPO 裁剪：每只股票只查上市后的季度（省 ~30-40% 查询配额）。
    # IPO 映射一次拉取缓存复用；封禁中返回空 dict → years_by_symbol 空 → 退化为全范围。
    ipo = get_ipo_year_map(symbols)
    years_by_symbol = {}
    for s in sym_sub:
        y0 = ipo.get(s)
        if y0 is not None:
            years_by_symbol[s] = range(max(2007, y0), 2026)
    n_queries = sum(4 * len(years_by_symbol.get(s, YEARS)) for s in sym_sub)
    print(f"\n=== batch {batch['idx']}: {batch['interface']} 股票 "
          f"{batch['start']}:{batch['end']}（{len(sym_sub)} 只，"
          f"IPO 裁剪后 {n_queries:,} 查询 @ {qps} q/s）===")
    result = fetch_interface(sym_sub, api, fields, qps=qps, years_by_symbol=years_by_symbol)

    # 配额安全哨兵：命中/查询比低于 12% → 疑似二次封禁，不标记 done
    total_hits = sum(df.notna().sum().sum() for df in result.values())
    min_hits = int(n_queries * MIN_HIT_RATIO)
    print(f"  batch {batch['idx']} 命中 {total_hits:,} 条（哨兵阈值 {min_hits:,}）")
    if total_hits < min_hits:
        print("  [WARN] 命中量异常低，疑似触发黑名单。该批不标记完成，等待冷却后重试。")
        return False
    return True


def print_status(batches: list[dict]):
    done = sum(1 for b in batches if b["done"])
    remaining = [b for b in batches if not b["done"]]
    print(f"批次进度: {done}/{len(batches)} 完成 | 剩余 {len(remaining)} 批 ≈ {len(remaining)} 天")
    for b in batches:
        mark = "✓" if b["done"] else ("·" if not b["done"] else "")
        print(f"  [{mark}] {b['idx']:2d} {b['interface']:<12} 股票 {b['start']:4d}:{b['end']:4d} "
              f"({b['end']-b['start']} 只)" + (f"  @ {b['done_at']}" if b["done_at"] else ""))


def main():
    qps = QPS
    arg_batch, force = None, False
    for i, a in enumerate(sys.argv):
        if a == "--qps" and i + 1 < len(sys.argv):
            qps = float(sys.argv[i + 1])
        if a == "--batch" and i + 1 < len(sys.argv):
            arg_batch = int(sys.argv[i + 1])
        if a == "--force":
            force = True
    if "--status" in sys.argv:
        print_status(_load_state())
        return

    batches = _load_state()
    if arg_batch is not None:
        if not (0 <= arg_batch < len(batches)):
            print(f"[ERROR] 批次号越界（0-{len(batches)-1}）")
            sys.exit(1)
        batch = batches[arg_batch]
        print(f"强制拉批次 {arg_batch}: {batch['interface']} 股票 {batch['start']}:{batch['end']}")
        ok = _run_batch(batch, qps)
        if ok:
            batch["done"] = True
            batch["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(batches)
        return

    # 正常路径：探测解封 → 拉第一个未完成批 → 只拉 1 批（配额）
    if not force:
        if not unblocked():
            print("baostock 仍封禁中（blacklist）。等待冷却，明天再试。")
            return
        print("baostock 已解封。")

    pending = [b for b in batches if not b["done"]]
    if not pending:
        print("全部批次已完成。可以跑 strategies/zz500_fundamental_trial/report.py 全链了。")
        return
    batch = pending[0]
    print(f"今天拉批次 {batch['idx']}: {batch['interface']} 股票 {batch['start']}:{batch['end']}")
    ok = _run_batch(batch, qps)
    if ok:
        batch["done"] = True
        batch["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_state(batches)
        print(f"\n批次 {batch['idx']} 完成。明天继续（剩余 {sum(1 for b in batches if not b['done'])} 批）。")


if __name__ == "__main__":
    main()
