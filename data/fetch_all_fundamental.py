"""
fetch_all_fundamental.py — 方向2 数据顺序拉取调度（解封后一键跑）。

baostock 风控教训：6 进程并行 36 q/s 触发「黑名单」封禁（error_code 10001011）。
本脚本强制安全路径：
  - 串行：一个接口完整跑完才下一个（绝不允许并行）
  - qps 限流：每查询最小间隔，默认 3 q/s（--qps 可覆盖，更保守用 2）
  - 接口级断点：已完整缓存的接口自动跳过 → crash 后重跑只补缺失接口
  - 完整性校验：每接口所有字段缓存存在且覆盖足够（列数/非空数）才算完成

用法:
    python data/fetch_all_fundamental.py                  # 串行 6 财报接口 + 4 估值
    python data/fetch_all_fundamental.py --qps 2          # 更保守限流
    python data/fetch_all_fundamental.py --only fund      # 只财报（估值另跑）
    python data/fetch_all_fundamental.py --only valuation # 只估值
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fundamental_fetcher import (
    INTERFACE_GROUPS, get_zz500_pit_symbols, fetch_interface, load_cached_field,
)
from data.valuation_fetcher import (
    fetch_valuation, load_cached_valuation, VALUATION_FIELDS,
)

# 完整性阈值（保守：避免把部分拉取的接口误判为"已完成"）
MIN_COLS = 500        # 缓存至少覆盖 500 只股票
MIN_VALUES = 20000    # 非空值至少 20000（全量预计 ~80000/字段）


def interface_done(name: str) -> bool:
    """接口是否已完整缓存（全部字段达到阈值）。"""
    _, fields = INTERFACE_GROUPS[name]
    for f in fields:
        df = load_cached_field(f)
        if df is None or len(df.columns) < MIN_COLS or df.notna().sum().sum() < MIN_VALUES:
            return False
    return True


def valuation_done() -> bool:
    for f in VALUATION_FIELDS:
        df = load_cached_valuation(f)
        if df is None or len(df.columns) < MIN_COLS:
            return False
    return True


def fetch_all_fundamental(qps: float):
    """串行拉取 6 个财报接口（接口级断点）。"""
    symbols = get_zz500_pit_symbols()
    print(f"PIT zz500 股票池: {len(symbols)} 只 | qps={qps}（串行，严禁并行）")
    for name in INTERFACE_GROUPS:
        if interface_done(name):
            print(f"[跳过] {name} 已完整缓存")
            continue
        api, fields = INTERFACE_GROUPS[name]
        print(f"\n=== {name} 接口: {fields} ===")
        t0 = time.time()
        fetch_interface(symbols, api, fields, qps=qps)
        ok = interface_done(name)
        print(f"  {name} 完成: {'✓ 校验通过' if ok else '⚠ 未达完整阈值（需重跑）'} | "
              f"耗时 {(time.time()-t0)/3600:.1f}h")
    print("\n财报 6 接口全部完成。")


def fetch_all_valuation(qps: float):
    """拉取估值 4 字段。"""
    if valuation_done():
        print("[跳过] 估值已完整缓存")
        return
    print(f"\n=== 估值 {VALUATION_FIELDS} ===")
    symbols = get_zz500_pit_symbols()
    t0 = time.time()
    fetch_valuation(symbols, qps=qps)
    ok = valuation_done()
    print(f"估值完成: {'✓ 校验通过' if ok else '⚠ 未达完整阈值'} | "
          f"耗时 {(time.time()-t0)/60:.0f}min")


if __name__ == "__main__":
    qps = 3.0
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--qps" and i + 1 < len(sys.argv):
            qps = float(sys.argv[i + 1])
        if a == "--only" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]

    print("=" * 72)
    print("方向2 数据顺序拉取（baostock 安全模式：串行 + 限流 + 断点）")
    print("=" * 72)

    if only is None or only == "fund":
        fetch_all_fundamental(qps)
    if only is None or only == "valuation":
        fetch_all_valuation(qps)

    print("\n全部完成。校验汇总：")
    for name in INTERFACE_GROUPS:
        print(f"  {name:<12} {'✓' if interface_done(name) else '✗ 未完整'}")
    for f in VALUATION_FIELDS:
        df = load_cached_valuation(f)
        print(f"  {f:<12} {'✓' if df is not None and len(df.columns) >= MIN_COLS else '✗ 未完整'}")
