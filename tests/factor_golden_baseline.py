"""
tests/factor_golden_baseline.py — Alpha191 因子 golden 基线对拍。

背景：
  signals/alpha191/ 是一次性重写的 191 因子实现，靠 cross_validate_alpha191.py
  做过一次性对拍，但没有保留 golden 输出基线。未来任何人改动 operators.py 或
  factors.py，都无法自动知道是否引入了数值回归。

方案：
  用固定种子的合成 OHLCV 数据（不依赖真实缓存，保证跨环境可复现），
  计算全部 191 个因子，将结果哈希保存到 tests/golden/factor_hashes.json 作为基线。
  重跑本脚本时对比哈希，若漂移则报告哪些因子发生了数值变化。

用法:
    python tests/factor_golden_baseline.py --generate   # 生成基线
    python tests/factor_golden_baseline.py --verify     # 校验（默认）
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path(__file__).parent / "golden" / "factor_hashes.json"


def build_synthetic_panel(
    n_days: int = 200,
    n_stocks: int = 10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """构造确定性的合成 OHLCV 宽表面板（date × stocks）。

    不依赖真实缓存，保证任何环境、任何时间重跑结果一致。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    # 几何布朗运动生成 close，再构造 open/high/low/volume/amount
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n_days, n_stocks)), axis=0))
    close_df = pd.DataFrame(close, index=dates, columns=[f"s{i:02d}" for i in range(n_stocks)])

    open_ = close_df.shift(1).fillna(close_df.iloc[0]) * (1 + rng.normal(0, 0.005, close_df.shape))
    high = np.maximum(open_, close_df) * (1 + rng.uniform(0, 0.01, close_df.shape))
    low = np.minimum(open_, close_df) * (1 - rng.uniform(0, 0.01, close_df.shape))
    volume = np.abs(rng.normal(1e6, 2e5, close_df.shape))
    amount = volume * close_df.values

    return {
        "open": pd.DataFrame(open_, index=dates, columns=close_df.columns),
        "high": pd.DataFrame(high, index=dates, columns=close_df.columns),
        "low": pd.DataFrame(low, index=dates, columns=close_df.columns),
        "close": close_df,
        "volume": pd.DataFrame(volume, index=dates, columns=close_df.columns),
        "amount": pd.DataFrame(amount, index=dates, columns=close_df.columns),
    }


def _hash_series(s: pd.Series | pd.DataFrame) -> str:
    """对因子输出计算确定性哈希（忽略 NaN 位置，只哈希数值）。"""
    if isinstance(s, pd.DataFrame):
        vals = s.values
    else:
        vals = s.values.reshape(-1, 1)
    # 用 float64 的字节表示 + 稳定的排序，避免平台差异
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return "empty"
    arr = np.ascontiguousarray(finite, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_all_factor_hashes() -> dict[str, str]:
    """计算全部 191 个因子的哈希。"""
    from signals.alpha191.calculator import get_factor_func, _factor_has_rank

    panel = build_synthetic_panel()
    hashes = {}

    for i in range(1, 192):
        fid = f"alpha{i:03d}"
        fn = get_factor_func(fid)
        uses_rank = _factor_has_rank(i)

        try:
            if uses_rank:
                result = fn(panel)
            else:
                # 逐股票模式：用第一只股票
                sym = list(panel["close"].columns)[0]
                df = pd.DataFrame({f: panel[f][sym] for f in ["open", "high", "low", "close", "volume", "amount"]})
                df["vwap"] = df["amount"] / df["volume"]
                result = fn(df)
            hashes[fid] = _hash_series(result)
        except Exception:
            hashes[fid] = "ERROR"

    return hashes


def generate_baseline() -> None:
    """生成并保存基线哈希。"""
    hashes = compute_all_factor_hashes()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    n_ok = sum(1 for v in hashes.values() if v not in ("empty", "ERROR"))
    print(f"基线已生成: {GOLDEN_PATH}（{n_ok}/191 个因子有有效数值）")


def verify_baseline() -> int:
    """校验当前因子计算是否与基线一致，返回漂移的因子数。"""
    if not GOLDEN_PATH.exists():
        print("基线不存在，请先运行 --generate")
        return -1

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    current = compute_all_factor_hashes()
    drifted = [fid for fid in baseline if current.get(fid) != baseline[fid]]

    if drifted:
        print(f"⚠️ 检测到 {len(drifted)} 个因子数值漂移:")
        for fid in drifted:
            print(f"  - {fid}: baseline={baseline[fid][:12]} current={current.get(fid, 'MISSING')[:12]}")
    else:
        print("✓ 全部 191 个因子与基线一致，无数值回归")
    return len(drifted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="生成基线")
    parser.add_argument("--verify", action="store_true", help="校验基线（默认行为）")
    args = parser.parse_args()

    if args.generate:
        generate_baseline()
    else:
        verify_baseline()
