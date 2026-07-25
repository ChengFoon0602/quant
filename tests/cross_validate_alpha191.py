"""
Alpha191 因子交叉验证：你的 pandas 实现 vs. aurumq-rl Polars 参考引擎。

策略:
  对每只股票分别计算因子值（保证时序算子隔离），然后逐股票比较。
  - 不使用 RANK/TSRANK 的因子 → per-stock 结果应完全一致
  - 使用 RANK/TSRANK 的因子 → 标记为 rank_dependent，不比较（你的 RANK
    是时序排名，参考的 rank 是截面排名，语义不同）

不修改 signals/alpha191/ 中的任何代码。
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ── 路径 ──────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
AURUMQ_STUB = PROJECT / "tests" / "_aurumq_stub"
sys.path.insert(0, str(AURUMQ_STUB))

from data.fetcher import load_daily
from aurumq_rl.factors.registry import GTJA191_REGISTRY  # noqa: E402
from aurumq_rl.factors import gtja191  # noqa: E402, F401 触发注册

# ══════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════

SAMPLE_SYMBOLS = [
    "000001", "000002", "000858",
    "600000", "600036", "600519", "600887", "601318",
    "000725", "002415",
]
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
PERFECT_CORR = 0.999
GOOD_CORR = 0.95
REPORT_PATH = PROJECT / "tests" / "alpha191_diff_report.md"


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _sym_suffix(sym: str) -> str:
    if sym.startswith(("0", "3")): return ".SZ"
    if sym.startswith("6"): return ".SH"
    return ".BJ"


def _your_factor_func(num: int):
    import importlib
    mod = importlib.import_module("signals.alpha191.factors")
    return getattr(mod, f"factor_{num:03d}")


def _ref_factor_func(num: int):
    """返回 raw impl（无 sanitize 裁尾）和 quality_flag。"""
    entry = GTJA191_REGISTRY.get(f"gtja_{num:03d}")
    if entry is None:
        return None, None
    raw = getattr(entry.impl, "__wrapped__", entry.impl)
    return raw, entry.quality_flag


def _factor_has_rank(factor_num: int) -> bool:
    """扫描因子函数及所调用的辅助函数源码，判断是否使用了 RANK/TSRANK。"""
    try:
        fn = _your_factor_func(factor_num)
        src = inspect.getsource(fn)
    except Exception:
        return False

    # 直接检查函数体
    if "RANK(" in src or "TSRANK(" in src or "RANK " in src:
        return True

    # 检查工厂函数引用（_ret_rank_pair, _corr_vol_pair, _bull_bear_pair 等内部用了 RANK）
    rank_helpers = [
        "_ret_rank_pair", "_corr_vol_pair",
        "_make_regbeta_factor", "_make_regresi_factor",
        "_make_count_up_factor", "_make_count_down_factor",
    ]
    for helper in rank_helpers:
        if helper in src:
            return True

    return False


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Alpha191 交叉验证 — 逐股票时序比较")
    print("=" * 70)

    # ── 加载数据 ──
    print("\n[1/3] 加载数据...")
    stock_data: dict[str, pd.DataFrame] = {}
    for sym in SAMPLE_SYMBOLS:
        df = load_daily(sym)
        if df is None or len(df) < 100:
            continue
        df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
        df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]
        if len(df) < 100:
            continue
        df["vwap"] = df["amount"] / df["volume"]
        stock_data[sym] = df
    print(f"  加载 {len(stock_data)} 只股票, "
          f"每只 {min(len(v) for v in stock_data.values())}~"
          f"{max(len(v) for v in stock_data.values())} 天")

    # ── 分类：是否有 RANK ──
    no_rank_factors = []
    rank_factors = []
    for n in range(1, 192):
        if _factor_has_rank(n):
            rank_factors.append(n)
        else:
            no_rank_factors.append(n)
    print(f"\n  不使用 RANK: {len(no_rank_factors)} 个因子 → 可逐股票精确比较")
    print(f"  使用 RANK:   {len(rank_factors)} 个因子 → RANK 语义差异, 不强行比较")

    # ── 逐股票、逐因子（无 RANK）比较 ──
    results = []  # list[dict]

    print(f"\n[2/3] 逐股票比较 {len(no_rank_factors)} 个非 RANK 因子...")
    for sym, df_stock in stock_data.items():
        code = f"{sym}{_sym_suffix(sym)}"

        # 构造单股票 Polars 面板给 aurumq-rl
        p_pd = pd.DataFrame({
            "stock_code": code,
            "trade_date": pd.to_datetime(df_stock.index),
            "open": df_stock["open"].values,
            "high": df_stock["high"].values,
            "low": df_stock["low"].values,
            "close": df_stock["close"].values,
            "volume": df_stock["volume"].values,
            "amount": df_stock["amount"].values,
            "vwap": df_stock["vwap"].values,
        })
        p_pd["prev_close"] = p_pd["close"].shift(1)
        p_pd["returns"] = p_pd["close"] / p_pd["prev_close"] - 1.0
        p_pl = pl.from_pandas(p_pd).with_columns(pl.col("trade_date").cast(pl.Date))

        for num in no_rank_factors:
            # 你的引擎
            try:
                fn_y = _your_factor_func(num)
                y_series = fn_y(df_stock)
                if isinstance(y_series, np.ndarray):
                    y_series = pd.Series(y_series, index=df_stock.index)
            except Exception:
                y_series = pd.Series(np.nan, index=df_stock.index)

            # 参考引擎
            ref_impl, qflag = _ref_factor_func(num)
            try:
                r_series = ref_impl(p_pl).to_numpy()
            except Exception:
                r_series = np.full(len(df_stock), np.nan)

            # 比较
            y_arr = y_series.values
            r_arr = r_series
            mask = ~np.isnan(y_arr) & ~np.isnan(r_arr)
            y, r = y_arr[mask], r_arr[mask]

            if len(y) < 10:
                cat = "insufficient_data"
                corr_val = np.nan
                max_rd = np.nan
            elif y.std() == 0 or r.std() == 0:
                cat = "constant"
                corr_val = np.nan
                max_rd = np.nan
            elif qflag == 2:
                cat = "stub_ref"
                corr_val = np.nan
                max_rd = np.nan
            else:
                corr_val = float(np.corrcoef(y, r)[0, 1])
                ref_scale = max(np.abs(r).max(), 1e-8)
                max_rd = float((np.abs(y - r) / ref_scale).max())

                if corr_val > PERFECT_CORR:
                    cat = "perfect"
                elif corr_val > GOOD_CORR:
                    cat = "good"
                elif corr_val > 0.9:
                    cat = "minor_diff"
                else:
                    cat = "sign_diff"

            results.append({
                "num": num,
                "stock": sym,
                "category": cat,
                "corr": corr_val,
                "max_rel_diff": max_rd,
                "quality_flag": qflag,
                "n_valid": len(y),
                "uses_rank": False,
            })

    # ── 面板模式：比较 RANK 因子（截面排名） ──
    if rank_factors:
        print(f"\n[2.5/3] 面板模式比较 {len(rank_factors)} 个 RANK 因子（截面排名）...")
        # 构建宽表面板
        panel = {}
        for field in ["open", "high", "low", "close", "volume", "amount"]:
            panel[field] = pd.DataFrame(
                {sym: df[field] for sym, df in stock_data.items()}
            ).sort_index()
        panel["vwap"] = panel["amount"] / panel["volume"]

        # 构建 aurumq-rl 长格式面板
        pl_dfs = []
        for sym, df_stock in stock_data.items():
            code = f"{sym}{_sym_suffix(sym)}"
            p_pd = pd.DataFrame({
                "stock_code": code,
                "trade_date": pd.to_datetime(df_stock.index),
                "open": df_stock["open"].values,
                "high": df_stock["high"].values,
                "low": df_stock["low"].values,
                "close": df_stock["close"].values,
                "volume": df_stock["volume"].values,
                "amount": df_stock["amount"].values,
                "vwap": df_stock["vwap"].values,
            })
            pl_dfs.append(p_pd)
        pl_long = pd.concat(pl_dfs, ignore_index=True)
        pl_long["prev_close"] = pl_long.groupby("stock_code")["close"].shift(1)
        pl_long["returns"] = pl_long["close"] / pl_long["prev_close"] - 1.0
        p_pl = pl.from_pandas(pl_long).with_columns(pl.col("trade_date").cast(pl.Date))

        for num in rank_factors:
            ref_impl, qflag = _ref_factor_func(num)
            if qflag == 2:
                # 参考实现存根，跳过
                results.append({
                    "num": num, "stock": "ALL", "category": "stub_ref",
                    "corr": np.nan, "max_rel_diff": np.nan,
                    "quality_flag": qflag, "n_valid": 0, "uses_rank": True,
                })
                continue

            # 你的引擎（面板模式，截面 RANK）
            try:
                fn_y = _your_factor_func(num)
                y_df = fn_y(panel)
                if not isinstance(y_df, pd.DataFrame):
                    y_df = pd.DataFrame(y_df)
            except Exception:
                y_df = pd.DataFrame(np.nan, index=panel["close"].index,
                                    columns=panel["close"].columns)

            # 参考引擎
            try:
                r_long = ref_impl(p_pl).to_numpy()
            except Exception:
                r_long = np.full(len(p_pl), np.nan)

            # 参考引擎结果 pivot 到宽表
            r_wide = pl_long[["trade_date", "stock_code"]].copy()
            r_wide["value"] = r_long
            try:
                r_df = r_wide.pivot_table(
                    index="trade_date", columns="stock_code", values="value", aggfunc="first"
                )
                r_df = r_df.sort_index()
            except Exception:
                r_df = pd.DataFrame(np.nan, index=y_df.index, columns=y_df.columns)

            # 对齐两个 DataFrame — 列名映射（aurumq-rl 带后缀，你的不带）
            sym_map = {f"{sym}{_sym_suffix(sym)}": sym for sym in stock_data}
            r_df = r_df.rename(columns=sym_map)
            common_cols = [c for c in y_df.columns if c in r_df.columns]
            common_idx = y_df.index.intersection(r_df.index)
            y_aligned = y_df.loc[common_idx, common_cols]
            r_aligned = r_df.loc[common_idx, common_cols]

            # 逐股票比较
            for sym in common_cols:
                y_arr = y_aligned[sym].values
                r_arr = r_aligned[sym].values
                mask = ~np.isnan(y_arr) & ~np.isnan(r_arr)
                y, r = y_arr[mask], r_arr[mask]

                if len(y) < 10:
                    cat = "insufficient_data"
                    corr_val = np.nan
                    max_rd = np.nan
                elif y.std() == 0 or r.std() == 0:
                    cat = "constant"
                    corr_val = np.nan
                    max_rd = np.nan
                else:
                    corr_val = float(np.corrcoef(y, r)[0, 1])
                    ref_scale = max(np.abs(r).max(), 1e-8)
                    max_rd = float((np.abs(y - r) / ref_scale).max())

                    if corr_val > PERFECT_CORR:
                        cat = "perfect"
                    elif corr_val > GOOD_CORR:
                        cat = "good"
                    elif corr_val > 0.9:
                        cat = "minor_diff"
                    else:
                        cat = "sign_diff"

                results.append({
                    "num": num, "stock": sym, "category": cat,
                    "corr": corr_val, "max_rel_diff": max_rd,
                    "quality_flag": qflag, "n_valid": len(y),
                    "uses_rank": True,
                })

    # ── 汇总 ──
    # 按因子聚合：如果任一股票是 sign_diff 则该因子是 sign_diff
    df_r = pd.DataFrame(results)
    factor_summary = []
    all_nums = set(df_r["num"].tolist()) if len(df_r) else set()

    for num in sorted(all_nums):
        sub = df_r[df_r["num"] == num]
        cats = sub["category"].tolist()
        # 取最差类别
        priority = {"error": 0, "sign_diff": 1, "insufficient_data": 2,
                    "minor_diff": 3, "constant": 4, "stub_ref": 5,
                    "good": 6, "perfect": 7}
        worst_cat = min(cats, key=lambda c: priority.get(c, 8))
        corrs = sub["corr"].dropna()
        uses_rank = bool(sub["uses_rank"].iloc[0]) if len(sub) else False
        factor_summary.append({
            "num": num,
            "category": worst_cat,
            "min_corr": float(corrs.min()) if len(corrs) else np.nan,
            "qflag": int(sub["quality_flag"].iloc[0]) if len(sub) else 0,
            "stocks_affected": int((sub["category"] != "perfect").sum()),
            "uses_rank": uses_rank,
            "n_valid_total": int(sub["n_valid"].sum()),
        })

    # 补上遗漏的因子（参考存根或其他未比较的）
    covered = {fs["num"] for fs in factor_summary}
    for num in range(1, 192):
        if num not in covered:
            _, qflag = _ref_factor_func(num)
            factor_summary.append({
                "num": num, "category": "stub_ref" if qflag == 2 else "error",
                "min_corr": np.nan, "qflag": qflag,
                "stocks_affected": 0,
                "uses_rank": _factor_has_rank(num),
                "n_valid_total": 0,
            })

    # ── 报告 ──
    print(f"\n[3/3] 生成报告...")
    _print_and_save(factor_summary, list(stock_data.keys()))
    return factor_summary


# ══════════════════════════════════════════════════════════════
#  报告输出
# ══════════════════════════════════════════════════════════════

def _print_and_save(factor_summary: list[dict], symbols_used: list[str]):
    cats = {}
    for fs in factor_summary:
        cats.setdefault(fs["category"], []).append(fs)

    desc = {
        "perfect": "corr > 0.999 — 逐股票完全一致",
        "good": "corr > 0.95 — 微小精度差异",
        "minor_diff": "0.9 < corr < 0.95 — 可接受的偏差",
        "sign_diff": "corr < 0.9 — 方向性分歧，需排查",
        "rank_dependent": "使用 RANK/TSRANK → 面板模式截面排名比较",
        "stub_ref": "参考实现存根 (qflag=2) — 跳过",
        "constant": "输出全常数 — 该样本无方差",
        "insufficient_data": "有效数据 < 10 点",
    }

    print("\n" + "=" * 70)
    print("  Alpha191 交叉验证结果")
    print("=" * 70)
    print(f"  总因子数: {len(factor_summary)}")
    print(f"  测试股票: {', '.join(symbols_used)}")
    print(f"  数据范围: {START_DATE} ~ {END_DATE}")
    print()

    for cat_name in ["perfect", "good", "minor_diff", "sign_diff",
                      "rank_dependent", "stub_ref", "constant", "insufficient_data"]:
        items = cats.get(cat_name, [])
        d = desc.get(cat_name, "")
        print(f"  [{cat_name}] {len(items)} 个 — {d}")

    # 非 perfect 详细
    for cat_name in ["sign_diff", "minor_diff", "good"]:
        items = cats.get(cat_name, [])
        if not items:
            continue
        print(f"\n  {'─' * 60}")
        print(f"  [{cat_name}] 详情:")
        for fs in sorted(items, key=lambda x: x.get("min_corr", 0)):
            print(f"    alpha{fs['num']:03d}  min_corr={fs['min_corr']:+.6f}  "
                  f"affected={fs['stocks_affected']}/{len(symbols_used)}  qflag={fs['qflag']}")

    # ── Markdown ──
    lines = [
        "# Alpha191 因子交叉验证报告",
        "",
        f"**生成日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 方法论说明",
        "",
        "比较策略：",
        "- **不使用 RANK/TSRANK 的因子**：逐只股票计算，两个引擎直接比较。"
        "此时 aurumq-rl 的时序算子（SUM, STD, DELTA, CORR, SMA 等）用 `over(\"stock_code\")` "
        "分区，等价于你的 per-stock 调用。",
        "- **使用 RANK/TSRANK 的因子**：面板模式比较。你的 RANK 现已是真正的截面排名 "
        "（`DataFrame.rank(axis=1, pct=True)`，跨股票排名），与 aurumq-rl 的 "
        "`rank(\"ordinal\").over(\"trade_date\")` 语义一致。"
        "两个引擎均以宽表面板为输入，逐股票提取结果后计算相关性。",
        "",
        "## 数据概况",
        "",
        f"- 测试股票: {', '.join(symbols_used)}",
        f"- 数据范围: {START_DATE} ~ {END_DATE}",
        f"- 因子总数: {len(factor_summary)}",
        f"- 非 RANK 因子: {sum(1 for fs in factor_summary if not fs['uses_rank'])}",
        f"- RANK 因子: {sum(1 for fs in factor_summary if fs['uses_rank'])}",
        "",
        "## 分级汇总",
        "",
        "| 类别 | 数量 | 说明 |",
        "|------|------|------|",
    ]
    for cat_name in ["perfect", "good", "minor_diff", "sign_diff",
                      "rank_dependent", "stub_ref", "constant", "insufficient_data"]:
        items = cats.get(cat_name, [])
        d = desc.get(cat_name, "")
        lines.append(f"| **{cat_name}** | {len(items)} | {d} |")

    # sign_diff 详情
    sign_items = cats.get("sign_diff", [])
    if sign_items:
        lines.append("")
        lines.append("## ⚠ 方向性分歧 (sign_diff)")
        lines.append("")
        lines.append("| 因子 | Min Corr | Stock Affected | qflag | 排查建议 |")
        lines.append("|------|----------|---------------|-------|---------|")
        for fs in sorted(sign_items, key=lambda x: x.get("min_corr", 0)):
            lines.append(
                f"| alpha{fs['num']:03d} | {fs['min_corr']:+.6f} | "
                f"{fs['stocks_affected']}/{len(symbols_used)} | {fs['qflag']} | "
                f"检查公式实现 |")

    # RANK 因子
    rank_items = cats.get("rank_dependent", [])
    if rank_items:
        lines.append("")
        lines.append("## RANK 相关因子 (rank_dependent)")
        lines.append("")
        lines.append(f"共 {len(rank_items)} 个因子使用了 RANK 或 TSRANK，未进行数值比较。")
        lines.append("这些因子的公式实现可能是正确的，但 `RANK` 算子在截面场景下需要改为 `groupby('trade_date').rank()`。")
        lines.append("")
        ids = sorted(fs["num"] for fs in rank_items)
        lines.append(f"涉及因子: {', '.join(f'alpha{i:03d}' for i in ids)}")

    lines.append("")
    lines.append("## 解读说明")
    lines.append("")
    lines.append("- **perfect**: 两个独立实现对同一只股票产生相同的输出 → 非 RANK 算子的公式和实现都正确")
    lines.append("- **sign_diff**: 需排查——可能是因子公式本身的实现错误（非 RANK 相关）")
    lines.append("- **rank_dependent**: RANK 语义差异，不是代码 bug。要验证这些因子，需要先把 `RANK` 改为真正的截面排名")
    lines.append("- **qflag**: aurumq-rl 的 quality_flag — 0=正确, 1=论文公式有已知勘误, 2=存根")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  报告已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
