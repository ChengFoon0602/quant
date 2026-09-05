"""
models/portfolio_backtest.py — 组合回测 + 条件 Alpha 归因。

任务:
  1. 将 LightGBM OOF 预测概率转化为交易组合:
     - 每天做多 Top 20%，做空 Bottom 20%（long-short）
     - 以及仅做多 Top 20%（long-only），对比市场基准
     - 双边摩擦成本 0.1%（买 0.026% / 卖 0.076%，铁律标准）
  2. 与 alpha001 单因子组合对比（同样 0.1% 成本）
  3. Bootstrap 夏普显著性检验
  4. 条件 Alpha 归因：按 market_vol_20d 分高/低波动，看各因子重要性变化

用法:
    cd D:/桌面文件/quant
    python models/portfolio_backtest.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from models.labels import align_X_y, build_labels

# ── 配置 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "strategies" / "feature_selection"
MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TOP_Q = 0.20
BOTTOM_Q = 0.20
# 成本口径（2026-09 修正）：默认走方向分离 buy_cost/sell_cost（铁律 0.1%）。
# COST_BPS 仅作向后兼容的「双边合计」常量保留，新代码不再使用。
COST_BPS = 0.00102  # 双边合计 ≈ 0.1%（买 0.026% + 卖 0.076%）
BUY_COST = 0.00026
SELL_COST = 0.00076
FWD_DAYS = 5
N_BOOT = 10000
BLOCK_SIZE = 20

MARKET_COLS = ["market_vol_20d", "market_turnover_20d"]
# alpha 因子列从 X_matrix 动态推断（因子池由 select_features.py 决定，禁止硬编码）
ALL_FEATURES: list[str] = []  # 在 main() 中由 X_long 列填充


def load_data():
    """加载 OOF 预测、X_matrix、close_matrix。"""
    pred_path = MODEL_DIR / "oof_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("先运行 python models/lgbm_trainer.py 生成 OOF 预测")
    pred_lgb = pd.read_csv(pred_path, index_col=0, parse_dates=True)

    x_path = FEATURE_DIR / "X_matrix.csv"
    X_raw = pd.read_csv(x_path, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col]).astype(float)

    # alpha001 因子矩阵
    alpha001 = X_long["alpha001"].unstack(level=1)

    # market_vol_20d
    market_vol = X_long["market_vol_20d"].unstack(level=1).iloc[:, 0]  # 每天所有股票相同

    # 重建 close_matrix：只取 X_matrix 内实际出现的股票
    # （禁止 sorted(cache)[:300] 切片：缓存扩容后该切片会漂移）
    symbols = sorted(X_long.index.get_level_values(1).unique())
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()

    common_stocks = pred_lgb.columns.intersection(close_matrix.columns).intersection(alpha001.columns)
    pred_lgb = pred_lgb[common_stocks]
    alpha001 = alpha001[common_stocks]
    close_matrix = close_matrix[common_stocks]

    # 市场基准：当日指数成员（= X_matrix 出现的样本）的等权日收益。
    # 权重恒定等权 → 无重叠 tranche，直接取截面均值；
    # 旧版 rolling(5).mean() 是平滑陷阱残留（虚压波动 → 基准夏普虚高），已移除
    member_mask = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    daily_ret = close_matrix.pct_change()
    mask_aligned = member_mask.reindex_like(daily_ret).fillna(False).astype(bool)
    market_ret = daily_ret.where(mask_aligned).mean(axis=1).dropna()

    return pred_lgb, alpha001, close_matrix, market_ret, market_vol, X_long


def build_portfolio(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    long_only: bool = False,
    short_only: bool = False,
    top_q: float = TOP_Q,
    bottom_q: float = BOTTOM_Q,
    cost: float | None = None,
    buy_cost: float = 0.00026,
    sell_cost: float = 0.00076,
    hold_days: int = 5,
    position_scale: pd.Series | None = None,
    gate: pd.Series | None = None,
    return_flows: bool = False,
    return_weights: bool = False,
) -> pd.DataFrame | tuple:
    """构建 overlapped 投资组合 —— 基于权重追踪的真实每日 P&L。

    ⚠️ 方法论修正（2026-07）：旧实现对"信号收益序列"做 rolling(hold_days).mean()，
    等于把 10 个不同日历日的收益平均后盖上"第 t 天"的戳，是移动平均滤波，
    人为把日波动压掉 √hold_days 倍 → 夏普虚高 ~8x。见 build_portfolio_naive 与
    report.md「方法论修正」章节。

    ⚠️ 口径修正（2026-09）：收益定义从错位 close(t+2)/close(t+1)-1 改为标准
    pct_change()（t→t+1 收益记在 t+1 日），与全仓库 engine/cross_section/labels 对齐。
    错位收益让「信号→收益」隔 2 天空窗，导致动量类 alpha 被系统性低估约 10%
    （非未来函数，但方向保守）。详见 docs/收益成本口径统一论证.md。

    正确构造（Method 1）：
      1. 每天由 signal[t] 生成目标权重向量 w[t]（多头 +1/n_top，空头 -1/n_bottom）。
      2. 实际持仓 W[t] = 过去 hold_days 天目标权重的平均（重叠 tranche）。
      3. 第 t 天组合收益 = W[t-1] · daily_ret[t]，其中
         daily_ret[t] = close[t]/close[t-1]-1（pct_change，收益锚定 t 日）。
         所有 tranche 在同一天经历同一市场波动 → 无虚假平滑。
      4. 成本 = 换手 × 方向分离费率：买入 buy_cost、卖出 sell_cost。
         cost 参数保留向后兼容（双边合计，传入时 buy_cost=sell_cost=cost/2）。

    position_scale: 可选的逐日仓位系数（post-multiply，乘在 W_held 上；如高波动降半仓）。
    gate: 可选的逐日 regime 闸门（pre-multiply，乘在 W_target 上、rolling 之前）。
          gate-off（0）日不开新 tranche，闸门开启后持仓用 hold 天爬坡——语义是
          "闸门关闭期间策略空仓，不纸上建仓"（P2 熊市做空端主口径）。
    short_only: 只写空头分支（bottom 分位等权做空），与 long_only 互斥。
    return_weights: 为 True 时返回 (df, W_held)，W_held 为实际持仓权重（融券成本/暴露统计）。
    """
    # 成本口径归一：cost 为 None 用方向分离；否则对半拆（向后兼容）
    if cost is not None:
        buy_cost = cost / 2.0
        sell_cost = cost / 2.0

    # 收益锚定日约定（全仓库统一）：t→t+1 收益记在 t+1 日，即 pct_change()
    daily_ret = close_matrix.pct_change()

    common_dates = pred_df.index.intersection(daily_ret.index)
    common_cols = pred_df.columns.intersection(daily_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = daily_ret.loc[common_dates, common_cols]

    if long_only and short_only:
        raise ValueError("long_only 与 short_only 互斥，不能同时为 True")

    # ── 每天生成目标权重向量 w[t] ──
    W_target = pd.DataFrame(0.0, index=common_dates, columns=common_cols)
    for d in common_dates:
        pv = p.loc[d]
        mask = pv.notna()
        if mask.sum() < max(int(1 / top_q), int(1 / bottom_q)) * 3:
            continue
        valid_p = pv[mask]
        top_thr = valid_p.quantile(1 - top_q)
        bottom_thr = valid_p.quantile(bottom_q)
        top = valid_p[valid_p >= top_thr].index
        bottom = valid_p[valid_p <= bottom_thr].index
        if not short_only and len(top):
            W_target.loc[d, top] = 1.0 / len(top)
        if not long_only and len(bottom):
            W_target.loc[d, bottom] = -1.0 / len(bottom)

    # ── regime 闸门（pre-multiply）：gate-off 日不开新 tranche ──
    if gate is not None:
        g = gate.reindex(W_target.index).ffill().fillna(0.0)
        W_target = W_target.mul(g, axis=0)

    # ── 实际持仓 = 过去 hold_days 天目标权重的平均（重叠 tranche）──
    W_held = W_target.rolling(hold_days, min_periods=1).mean()

    # 逐日仓位系数（动态仓位）
    if position_scale is not None:
        scale = position_scale.reindex(W_held.index).ffill().fillna(1.0)
        W_held = W_held.mul(scale, axis=0)

    # ── 第 t 天 P&L = 昨日持仓 · 今日股票收益 ──
    W_lag = W_held.shift(1)
    port_gross = (W_lag * r).sum(axis=1, min_count=1)

    # ── 换手成本：方向分离（买 buy_cost / 卖 sell_cost）──
    delta_w = W_held - W_held.shift(1)
    turnover = delta_w.abs().sum(axis=1)
    buy_turnover = delta_w.clip(lower=0.0).sum(axis=1)
    sell_turnover = (-delta_w).clip(lower=0.0).sum(axis=1)
    port_ret = port_gross - buy_turnover * buy_cost - sell_turnover * sell_cost

    # 丢弃建仓爬坡期
    port_ret = port_ret.iloc[hold_days:].dropna()
    cum = (1 + port_ret).cumprod()

    df = pd.DataFrame({"port_ret": port_ret, "turnover": turnover.reindex(port_ret.index)})
    df["cum"] = cum
    if return_weights:
        return df, W_held.reindex(port_ret.index)
    if return_flows:
        # 逐日逐股交易流 |ΔW|（缩放后真实持仓变化），供容量检验模拟冲击成本
        flows = (W_held - W_held.shift(1)).abs().reindex(port_ret.index)
        return df, flows
    return df


def build_portfolio_naive(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    long_only: bool = False,
    top_q: float = TOP_Q,
    bottom_q: float = BOTTOM_Q,
    cost: float = COST_BPS,
    hold_days: int = 5,
) -> pd.DataFrame:
    """⚠️ 错误的旧实现，仅供「方法论修正」章节对比展示，禁止用于生产结论。

    对"信号收益序列"做 rolling(hold_days).mean() → 移动平均平滑 → 夏普虚高 √hold_days 倍。
    收益定义已统一为 pct_change（2026-09），使 naive vs correct 的差异纯粹来自平滑陷阱，
    不混入收益口径差异。
    """
    daily_ret = close_matrix.pct_change()
    common_dates = pred_df.index.intersection(daily_ret.index)
    common_cols = pred_df.columns.intersection(daily_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = daily_ret.loc[common_dates, common_cols]

    signal_rets = []
    for d in common_dates:
        pv, rv = p.loc[d], r.loc[d]
        mask = pv.notna() & rv.notna()
        if mask.sum() < max(int(1 / top_q), int(1 / bottom_q)) * 3:
            signal_rets.append(np.nan)
            continue
        valid_p, valid_r = pv[mask], rv[mask]
        top_thr = valid_p.quantile(1 - top_q)
        bottom_thr = valid_p.quantile(bottom_q)
        top_ret = valid_r[valid_p >= top_thr].mean()
        bottom_ret = valid_r[valid_p <= bottom_thr].mean() if not long_only else 0.0
        signal_rets.append(top_ret - bottom_ret)

    signal_series = pd.Series(signal_rets, index=common_dates)
    port_ret = signal_series.rolling(hold_days).mean()
    if long_only:
        port_ret = port_ret - cost / hold_days
    else:
        port_ret = port_ret - 2 * cost / hold_days
    port_ret = port_ret.dropna()
    cum = (1 + port_ret).cumprod()
    return pd.DataFrame({"port_ret": port_ret, "cum": cum})


def performance_metrics(ret_series: pd.Series) -> dict:
    ret = ret_series.dropna()
    if len(ret) == 0:
        return {"annual": 0.0, "sharpe": 0.0, "mdd": 0.0, "n": 0}
    ann = (1 + ret.mean()) ** 252 - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    cum = (1 + ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {"annual": ann, "sharpe": sharpe, "mdd": dd.min(), "n": len(ret)}


def block_bootstrap_sharpe(ret_series: pd.Series, n_boot: int = N_BOOT, block_size: int = BLOCK_SIZE) -> tuple:
    """Block Bootstrap 检验夏普是否显著 > 0。"""
    ret = ret_series.dropna().values
    n = len(ret)
    if n < block_size * 2:
        return np.array([]), 1.0

    observed = performance_metrics(ret_series)["sharpe"]
    boot_sharpes = []
    for _ in range(n_boot):
        # 拼接 block
        blocks = []
        while len(blocks) < n:
            start = np.random.randint(0, n - block_size + 1)
            blocks.extend(ret[start:start + block_size])
        sample = np.array(blocks[:n])
        mean_s = sample.mean()
        std_s = sample.std()
        if std_s > 0:
            boot_sharpes.append(mean_s / std_s * np.sqrt(252))
        else:
            boot_sharpes.append(0.0)

    boot_sharpes = np.array(boot_sharpes)
    p_value = (boot_sharpes <= 0).mean() if observed > 0 else (boot_sharpes >= 0).mean()
    return boot_sharpes, p_value


def _permute_column(X_reg: pd.DataFrame, col: str) -> pd.DataFrame:
    """对单个特征做 permutation。

    alpha 因子：截面打乱（按日期 groupby shuffle）。
    市场特征：时间序列打乱，因为每天所有股票值相同。
    """
    X_perm = X_reg.copy()
    if col in MARKET_COLS:
        date_vals = X_perm[col].groupby(level=0).first()
        shuffled_dates = np.random.permutation(date_vals.index)
        shuffled_map = dict(zip(date_vals.index, shuffled_dates))
        permuted = X_perm.index.get_level_values(0).map(lambda d: date_vals.loc[shuffled_map[d]])
        X_perm[col] = permuted.values
    else:
        X_perm[col] = X_perm[col].groupby(level=0).transform(lambda x: x.sample(frac=1).values)
    return X_perm


def conditional_permutation_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    market_feature: pd.Series,
    feature_cols: list[str],
    n_bins: int = 2,
) -> dict:
    """按市场状态分箱，计算每个特征的 Permutation Importance (AUC 下降)。"""
    from sklearn.metrics import roc_auc_score

    # 市场状态分箱
    date_to_vol = market_feature.groupby(level=0).first().to_dict()
    mf_values = X.index.get_level_values(0).map(date_to_vol)
    mf = pd.Series(mf_values, index=X.index)
    valid_mask = mf.notna()
    mf = mf[valid_mask]
    X = X.loc[valid_mask].copy()
    # 用 rank 分箱避免 qcut 因重复值失败
    ranks = mf.rank(pct=True)
    bins = pd.cut(ranks, bins=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])

    base_auc = roc_auc_score(X["label"], model.predict(X[feature_cols].values))
    results = {}
    for regime in bins.unique():
        if pd.isna(regime):
            continue
        mask = bins == regime
        X_reg = X.loc[mask, feature_cols]
        y_reg = X.loc[mask, "label"]
        if y_reg.nunique() < 2:
            continue
        base_reg = roc_auc_score(y_reg, model.predict(X_reg.values))
        imp = {}
        for col in feature_cols:
            X_perm = _permute_column(X_reg, col)
            perm_auc = roc_auc_score(y_reg, model.predict(X_perm.values))
            imp[col] = base_reg - perm_auc
        results[regime] = imp
    return results, base_auc


def plot_results(results: dict, boot_dist: dict):
    """生成回测对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计净值对比
    ax = axes[0, 0]
    for name, df in results.items():
        ax.plot(df.index, df["cum"], label=name, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("累计净值对比（扣 0.1% 双边成本）")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. 月度收益热力图
    ax = axes[0, 1]
    lgb_ret = results.get("LightGBM LS")
    if lgb_ret is not None:
        monthly = lgb_ret["port_ret"].dropna().resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = pd.MultiIndex.from_arrays([monthly.index.year, monthly.index.month])
        pivot = monthly.unstack()
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("LightGBM Long-Short 月度收益")
        plt.colorbar(im, ax=ax)

    # 3. Bootstrap SR 分布
    ax = axes[1, 0]
    colors = {"LightGBM LS": "green", "alpha001 LS": "blue", "LightGBM Long-only": "orange"}
    for name, (dist, p) in boot_dist.items():
        if len(dist) == 0:
            continue
        ax.hist(dist, bins=50, alpha=0.5, label=f"{name} (p={p:.4f})", color=colors.get(name))
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Bootstrap 夏普比率分布")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 绩效指标表
    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for name, df in results.items():
        m = performance_metrics(df["port_ret"])
        _, p = boot_dist.get(name, (np.array([]), 1.0))
        rows.append([name, f"{m['annual']:.2%}", f"{m['sharpe']:.3f}", f"{m['mdd']:.2%}", f"{p:.4f}"])
    table = ax.table(cellText=rows, colLabels=["组合", "年化", "夏普", "最大回撤", "Bootstrap p"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title("绩效汇总")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "portfolio_backtest_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图表保存: {fig_path}")


def main():
    print("=" * 70)
    print("组合回测 + 条件 Alpha 归因")
    print("=" * 70)

    pred_lgb, alpha001, close_matrix, market_ret, market_vol, X_long = load_data()

    # ── 组合构建 ──
    print("\n构建组合（铁律成本：买 0.026% / 卖 0.076%）...")
    lgb_ls = build_portfolio(pred_lgb, close_matrix, long_only=False)
    lgb_lo = build_portfolio(pred_lgb, close_matrix, long_only=True)
    a001_ls = build_portfolio(alpha001, close_matrix, long_only=False)

    # 市场基准净值
    market_cum = (1 + market_ret.fillna(0)).cumprod()

    results = {
        "LightGBM LS": lgb_ls,
        "LightGBM Long-only": lgb_lo,
        "alpha001 LS": a001_ls,
    }

    # ── 绩效指标 ──
    print("\n绩效汇总:")
    boot_dist = {}
    for name, df in results.items():
        m = performance_metrics(df["port_ret"])
        dist, p = block_bootstrap_sharpe(df["port_ret"])
        boot_dist[name] = (dist, p)
        status = "✓" if (m["sharpe"] > 1.40 and p < 0.05) else "✗"
        print(f"  {name:25s} 年化={m['annual']:+.2%}  夏普={m['sharpe']:.3f}  "
              f"最大回撤={m['mdd']:.2%}  Bootstrap p={p:.4f}  {status}")

    # 市场基准
    m_mkt = performance_metrics(market_ret)
    print(f"  {'市场等权':25s} 年化={m_mkt['annual']:+.2%}  夏普={m_mkt['sharpe']:.3f}  "
          f"最大回撤={m_mkt['mdd']:.2%}")

    # ── 条件归因 ──
    print("\n条件 Alpha 归因（按 market_vol_20d 分高低波动）...")
    universe = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    labels = build_labels(close_matrix, fwd_days=FWD_DAYS, top_q=0.2, bottom_q=0.2,
                          universe=universe)
    aligned = align_X_y(X_long, labels)

    # 用 fold 5 模型做条件归因（训练数据最多）
    model_path = MODEL_DIR / "lgbm_fold_5.txt"
    if model_path.exists():
        # 特征列表从 X_matrix 动态推断（与 lgbm_trainer 的列序一致：alpha 在前，市场特征在后）
        all_features = [c for c in X_long.columns if c not in MARKET_COLS] + MARKET_COLS
        ALL_FEATURES[:] = all_features  # 供 _permute_column 等模块级引用
        model = lgb.Booster(model_file=str(model_path))
        valid = aligned.dropna(subset=["label"])
        market_vol_long = X_long["market_vol_20d"]
        imp_dict, base_auc = conditional_permutation_importance(
            model, valid, market_vol_long, all_features, n_bins=2
        )
        print(f"  基础 AUC = {base_auc:.4f}")
        for regime, imps in imp_dict.items():
            print(f"\n  {regime} 波动区间:")
            for feat, drop in sorted(imps.items(), key=lambda x: x[1], reverse=True):
                print(f"    {feat:20s} AUC 下降 = {drop:+.6f}")

        # 条件归因图
        fig, ax = plt.subplots(figsize=(12, 6))
        regimes = list(imp_dict.keys())
        plot_df = pd.DataFrame(imp_dict).T[all_features]
        x = np.arange(len(all_features))
        width = 0.35
        for i, regime in enumerate(regimes):
            ax.bar(x + i * width, plot_df.loc[regime].values, width, label=regime)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(all_features, rotation=45, ha="right")
        ax.set_ylabel("Permutation Importance (AUC 下降)")
        ax.set_title("条件 Alpha 归因：高/低波动区间的特征重要性")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        att_path = FIGURES_DIR / "conditional_attribution.png"
        fig.savefig(att_path, dpi=150)
        plt.close(fig)
        print(f"\n条件归因图保存: {att_path}")
    else:
        print("  未找到 fold 5 模型，跳过条件归因")

    # ── 作图 ──
    plot_results(results, boot_dist)

    # ── 保存结果 ──
    summary = []
    for name, df in results.items():
        m = performance_metrics(df["port_ret"])
        _, p = boot_dist[name]
        summary.append({
            "portfolio": name,
            "annual_return": m["annual"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["mdd"],
            "bootstrap_p": p,
            "pass": m["sharpe"] > 1.40 and p < 0.05,
        })
    pd.DataFrame(summary).to_csv(MODEL_DIR / "portfolio_backtest_summary.csv", index=False)
    print("\n结果保存: models/portfolio_backtest_summary.csv")


if __name__ == "__main__":
    main()
