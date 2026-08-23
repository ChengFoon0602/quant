"""
viz/plotting.py — 标准化量化研究报告学术可视化库。

遵循学术规范与研究报告标准：
  - 中英混排支持，配置 SimHei / Microsoft YaHei / Arial
  - 自动网格线与高对比度调色板
  - 关键数值、t 统计量、样本量与置信区间显式标注
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 配置中文字体与负号
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 300


def plot_equity_curve(
    curves: Dict[str, pd.Series],
    title: str = "策略累计净值与回撤对比",
    log_scale: bool = False,
    benchmark_key: Optional[str] = "基准",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """绘制累计净值曲线 + 下方水下回撤 (Underwater Drawdown) 面积图。

    Parameters
    ----------
    curves : Dict[str, pd.Series]
        {曲线名称: 累计净值序列(cum)} 字典。
    title : str, default "策略累计净值与回撤对比"
        图表主标题。
    log_scale : bool, default False
        是否在主净值图上使用对数坐标。
    benchmark_key : Optional[str], default "基准"
        基准曲线名称（用于区分主策略与基准样式）。
    save_path : Optional[Union[str, Path]], default None
        若提供则保存为图片。

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(curves))))
    main_key = list(curves.keys())[0]

    for idx, (name, cum) in enumerate(curves.items()):
        s = cum.dropna()
        if s.empty:
            continue
        is_bench = (benchmark_key is not None and benchmark_key in name)
        lw = 1.5 if is_bench else 2.0
        ls = "--" if is_bench else "-"
        alpha = 0.7 if is_bench else 1.0
        color = "#7f7f7f" if is_bench else colors[idx]

        ax1.plot(s.index, s.values, label=name, lw=lw, ls=ls, alpha=alpha, color=color)

        # 在副图绘制第一条主策略的回撤
        if idx == 0:
            peak = s.cummax()
            dd = (s - peak) / peak
            ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.3, label=f"{name} 回撤")
            ax2.plot(dd.index, dd.values, color=color, lw=1.0)

    if log_scale:
        ax1.set_yscale("log")

    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Wealth", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_crowding_timeseries(
    indicators_df: pd.DataFrame,
    composite_col: str = "composite_z",
    events: Optional[List[Tuple[pd.Timestamp, float]]] = None,
    title: str = "风格因子拥挤度时序与极端事件",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """绘制因子拥挤度时序图（C1-C4 及综合拥挤度 Z-score）。

    Parameters
    ----------
    indicators_df : pd.DataFrame
        包含 C1_extreme_exposure, C2_style_homogeneity, C3_turnover_crowding, C4_return_spike
        以及 composite_col 的时序数据。
    composite_col : str, default "composite_z"
        综合拥挤度列名。
    events : Optional[List[Tuple[pd.Timestamp, float]]], default None
        极端事件列表 [(日期, 拥挤度值)]。
    title : str
        主标题。
    save_path : Optional[Union[str, Path]], default None
        保存路径。

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.5]}
    )

    # 上子图：四大独立拥挤度维度
    sub_cols = [
        ("C1_extreme_exposure", "C1 极端暴露", "#1f77b4"),
        ("C2_style_homogeneity", "C2 风格同质化", "#ff7f0e"),
        ("C3_turnover_crowding", "C3 换手拥挤", "#2ca02c"),
        ("C4_return_spike", "C4 收益波动尖峰", "#d62728"),
    ]

    for col, label, color in sub_cols:
        if col in indicators_df.columns:
            s = indicators_df[col].dropna()
            # 标准化到 [0, 1] 方便同屏直观对比走势
            norm_s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            ax1.plot(norm_s.index, norm_s.values, label=label, color=color, lw=1.5, alpha=0.85)

    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Normalized Index (0~1)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)

    # 下子图：综合拥挤度 Z-score 与阈值
    if composite_col in indicators_df.columns:
        comp = indicators_df[composite_col].dropna()
        ax2.plot(comp.index, comp.values, label="综合拥挤度 (Composite Z)", color="#333333", lw=1.8)
        ax2.axhline(0.0, color="gray", linestyle=":", lw=1.0)

        # 90% 分位数线
        q90 = comp.quantile(0.90)
        ax2.axhline(q90, color="red", linestyle="--", lw=1.2, label=f"90% 极端阈值 (Z={q90:.2f})")

        # 标记极端事件
        if events:
            for edate, evalue in events:
                if edate in comp.index or (comp.index.min() <= edate <= comp.index.max()):
                    ax2.axvline(edate, color="red", linestyle="-.", alpha=0.6, lw=1.0)
                    ax2.scatter([edate], [evalue], color="red", s=40, zorder=5)

    ax2.set_ylabel("Composite Z-Score", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_stratified_returns(
    bucket_returns: Union[pd.Series, Dict[str, float]],
    t_stats: Optional[Dict[str, float]] = None,
    n_obs: Optional[Dict[str, int]] = None,
    title: str = "分层条件因子多空收益",
    xlabel: str = "拥挤度分组",
    ylabel: str = "未来月均收益 (%)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """绘制分层条件收益条形图，柱上方自动标注数值与 (t=..., n=...)。

    Parameters
    ----------
    bucket_returns : pd.Series or Dict[str, float]
        分组收益率（例如 {'Q1(低拥挤)': 0.34, 'Q2': 0.15, ...}）。
    t_stats : Optional[Dict[str, float]]
        各组对应的 t 统计量。
    n_obs : Optional[Dict[str, int]]
        各组对应的样本数。
    title : str
        图表标题。
    save_path : Optional[Union[str, Path]], default None
        保存路径。

    Returns
    -------
    plt.Figure
    """
    if isinstance(bucket_returns, dict):
        s = pd.Series(bucket_returns)
    else:
        s = bucket_returns.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in s.values]
    bars = ax.bar(s.index, s.values, color=colors, alpha=0.8, width=0.55, edgecolor="black", lw=0.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 柱上方数值与统计量标注
    y_range = max(abs(s.min()), abs(s.max())) if len(s) > 0 else 1.0
    offset = y_range * 0.05

    for bar, (name, val) in zip(bars, s.items()):
        height = bar.get_height()
        t_str = f"t={t_stats[name]:.2f}" if (t_stats and name in t_stats) else ""
        n_str = f"n={n_obs[name]}" if (n_obs and name in n_obs) else ""

        ann_parts = [f"{val:+.3f}%"]
        if t_str or n_str:
            ann_parts.append(f"({', '.join(filter(None, [t_str, n_str]))})")
        ann_text = "\n".join(ann_parts)

        va = "bottom" if height >= 0 else "top"
        y_pos = height + (offset if height >= 0 else -offset)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            ann_text,
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="medium",
        )

    # 留出上下空间放文本
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - offset * 2, ylim[1] + offset * 2)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bootstrap_distribution(
    boot_sharpes: Union[np.ndarray, pd.Series],
    observed_sharpe: float,
    p_value: float,
    ci_low: float,
    ci_high: float,
    title: str = "Bootstrap 夏普比率抽样分布",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """绘制 Bootstrap 夏普比率抽样分布直方图，标注观察值、置信区间与 p-value。

    Parameters
    ----------
    boot_sharpes : np.ndarray or pd.Series
        Bootstrap 重采样夏普数组。
    observed_sharpe : float
        样本观察到的夏普比率。
    p_value : float
        统计检验 p 值。
    ci_low : float
        95% 置信区间下界。
    ci_high : float
        95% 置信区间上界。
    title : str
        图表标题。
    save_path : Optional[Union[str, Path]]
        保存路径。

    Returns
    -------
    plt.Figure
    """
    arr = np.asarray(boot_sharpes)
    fig, ax = plt.subplots(figsize=(8, 5))

    # 直方图
    n, bins, patches = ax.hist(
        arr, bins=40, density=True, color="#4682b4", alpha=0.6, edgecolor="white", lw=0.5
    )

    # 0 基准线
    ax.axvline(0.0, color="gray", linestyle=":", lw=1.2, label="H0 基准 (SR=0)")

    # 实际观察值
    ax.axvline(
        observed_sharpe,
        color="#d62728",
        linestyle="-",
        lw=2.0,
        label=f"实际夏普 (SR={observed_sharpe:.3f}, p={p_value:.4f})",
    )

    # 95% 置信区间
    ax.axvline(ci_low, color="#2ca02c", linestyle="--", lw=1.2)
    ax.axvline(ci_high, color="#2ca02c", linestyle="--", lw=1.2, label=f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    ax.axvspan(ci_low, ci_high, color="#2ca02c", alpha=0.1)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
