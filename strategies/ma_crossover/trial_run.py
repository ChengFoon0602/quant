"""
trial_run.py — 端到端验证：均线交叉策略 + 向量化回测 + 可视化。

策略逻辑：快线上穿慢线 → 买入，下穿 → 卖出。单票、全仓进出。
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily

# ── 参数 ─────────────────────────────────────────────────
SYMBOL = "000001"        # 平安银行
FAST, SLOW = 5, 20       # 均线窗口
COMMISSION = 0.0003      # 单边手续费 + 印花税
START_CASH = 100_000     # 初始资金

# ── 1. 加载数据 ──────────────────────────────────────────
df = load_daily(SYMBOL)
if df is None:
    raise RuntimeError(f"{SYMBOL} 无缓存数据，先运行 python bootstrap.py")

close = df["close"].astype(float)

# ── 2. 策略信号（向量化） ─────────────────────────────────
fast_ma = close.rolling(FAST).mean()
slow_ma = close.rolling(SLOW).mean()

# 信号: 1=持仓, 0=空仓
signal = (fast_ma > slow_ma).astype(int)
# 信号变化日：快慢线交叉
cross_up = (signal.diff() == 1)     # 金叉 → 买入
cross_down = (signal.diff() == -1)  # 死叉 → 卖出

# 持仓信号前移一天：当日收盘信号决定次日开盘操作
position = signal.shift(1).fillna(0).astype(int)

# ── 3. 向量化回测 ─────────────────────────────────────────
daily_ret = close.pct_change()                     # 日收益率
strategy_ret = position * daily_ret                 # 持仓日的策略收益

# 交易成本：只在换仓日扣除
trades = position.diff().abs()                      # |Δpos| = 1 表示当日有交易
strategy_ret_net = strategy_ret - trades * COMMISSION

# 累计净值
equity = (1 + strategy_ret_net).cumprod() * START_CASH
benchmark = (1 + daily_ret).cumprod() * START_CASH   # 买入持有基准

# ── 4. 绩效指标 ──────────────────────────────────────────
total_return = equity.iloc[-1] / START_CASH - 1
bm_return = benchmark.iloc[-1] / START_CASH - 1
n_years = (df.index[-1] - df.index[0]).days / 365.25
ann_return = (1 + total_return) ** (1 / n_years) - 1

# 最大回撤
dd = equity / equity.cummax() - 1
max_dd = dd.min()

# 夏普比率（假设无风险利率 2%）
excess = strategy_ret_net - 0.02 / 252
sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

# 胜率/盈亏比
trade_dates = trades[trades > 0].index
win_rate = (strategy_ret_net[trade_dates] > 0).mean() if len(trade_dates) > 0 else 0

n_trades = int(trades.sum())
print(f"品种: {SYMBOL}  |  {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"均线: MA{FAST} / MA{SLOW}")
print(f"交易次数: {n_trades}")
print(f"年化收益: {ann_return*100:.1f}%  (基准 {bm_return*100:.1f}%)")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.1f}%")
print(f"胜率: {win_rate*100:.1f}%")

# ── 5. 可视化 ────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})

# 图1: 价格 + 均线 + 交易信号
ax = axes[0]
ax.plot(df.index, close, color="#333", linewidth=0.8, alpha=0.7, label="close")
ax.plot(df.index, fast_ma, color="#2196F3", linewidth=0.8, label=f"MA{FAST}")
ax.plot(df.index, slow_ma, color="#FF5722", linewidth=0.8, label=f"MA{SLOW}")
# 标记金叉/死叉
buy_idx = df.index[cross_up.values]
sell_idx = df.index[cross_down.values]
ax.scatter(buy_idx, close.loc[buy_idx], marker="^", c="#4CAF50", s=40, alpha=0.8, label="金叉")
ax.scatter(sell_idx, close.loc[sell_idx], marker="v", c="#F44336", s=40, alpha=0.8, label="死叉")
ax.set_ylabel("Price (¥)")
ax.legend(loc="upper left", fontsize=8, ncol=5)
ax.grid(True, alpha=0.3)
ax.set_title(f"{SYMBOL}  MA{FAST}/{SLOW} 均线交叉策略")

# 图2: 权益曲线
ax = axes[1]
ax.plot(df.index, equity, color="#4CAF50", linewidth=0.8, label="策略")
ax.plot(df.index, benchmark, color="#999", linewidth=0.6, alpha=0.6, label="买入持有")
ax.fill_between(df.index, equity, benchmark, where=(equity >= benchmark),
                color="#4CAF50", alpha=0.1)
ax.fill_between(df.index, equity, benchmark, where=(equity < benchmark),
                color="#F44336", alpha=0.1)
ax.set_ylabel("Equity (¥)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# 图3: 回撤
ax = axes[2]
ax.fill_between(df.index, 0, dd * 100, color="#F44336", alpha=0.3, linewidth=0)
ax.plot(df.index, dd * 100, color="#F44336", linewidth=0.6)
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("trial_run.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存: trial_run.png")
