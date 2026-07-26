"""
strategies/factor_discovery/plot_bootstrap_sweep.py — 汇总图：106 因子 Bootstrap 扫描结果。

用法:
    cd D:/桌面文件/quant/strategies/factor_discovery
    python plot_bootstrap_sweep.py
"""
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("bootstrap_full_sweep.csv")
df_block = pd.read_csv("bootstrap_block_sensitivity.csv")

tested = df.dropna(subset=["p_value"])
paper_tiger = df[df["eff_ratio"] < 0.5]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. p 值分布直方图
ax = axes[0, 0]
ax.hist(tested["p_value"], bins=20, color="#1f77b4", edgecolor="white", alpha=0.85)
ax.axvline(0.05, color="#d62728", linestyle="--", linewidth=1.5, label="α=0.05")
ax.set_xlabel("Bootstrap p-value"); ax.set_ylabel("因子数")
ax.set_title(f"106 因子 Bootstrap p 值分布（{len(tested)} 个可测，0 个 p<0.05）")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 2. LS_SR 点估计 vs p 值散点
ax = axes[0, 1]
ax.scatter(tested["ls_sr"], tested["p_value"], s=18, alpha=0.6, color="#2ca02c")
ax.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.2, label="α=0.05")
ax.set_xlabel("LS_SR 点估计（方向对齐后）"); ax.set_ylabel("Bootstrap p-value")
ax.set_title("点估计越高不代表越显著——p 值几乎不随 SR 变化")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 3. 截面有效比分布（纸老虎占比）
ax = axes[1, 0]
eff = df["eff_ratio"].dropna() * 100
ax.hist(eff, bins=20, color="#ff7f0e", edgecolor="white", alpha=0.85)
ax.axvline(50, color="#d62728", linestyle="--", linewidth=1.5, label="50% 门槛")
n_tiger = (df["eff_ratio"] < 0.5).sum()
ax.set_xlabel("截面有效比 (%)"); ax.set_ylabel("因子数")
ax.set_title(f"截面有效比分布——{n_tiger}/106 个是纸老虎（<50%）")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 4. block_size 敏感性
ax = axes[1, 1]
ax.plot(df_block["block_size"], df_block["p_value"], "o-", color="#9467bd", linewidth=1.5, markersize=8)
ax.fill_between(df_block["block_size"], df_block["p_value"].min(), df_block["p_value"].max(),
                alpha=0.1, color="#9467bd")
ax.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.2, label="α=0.05")
ax.set_xlabel("block_size（交易日）"); ax.set_ylabel("Bootstrap p-value（alpha001）")
ax.set_title(f"block_size 敏感性：p 值稳定在 [{df_block['p_value'].min():.3f}, {df_block['p_value'].max():.3f}]")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/bootstrap_sweep_summary.png", dpi=150)
plt.close()
print("图表保存: figures/bootstrap_sweep_summary.png")
