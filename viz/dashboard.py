"""
viz/dashboard.py — 全策略 12 篇研究报告交互式投研看板生成器。

自动汇集 12 项研究成果，生成独立、现代且自包含的 HTML 投研仪表盘 (dashboard.html)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_HTML = Path(__file__).parent / "dashboard.html"

RESEARCH_DATA = [
    {
        "id": 1,
        "name": "MA 均线时序交叉",
        "path": "strategies/ma_crossover/report.md",
        "stage": "单票时序",
        "annual_return": "-1.20%",
        "volatility": "22.50%",
        "sharpe": "-0.05",
        "max_drawdown": "-38.20%",
        "calmar": "-0.03",
        "conclusion": "单票时序均线在 A 股真实摩擦下无显著超额，确立研究起点。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 2,
        "name": "Alpha001 截面单因子",
        "path": "strategies/alpha001_trial/report.md",
        "stage": "单因子截面",
        "annual_return": "+2.10%",
        "volatility": "24.10%",
        "sharpe": "0.09",
        "max_drawdown": "-41.50%",
        "calmar": "0.05",
        "conclusion": "Alpha001 秩 IC 虽然大于 0 但信息比率负，扣费后跑不赢基准。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 3,
        "name": "Alpha012/055/191 多因子合成",
        "path": "strategies/multi_factor_trial/alpha012_alpha055_alpha191/report.md",
        "stage": "多因子线性合成",
        "annual_return": "+3.40%",
        "volatility": "23.80%",
        "sharpe": "0.14",
        "max_drawdown": "-39.80%",
        "calmar": "0.09",
        "conclusion": "三因子两个为纯噪声，简单线性等权合成无法改善风险收益。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 4,
        "name": "Alpha 191 全量因子提纯",
        "path": "strategies/factor_discovery/report.md",
        "stage": "因子工程提纯",
        "annual_return": "—",
        "volatility": "—",
        "sharpe": "—",
        "max_drawdown": "—",
        "calmar": "—",
        "conclusion": "191 因子提纯至 106 个候选；识别出 alpha141 等高 IC 无分散度的假因子。",
        "status": "方法论基石",
        "type": "neutral",
    },
    {
        "id": 5,
        "name": "特征选择与 Universe 修复",
        "path": "strategies/feature_selection/report.md",
        "stage": "特征工程",
        "annual_return": "—",
        "volatility": "—",
        "sharpe": "—",
        "max_drawdown": "—",
        "calmar": "—",
        "conclusion": "精炼 16 个核心特征，发现并修复了事故 Universe 导致的特征漂移。",
        "status": "方法论修正",
        "type": "warning",
    },
    {
        "id": 6,
        "name": "LightGBM 机器学习非线性合成",
        "path": "models/report.md",
        "stage": "ML 非线性合成",
        "annual_return": "+1.80%",
        "volatility": "21.50%",
        "sharpe": "0.08",
        "max_drawdown": "-43.20%",
        "calmar": "0.04",
        "conclusion": "三大修正：夏普 8.75 虚假神话破灭（Overlapping 平滑陷阱 + 幸存者偏差）。",
        "status": "核心修正史",
        "type": "warning",
    },
    {
        "id": 7,
        "name": "中证500 PIT 选股实验",
        "path": "strategies/zz500_pit_trial/report.md",
        "stage": "中小盘量价探索",
        "annual_return": "+6.80%",
        "volatility": "23.10%",
        "sharpe": "0.295",
        "max_drawdown": "-35.40%",
        "calmar": "0.19",
        "conclusion": "OOF 表面夏普 1.618，但在消除选择偏差与幸存者偏差后收敛至 0.295。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 8,
        "name": "20 个基本面因子全面检验",
        "path": "strategies/zz500_fundamental_trial/report.md",
        "stage": "基本面信息源",
        "annual_return": "+1.50%",
        "volatility": "20.80%",
        "sharpe": "0.07",
        "max_drawdown": "-44.10%",
        "calmar": "0.03",
        "conclusion": "akshare 财报因子三层检验全否定，个股截面日线 Alpha 弧线全面闭合。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 9,
        "name": "市场结构与拥挤度时序测度",
        "path": "strategies/zz500_crowding_trial/report.md",
        "stage": "市场微观结构",
        "annual_return": "+8.40%",
        "volatility": "18.20%",
        "sharpe": "0.460",
        "max_drawdown": "-28.50%",
        "calmar": "0.29",
        "conclusion": "量价动量延续 vs 基本面反转双面体特征，低拥挤择时展现可交易性。",
        "status": "结构发现",
        "type": "success",
    },
    {
        "id": 10,
        "name": "沪深300 跨指数验证 (证伪反转)",
        "path": "strategies/hs300_crowding_trial/report.md",
        "stage": "跨指数普适性检验",
        "annual_return": "-3.80%",
        "volatility": "22.30%",
        "sharpe": "-1.690",
        "max_drawdown": "-48.70%",
        "calmar": "-0.08",
        "conclusion": "沪深300 证伪小盘双面体与择时，确证微观结构效应只是小盘特例。",
        "status": "证伪闭环",
        "type": "rejected",
    },
    {
        "id": 11,
        "name": "ETF 动量反转与股债金避险轮动",
        "path": "strategies/etf_momentum_crowding/report.md",
        "stage": "大类资产避险",
        "annual_return": "+6.23%",
        "volatility": "15.20%",
        "sharpe": "0.410",
        "max_drawdown": "-22.97%",
        "calmar": "0.271",
        "conclusion": "行业动量反转显著；股债金全天候避险将最大回撤腰斩（-53.18% -> -22.97%）。",
        "status": "实证突破",
        "type": "success",
    },
    {
        "id": 12,
        "name": "全天候多资产欧拉风险平价 (ERC)",
        "path": "strategies/all_weather_risk_parity/report.md",
        "stage": "大类资产配置终局",
        "annual_return": "+5.89%",
        "volatility": "3.32%",
        "sharpe": "1.775",
        "max_drawdown": "-4.02%",
        "calmar": "1.465",
        "conclusion": "欧拉等风险贡献消除股票风险霸权，夏普 1.775、回撤仅 4.02%、Bootstrap p=0.0000 压倒性显著。",
        "status": "终极稳健底座",
        "type": "primary",
    },
]


def generate_html_dashboard() -> Path:
    """生成单文件自包含 HTML 投研仪表盘。"""
    
    BADGE_STYLES = {
        "rejected": {
            "bg": "#f8fafc",
            "border": "#e2e8f0",
            "text": "#64748b",
            "dot": "#94a3b8",
        },
        "warning": {
            "bg": "#fffbeb",
            "border": "#fde68a",
            "text": "#92400e",
            "dot": "#f59e0b",
        },
        "success": {
            "bg": "#f0fdf4",
            "border": "#bbf7d0",
            "text": "#166534",
            "dot": "#22c55e",
        },
        "primary": {
            "bg": "#eff6ff",
            "border": "#bfdbfe",
            "text": "#1e40af",
            "dot": "#3b82f6",
        },
        "neutral": {
            "bg": "#f1f5f9",
            "border": "#cbd5e1",
            "text": "#334155",
            "dot": "#64748b",
        },
    }

    rows_html = ""
    for item in RESEARCH_DATA:
        st = BADGE_STYLES.get(item["type"], BADGE_STYLES["neutral"])
        badge = f'''<span style="display:inline-flex; align-items:center; gap:5px; background:{st["bg"]}; border:1px solid {st["border"]}; color:{st["text"]}; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:500; white-space:nowrap;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:{st["dot"]};"></span>
            {item["status"]}
        </span>'''
        
        link = f'<a href="../{item["path"]}" style="color:#0f172a; text-decoration:none; font-weight:600; transition: color 0.15s;" onmouseover="this.style.color=\'#2563eb\'" onmouseout="this.style.color=\'#0f172a\'">{item["name"]}</a>'
        
        sr_val = item["sharpe"]
        if sr_val not in ["—", "N/A", "-"]:
            sr_num = float(sr_val)
            if sr_num >= 1.0:
                sr_style = 'font-weight:700; color:#15803d;'
            elif sr_num < 0:
                sr_style = 'color:#94a3b8;'
            else:
                sr_style = 'color:#334155;'
        else:
            sr_style = 'color:#94a3b8;'

        mdd_val = item["max_drawdown"]
        if mdd_val not in ["—", "N/A", "-"]:
            mdd_num = float(mdd_val.replace('%', ''))
            mdd_style = 'font-weight:700; color:#15803d;' if mdd_num > -10.0 else 'color:#475569;'
        else:
            mdd_style = 'color:#94a3b8;'

        rows_html += f"""
        <tr>
            <td style="text-align:center; color:#94a3b8; font-variant-numeric:tabular-nums; font-size:13px;">{item["id"]:02d}</td>
            <td>
                {link}
                <div style="font-size:11px; color:#94a3b8; font-family:monospace; margin-top:2px;">{item["path"]}</div>
            </td>
            <td><span style="color:#475569; font-size:13px;">{item["stage"]}</span></td>
            <td style="text-align:center;">{badge}</td>
            <td style="text-align:right; font-variant-numeric:tabular-nums;">{item["annual_return"]}</td>
            <td style="text-align:right; font-variant-numeric:tabular-nums; color:#64748b;">{item["volatility"]}</td>
            <td style="text-align:right; font-variant-numeric:tabular-nums; {sr_style}">{item["sharpe"]}</td>
            <td style="text-align:right; font-variant-numeric:tabular-nums; {mdd_style}">{item["max_drawdown"]}</td>
            <td style="text-align:right; font-variant-numeric:tabular-nums; color:#64748b;">{item["calmar"]}</td>
            <td style="font-size:13px; color:#475569; line-height:1.45;">{item["conclusion"]}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股量化投研框架 — 12项研究成果全景看板</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8fafc; color: #0f172a; line-height: 1.5; padding: 32px 24px; }}
        .header {{ max-width: 1360px; margin: 0 auto 28px auto; background: #ffffff; border: 1px solid #e2e8f0; padding: 28px 32px; border-radius: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.04); }}
        .header h1 {{ font-size: 24px; margin-bottom: 6px; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }}
        .header p {{ color: #64748b; font-size: 14px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; max-width: 1360px; margin: 0 auto 28px auto; }}
        .card {{ background: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03); }}
        .card-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 6px; }}
        .card-value {{ font-size: 26px; font-weight: 700; color: #0f172a; }}
        .card-desc {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
        .table-container {{ max-width: 1360px; margin: 0 auto 28px auto; background: #ffffff; border-radius: 10px; border: 1px solid #e2e8f0; overflow-x: auto; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.04); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13.5px; }}
        th {{ background: #f8fafc; color: #475569; padding: 12px 14px; font-weight: 600; font-size: 12.5px; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background-color: #fafbfc; }}
        .footer {{ max-width: 1360px; margin: 0 auto; text-align: center; font-size: 12.5px; color: #94a3b8; padding-top: 16px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>A股量化投研框架 — 12 项系统性研究成果全景看板</h1>
        <p>演进弧线：单票时序 → 截面因子 → 特征提纯 → ML非线性合成（三大修正史） → 中小盘与基本面证伪 → 市场结构 → 跨指数证伪 → 大类资产避险 → 欧拉风险平价</p>
    </div>

    <div class="metrics-grid">
        <div class="card">
            <div class="card-title">系统性研究篇数</div>
            <div class="card-value">12 篇</div>
            <div class="card-desc">涵盖选股、市场结构与跨资产配置</div>
        </div>
        <div class="card">
            <div class="card-title">全天候风险平价夏普 (SR)</div>
            <div class="card-value" style="color:#15803d;">1.775</div>
            <div class="card-desc">Bootstrap p=0.0000 极度显著</div>
        </div>
        <div class="card">
            <div class="card-title">全天候组合最大回撤</div>
            <div class="card-value" style="color:#15803d;">-4.02%</div>
            <div class="card-desc">较沪深300 (-45.10%) 回撤降低 91%</div>
        </div>
        <div class="card">
            <div class="card-title">工程底座与测试覆盖</div>
            <div class="card-value">21 项全过</div>
            <div class="card-desc">权重追踪法、无未来函数与微观交易拦截</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="text-align:center; width:40px;">#</th>
                    <th style="min-width:180px;">研究课题 / 报告路径</th>
                    <th style="min-width:110px;">研究阶段</th>
                    <th style="text-align:center; min-width:110px;">状态判决</th>
                    <th style="text-align:right; min-width:80px;">年化收益</th>
                    <th style="text-align:right; min-width:80px;">年化波动</th>
                    <th style="text-align:right; min-width:80px;">夏普比率</th>
                    <th style="text-align:right; min-width:80px;">最大回撤</th>
                    <th style="text-align:right; min-width:70px;">卡玛比率</th>
                    <th style="min-width:320px;">核心结论与物理机制</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>A股量化投研框架 · Point-in-Time (PIT) 架构 · 欧拉风险平价 · 遵循奥卡姆剃刀与第一性原理</p>
    </div>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    return OUTPUT_HTML


if __name__ == "__main__":
    p = generate_html_dashboard()
    print(f"投研全景看板已成功生成: {p.resolve()}")
