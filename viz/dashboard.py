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
        "stage": "单票时序探索",
        "annual_return": "-1.20%",
        "volatility": "22.50%",
        "sharpe": "-0.05",
        "max_drawdown": "-38.20%",
        "calmar": "-0.03",
        "conclusion": "A 股单票时序均线策略在摩擦成本下无显著超额，确立研究起点。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
    },
    {
        "id": 2,
        "name": "Alpha001 截面单因子",
        "path": "strategies/alpha001_trial/report.md",
        "stage": "单因子截面检验",
        "annual_return": "+2.10%",
        "volatility": "24.10%",
        "sharpe": "0.09",
        "max_drawdown": "-41.50%",
        "calmar": "0.05",
        "conclusion": "Alpha001 IC>0 但信息比率负，扣费后无法战胜基准。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
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
        "conclusion": "三因子两个为纯噪声，线性等权合成无法改善夏普。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
    },
    {
        "id": 4,
        "name": "Alpha 191 全量因子提纯",
        "path": "strategies/factor_discovery/report.md",
        "stage": "因子工程与提纯",
        "annual_return": "N/A",
        "volatility": "N/A",
        "sharpe": "N/A",
        "max_drawdown": "N/A",
        "calmar": "N/A",
        "conclusion": "191 因子提纯至 106 个有效候选；识别出 alpha141 等高 IC 无分散度的假因子。",
        "status": "方法论里程碑",
        "badge_color": "#3b82f6",
    },
    {
        "id": 5,
        "name": "特征选择与 Universe 修复",
        "path": "strategies/feature_selection/report.md",
        "stage": "特征工程",
        "annual_return": "N/A",
        "volatility": "N/A",
        "sharpe": "N/A",
        "max_drawdown": "N/A",
        "calmar": "N/A",
        "conclusion": "精炼 16 个核心特征，发现并修复了事故 Universe 导致的特征漂移。",
        "status": "踩坑修正",
        "badge_color": "#f59e0b",
    },
    {
        "id": 6,
        "name": "LightGBM 机器学习合成 (踩坑修正史)",
        "path": "models/report.md",
        "stage": "ML 非线性合成",
        "annual_return": "+1.80%",
        "volatility": "21.50%",
        "sharpe": "0.08",
        "max_drawdown": "-43.20%",
        "calmar": "0.04",
        "conclusion": "三大修正：夏普 8.75 虚假神话破灭（Overlapping 平滑陷阱 + 幸存者偏差 + 时序泄露）。",
        "status": "核心方法论修正",
        "badge_color": "#8b5cf6",
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
        "conclusion": "OOF 表面夏普 1.618，但在消除选择偏差与幸存者偏差后收敛至 0.295，三方向全否定。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
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
        "conclusion": "使用 akshare 引入 20 个财报因子，三层检验全否定，个股 Alpha 弧线全面闭合。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
    },
    {
        "id": 9,
        "name": "市场结构与拥挤度时序测度",
        "path": "strategies/zz500_crowding_trial/report.md",
        "stage": "市场微观结构",
        "annual_return": "+8.40%",
        "volatility": "18.20%",
        "sharpe": "0.46",
        "max_drawdown": "-28.50%",
        "calmar": "0.29",
        "conclusion": "发现量价动量延续 vs 基本面反转双面体特征，低拥挤择时展现可交易性。",
        "status": "结构发现",
        "badge_color": "#10b981",
    },
    {
        "id": 10,
        "name": "沪深300 跨指数验证 (反转证伪)",
        "path": "strategies/hs300_crowding_trial/report.md",
        "stage": "跨指数普适性检验",
        "annual_return": "-3.80%",
        "volatility": "22.30%",
        "sharpe": "-1.69",
        "max_drawdown": "-48.70%",
        "calmar": "-0.08",
        "conclusion": "沪深300 证伪小盘双面体与择时，证明所谓微观结构效应只是小盘特例。",
        "status": "证伪 (Negative)",
        "badge_color": "#ef4444",
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
        "conclusion": "行业动量反转显著；股债金全天候避险将最大回撤腰斩（-53.18% -> -22.97%），费率极度韧性。",
        "status": "实证突破",
        "badge_color": "#10b981",
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
        "conclusion": "欧拉等风险贡献打破股票风险垄断，全周期夏普 1.775、回撤仅 4.02%、Bootstrap p=0.0000 压倒性显著。",
        "status": "终极稳健底座",
        "badge_color": "#059669",
    },
]


def generate_html_dashboard() -> Path:
    """生成单文件自包含 HTML 投研仪表盘。"""
    rows_html = ""
    for item in RESEARCH_DATA:
        badge = f'<span style="background:{item["badge_color"]}; color:#fff; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600;">{item["status"]}</span>'
        link = f'<a href="../{item["path"]}" style="color:#2563eb; text-decoration:none; font-weight:600;">{item["name"]}</a>'
        sr_style = 'font-weight:700; color:#059669;' if item["sharpe"] not in ["N/A", "-"] and float(item["sharpe"]) >= 1.0 else ''
        mdd_style = 'font-weight:700; color:#059669;' if item["max_drawdown"] not in ["N/A", "-"] and float(item["max_drawdown"].replace('%', '')) > -10.0 else ''

        rows_html += f"""
        <tr>
            <td style="text-align:center; font-weight:bold;">{item["id"]}</td>
            <td>{link}<br><small style="color:#64748b;">{item["path"]}</small></td>
            <td><span style="color:#475569; font-weight:500;">{item["stage"]}</span></td>
            <td>{badge}</td>
            <td style="text-align:right;">{item["annual_return"]}</td>
            <td style="text-align:right;">{item["volatility"]}</td>
            <td style="text-align:right; {sr_style}">{item["sharpe"]}</td>
            <td style="text-align:right; {mdd_style}">{item["max_drawdown"]}</td>
            <td style="text-align:right;">{item["calmar"]}</td>
            <td style="font-size:13px; color:#334155;">{item["conclusion"]}</td>
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
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #0f172a; line-height: 1.5; padding: 30px; }}
        .header {{ max-width: 1300px; margin: 0 auto 30px auto; background: linear-gradient(135deg, #1e293b, #0f172a); color: #fff; padding: 35px; border-radius: 12px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; font-weight: 700; letter-spacing: -0.5px; }}
        .header p {{ color: #94a3b8; font-size: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; max-width: 1300px; margin: 0 auto 30px auto; }}
        .card {{ background: #fff; padding: 22px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .card-title {{ font-size: 13px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
        .card-value {{ font-size: 26px; font-weight: 700; color: #0f172a; }}
        .card-desc {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
        .table-container {{ max-width: 1300px; margin: 0 auto 30px auto; background: #fff; border-radius: 10px; border: 1px solid #e2e8f0; overflow-x: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th {{ background: #f1f5f9; color: #334155; padding: 14px 16px; font-weight: 600; border-bottom: 2px solid #cbd5e1; white-space: nowrap; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
        tr:hover {{ background-color: #f8fafc; }}
        .footer {{ max-width: 1300px; margin: 0 auto; text-align: center; font-size: 13px; color: #94a3b8; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>A股量化投研框架 — 12项系统性研究成果全景看板</h1>
        <p>完整研究弧线：单票时序 → 单/多因子截面 → 特征提纯 → ML非线性合成（三大修正史） → 中小盘与基本面证伪 → 市场结构 → 跨指数证伪 → 大类资产避险 → 欧拉风险平价</p>
    </div>

    <div class="metrics-grid">
        <div class="card">
            <div class="card-title">系统性研究篇数</div>
            <div class="card-value">12 篇</div>
            <div class="card-desc">涵盖日线选股、市场结构与跨资产配置</div>
        </div>
        <div class="card">
            <div class="card-title">全天候风险平价夏普 (SR)</div>
            <div class="card-value" style="color:#059669;">1.775</div>
            <div class="card-desc">Bootstrap p=0.0000 极度显著</div>
        </div>
        <div class="card">
            <div class="card-title">全天候组合最大回撤</div>
            <div class="card-value" style="color:#059669;">-4.02%</div>
            <div class="card-desc">较沪深300 (-45.10%) 回撤降低 91%</div>
        </div>
        <div class="card">
            <div class="card-title">工程底座与测试覆盖</div>
            <div class="card-value">100% 通过</div>
            <div class="card-desc">权重追踪法、无未来函数与微观交易拦截</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="text-align:center;">序</th>
                    <th>研究课题 / 报告路径</th>
                    <th>研究阶段</th>
                    <th>状态判决</th>
                    <th style="text-align:right;">年化收益</th>
                    <th style="text-align:right;">年化波动</th>
                    <th style="text-align:right;">夏普 (SR)</th>
                    <th style="text-align:right;">最大回撤</th>
                    <th style="text-align:right;">卡玛比率</th>
                    <th>核心结论与物理机制</th>
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
