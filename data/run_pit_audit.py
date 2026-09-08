import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pit_auditor import audit_pit_leakage
from data.akshare_fundamental_fetcher import fetch_stock_financial, _report_period_to_pit, SINA_TO_FACTOR

def run_audit():
    # 1. 抽取几只代表性股票的缓存数据作为对齐后的矩阵
    cache_path = os.path.join("D:\\桌面文件\\quant\\data\\cache_fundamental", "roeAvg.csv")
    if not os.path.exists(cache_path):
        print("未找到 roeAvg.csv 缓存，跳过审计。")
        return
        
    print("加载缓存矩阵...")
    aligned_matrix = pd.read_csv(cache_path, parse_dates=['date'], index_col='date', low_memory=False)
    
    # 选几只股票进行审计 (比如 sz000001, sh600000 等，我们在矩阵里找几只有数据的)
    sample_stocks = [col for col in aligned_matrix.columns if '000001' in col or '600000' in col][:2]
    if not sample_stocks:
        sample_stocks = aligned_matrix.columns[:2].tolist()
        
    print(f"抽取审计样本: {sample_stocks}")
    aligned_matrix = aligned_matrix[sample_stocks]
    
    # 2. 模拟 raw_financials 抓取，作为“真相”
    raw_records = []
    print("抓取底层新浪财报数据做比对（真相）...")
    for sym in sample_stocks:
        try:
            df = fetch_stock_financial(sym)
            if df is None or df.empty or "日期" not in df.columns:
                continue
            df["_period"] = df["日期"].astype(str)
            # 根据 SINA_TO_FACTOR, 我们要找对应的 roeAvg
            # roeAvg 在字典里对应 "加权净资产收益率(%)" 或 "净资产收益率(%)"
            for col, factor in SINA_TO_FACTOR.items():
                if factor != "roeAvg":
                    continue
                if col not in df.columns:
                    continue
                for _, row in df.iterrows():
                    val = row[col]
                    if pd.isna(val): continue
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    
                    end_date = pd.Timestamp(row["_period"])
                    # 法定披露期 + 1 天作为我们的保守 ann_date
                    safe_ann_date = _report_period_to_pit(row["_period"])
                    
                    raw_records.append({
                        "stock_code": sym,
                        "ann_date": safe_ann_date,
                        "end_date": end_date,
                        "value": fv
                    })
        except Exception as e:
            print(f"抓取 {sym} 失败: {e}")
            
    raw_financials = pd.DataFrame(raw_records)
    
    if raw_financials.empty:
        print("无法构建原始财务表，退出审计。")
        return
        
    # 3. 运行核心审计逻辑
    report = audit_pit_leakage(
        raw_financials=raw_financials,
        aligned_matrix=aligned_matrix,
        stock_code_col='stock_code',
        ann_date_col='ann_date',
        end_date_col='end_date',
        value_col='value'
    )
    
    print("\n" + "="*50)
    print("审计结论:", report['status'])
    print(f"检查数据点: {report['total_checks']}")
    print(f"发现泄露数: {report['leakage_count']} (占比 {report['leakage_rate']:.2%})")
    print("="*50)
    
    if report['leakage_count'] > 0:
        print("泄露样本:")
        print(report['sample_leakages'])
    else:
        print("完美通过！底层特征构建逻辑未发现未来函数跨期泄露。")

if __name__ == "__main__":
    run_audit()
