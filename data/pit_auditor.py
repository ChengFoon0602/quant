import pandas as pd
import numpy as np
import logging

def audit_pit_leakage(raw_financials: pd.DataFrame, 
                      aligned_matrix: pd.DataFrame, 
                      stock_code_col: str = 'stock_code',
                      ann_date_col: str = 'ann_date',
                      end_date_col: str = 'end_date',
                      value_col: str = 'value') -> dict:
    """
    PIT (Point-in-Time) 数据防前视 (Look-ahead bias) 审计工具。
    专门用于校验你在 build_pit_matrix.py 产出的特征矩阵是否发生了由于混淆“报告期(End Date)”
    与“实际公布日(Ann Date)”而导致的未来函数泄露。
    
    参数:
    raw_financials: 原始财务明细表 (长表格式)，必须包含股票代码、实际发布日、报告期和特征值
    aligned_matrix: 对齐后的特征矩阵 (宽表格式，index: 交易日，columns: 股票代码)
    
    返回:
    dict: 包含错误数、泄漏率和错误明细的审计报告
    """
    logging.info("开始执行 PIT 泄露审计...")
    
    # 将对齐的宽表逆透视为长表，方便比对
    # aligned_matrix 的 index 必须是 datetime，表示交易日
    melted = aligned_matrix.reset_index().melt(
        id_vars=aligned_matrix.index.name or 'index', 
        var_name=stock_code_col, 
        value_name='aligned_value'
    ).rename(columns={aligned_matrix.index.name or 'index': 'trade_date'})
    
    # 过滤掉矩阵中没有数据 (NaN) 的部分
    melted = melted.dropna(subset=['aligned_value'])
    
    # 为了验证 T 日使用的值，我们需要在原始表中找到截至 T 日能看到的最新的财报记录
    # 也就是满足 ann_date <= trade_date 且 ann_date 最大的那条记录
    
    # 按股票代码和公布日排序原始表
    raw_sorted = raw_financials.sort_values(by=[stock_code_col, ann_date_col, end_date_col])
    # 处理可能存在的同一天公布多份财报 (如同时公布一季报和年报)，取 end_date 最新的
    raw_dedup = raw_sorted.drop_duplicates(subset=[stock_code_col, ann_date_col], keep='last')
    
    # 使用 pandas 的 merge_asof 进行时点对齐:
    # 针对 melted 中的每一个 (stock_code, trade_date)，在 raw_dedup 中找到
    # stock_code 相同，且 ann_date <= trade_date 的最近一条记录
    
    melted = melted.sort_values('trade_date')
    raw_dedup = raw_dedup.sort_values(ann_date_col)
    
    merged = pd.merge_asof(
        melted,
        raw_dedup,
        left_on='trade_date',
        right_on=ann_date_col,
        by=stock_code_col,
        direction='backward'
    )
    
    # 校验点 1：对齐后的值，是否等于截面合法能看到的最新值？
    # 如果不等于，说明可能前向填充逻辑出错，或者误用了 end_date 作为关联依据
    # 注意浮点数比较误差
    merged['is_leakage'] = ~np.isclose(merged['aligned_value'], merged[value_col], equal_nan=True)
    
    leakage_cases = merged[merged['is_leakage']]
    total_checks = len(merged)
    leakage_count = len(leakage_cases)
    
    report = {
        'total_checks': total_checks,
        'leakage_count': leakage_count,
        'leakage_rate': leakage_count / total_checks if total_checks > 0 else 0.0,
        'status': 'FAIL' if leakage_count > 0 else 'PASS',
        'sample_leakages': leakage_cases.head(10) if leakage_count > 0 else None
    }
    
    if report['status'] == 'FAIL':
        logging.warning(f"⚠️ PIT 审计未通过！发现 {leakage_count} 处可能存在的时序泄露。")
        logging.warning("这通常是因为在拼接面板时，误用了财报的截止期(end_date)进行 ffill，而不是实际公告日(ann_date)。")
    else:
        logging.info("✅ PIT 审计通过，未发现因财报对齐导致的未来函数。")
        
    return report
