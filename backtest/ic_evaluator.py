import pandas as pd
import numpy as np

def compute_rank_ic(factor_df: pd.DataFrame, fwd_ret_df: pd.DataFrame) -> pd.Series:
    """
    计算单周期截面 Rank IC
    
    参数:
    factor_df: t 时刻的因子暴露矩阵
    fwd_ret_df: 对齐到 t 时刻的未来收益矩阵 (例如 t+1 的单日收益，或者 t 到 t+N 的累计收益)
    
    返回:
    pd.Series: 索引为日期，值为每天的 Rank IC (Spearman correlation)
    """
    # corrwith 默认计算列，通过 axis=1 计算行 (截面)
    return factor_df.corrwith(fwd_ret_df, axis=1, method='spearman')

def compute_ic_decay(factor_df: pd.DataFrame, daily_ret_df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    计算因子 IC 的多周期衰减 (IC Decay)
    通过测试 t 时刻的因子暴露，与 t+1, t+2, ..., t+max_lag 每天的独立单日收益的 Rank IC。
    
    参数:
    factor_df: t 时刻因子矩阵
    daily_ret_df: 标的每日收益矩阵 (未 shift 的原本收益)
    max_lag: 观测的最大滞后天数
    
    返回:
    pd.DataFrame: 包含均值 IC (Mean IC)、IR (Information Ratio) 和正 IC 胜率 (Positive Rate) 的衰减报表
    """
    results = []
    
    for lag in range(1, max_lag + 1):
        # 收益矩阵向上平移 lag，使得 t 行对应的是原本的 t+lag 行（即未来收益）
        # 这里严格遵循铁律，确保 factor 不动，将未来的收益移过来对齐
        lagged_ret = daily_ret_df.shift(-lag)
        
        ic_series = compute_rank_ic(factor_df, lagged_ret)
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ir = mean_ic / std_ic if std_ic != 0 else np.nan
        pos_rate = (ic_series > 0).sum() / ic_series.count() if ic_series.count() > 0 else np.nan
        
        results.append({
            'Lag': lag,
            'Mean_IC': mean_ic,
            'IC_IR': ir,
            'Pos_Rate': pos_rate
        })
        
    return pd.DataFrame(results).set_index('Lag')
