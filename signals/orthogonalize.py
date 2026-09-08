import pandas as pd
import numpy as np

def cross_sectional_orthogonalize(target_factor: pd.DataFrame, base_factors: dict) -> pd.DataFrame:
    """
    横截面施密特正交化 (Cross-Sectional Orthogonalization)
    针对每个截面 (每一天)，用目标因子对一组基准因子做 OLS 回归，返回残差 (Residual) 作为纯净因子。
    
    参数:
    target_factor (pd.DataFrame): 目标被净化因子矩阵 (index: 日期, columns: 股票)
    base_factors (dict): 基准因子群，key 为因子名，value 为 pd.DataFrame (如 {'alpha012': df1, 'alpha055': df2})
    
    返回:
    pd.DataFrame: 净化后的残差因子，形状与 target_factor 相同。
    """
    if not base_factors:
        return target_factor.copy()
        
    dates = target_factor.index
    columns = target_factor.columns
    
    residual_df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    
    # 对齐所有的基础因子矩阵
    base_dfs = []
    for k, v in base_factors.items():
        base_dfs.append(v.reindex(index=dates, columns=columns))
        
    for date in dates:
        y = target_factor.loc[date].values
        x_list = [df.loc[date].values for df in base_dfs]
        
        X = np.column_stack(x_list)
        
        # 严格过滤缺失值，确保只有因变量和自变量都非空的标的进入截面回归
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        
        if valid_mask.sum() > X.shape[1] + 1: # 自由度检查
            # 增加常数项列
            X_valid = np.column_stack([np.ones(valid_mask.sum()), X[valid_mask]])
            y_valid = y[valid_mask]
            
            try:
                # OLS 回归: Y = X * beta + epsilon
                beta, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                resid = y_valid - X_valid @ beta
                
                # 提取残差重新拼回原长度的截面
                res = np.full(len(y), np.nan)
                res[valid_mask] = resid
                residual_df.loc[date] = res
            except Exception:
                # 发生截面奇异矩阵等异常时，当天抛弃
                pass
                
    return residual_df
