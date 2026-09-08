import pandas as pd
import numpy as np
import os
import sys

# 把上级目录加入 sys.path 方便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.orthogonalize import cross_sectional_orthogonalize

def test_orthogonality():
    """测试正交化后的残差与基准因子相关性是否为 0"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10)
    stocks = ['AAPL', 'MSFT', 'GOOG']
    
    # 构建基准因子 (例如 Size)
    base_factor = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=stocks)
    
    # 构建目标因子 (故意使其与 base_factor 有强相关性)
    noise = pd.DataFrame(np.random.randn(10, 3) * 0.1, index=dates, columns=stocks)
    target_factor = base_factor * 2 + noise
    
    # 正交化
    base_dict = {'size': base_factor}
    residual_factor = cross_sectional_orthogonalize(target_factor, base_dict)
    
    # 验证每一个截面，残差与基准的相关系数
    for date in dates:
        x = base_factor.loc[date].values
        y_res = residual_factor.loc[date].values
        
        # 相关系数
        corr = np.corrcoef(x, y_res)[0, 1]
        assert np.abs(corr) < 1e-10, f"截面 {date} 相关性未完全剥离: corr={corr}"
        
    print("✅ 正交化核心算法测试通过：残差与基底特征严格无相关性。")

def test_look_ahead_bias():
    """测试正交化是否引入了未来函数"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=5)
    stocks = ['S1', 'S2', 'S3']
    
    # 构建基准和目标
    base = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)
    target = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)
    
    # 分两次算：
    # 1. 算全量5天的正交化
    res_full = cross_sectional_orthogonalize(target, {'b': base})
    
    # 2. 算前3天的正交化
    res_partial = cross_sectional_orthogonalize(target.iloc[:3], {'b': base.iloc[:3]})
    
    # 判断：res_partial 的前3天，必须与 res_full 的前3天完全一致
    # 只要有微小差异，就说明 t 日的正交化受到了 t 日以后数据的影响
    diff = np.abs(res_partial.values - res_full.iloc[:3].values).max()
    assert diff < 1e-10, f"⚠️ 未来函数检查失败！不同长度数据输入的正交结果存在差异，差异最大值: {diff}"
    
    print("✅ 防前视 (Look-ahead bias) 测试通过：当前时刻的正交未污染未来数据。")

if __name__ == '__main__':
    test_orthogonality()
    test_look_ahead_bias()
