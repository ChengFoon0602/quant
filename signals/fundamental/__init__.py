"""signals/fundamental — 基本面因子计算模块。

与 Alpha 191 纯价量因子的区别：
  - 数据源: baostock 季频财务指标（非日线 OHLCV）
  - 调仓频率: 季频（财报发布后更新），非日频
  - PIT: 法定截止日索引（保守方案），非实时可用
"""
