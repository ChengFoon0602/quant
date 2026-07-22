"""
bootstrap.py — 首次运行脚本：拉取沪深 300 成分股历史日线数据并缓存到本地。

用法: python bootstrap.py
"""

import sys
sys.path.insert(0, ".")

from data.fetcher import sync_index, cache_summary

# 拉取全部沪深 300 成分股日线，2010 年至今
sync_index(index_code="000300", start="20100101", end="20251231")

# 打印缓存概况
print("\n── 本地缓存概况 ──")
print(cache_summary().to_string(index=False))
