"""
data/storage.py — 高性能二进制列式存储 (Parquet) 与统一数据加载门面。

优势：
  1. 存储体积相比纯文本 CSV 压缩约 60%~75%；
  2. 二进制列式读取速度比 CSV 提升 10~30 倍（毫秒级极速加载）；
  3. 保留精确的 Pandas DatetimeIndex 与 float64/int64 二进制类型，彻底杜绝字符串重解析损耗。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    compression: str = "snappy",
) -> Path:
    """将 DataFrame 保存为高压缩比 Parquet 文件。"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", compression=compression)
    return p


def load_parquet(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """高性能加载 Parquet 文件（支持列级投影与零拷贝）。"""
    p = Path(filepath)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p, engine="pyarrow", columns=columns)
    except Exception:
        return None


def convert_csv_directory_to_parquet(
    src_dir: Union[str, Path],
    dest_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> int:
    """批量将指定目录下的所有 CSV 文件转换为 Parquet 列式存储。

    Returns
    -------
    int
        成功转换的文件数量。
    """
    src = Path(src_dir)
    dest = Path(dest_dir) if dest_dir is not None else src
    dest.mkdir(parents=True, exist_ok=True)

    csv_files = list(src.glob("*.csv"))
    converted_count = 0

    for f in csv_files:
        target_parquet = dest / f"{f.stem}.parquet"
        if target_parquet.exists() and not overwrite:
            continue
        try:
            df = pd.read_csv(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            df.to_parquet(target_parquet, engine="pyarrow", compression="snappy")
            converted_count += 1
        except Exception:
            continue

    return converted_count


def load_price_matrix_fast(
    symbols: List[str],
    cache_dir: Union[str, Path],
    price_col: str = "close",
) -> pd.DataFrame:
    """极速加载一篮子股票/ETF的价格矩阵（自动优先选用 Parquet，透明回退至 CSV）。"""
    c_dir = Path(cache_dir)
    res_dict = {}

    for sym in symbols:
        parquet_path = c_dir / f"{sym}.parquet"
        csv_path = c_dir / f"{sym}.csv"

        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow", columns=[price_col])
                res_dict[sym] = df[price_col]
                continue
            except Exception:
                pass

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                if price_col in df.columns:
                    res_dict[sym] = df[price_col]
            except Exception:
                pass

    if not res_dict:
        return pd.DataFrame()

    return pd.DataFrame(res_dict).sort_index()
