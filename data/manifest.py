"""
data/manifest.py — 数据可复现性清单（Data Manifest）。

背景：
  数据缓存（data/cache/ 等）被 .gitignore 排除，不进入版本控制。
  这导致「回测结果可复现」的铁律在数据层失效——baostock/akshare 的历史数据
  若被修正、或成分股列表更新，重跑结果会漂移，且无法定位是哪一次抓取引入的。

方案：
  本模块为缓存数据生成一份清单 JSON（data/manifest.json），记录每只股票/文件的
  行数、日期范围、抓取时间、内容哈希（sha256），作为数据层的「快照锚点」。
  清单文件应纳入 git 版本控制（.gitignore 已单独放行）。

用法:
    from data.manifest import build_manifest, verify_manifest

    build_manifest()                       # 生成/更新 data/manifest.json
    report = verify_manifest()             # 校验当前缓存与清单是否一致
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

MANIFEST_PATH = Path(__file__).parent / "manifest.json"

# 需要纳入清单的缓存目录（相对于 data/ 目录）
TRACKED_DIRS = [
    "cache",
    "cache_fundamental",
    "cache_valuation",
    "cache_index",
    "cache_etf",
    "cache_meta",
]


def _sha256_of_file(path: Path) -> str:
    """计算文件内容的 sha256（分块读取，避免大文件一次性载入内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_stats(path: Path) -> Dict:
    """提取单个缓存文件的统计信息。"""
    stat = path.stat()
    entry: Dict = {
        "file": str(path.name),
        "size_bytes": stat.st_size,
        "mtime": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
        "sha256": _sha256_of_file(path),
    }

    # CSV 文件额外提取行数与日期范围
    if path.suffix == ".csv":
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            entry["rows"] = int(len(df))
            if "date" in df.columns:
                entry["start"] = str(pd.Timestamp(df["date"].min()).date())
                entry["end"] = str(pd.Timestamp(df["date"].max()).date())
        except Exception:
            # 非标准 CSV（如 json 缓存、industry.csv 等），跳过行数提取
            pass

    return entry


def build_manifest() -> Dict:
    """扫描所有跟踪目录，生成数据清单。

    Returns
        manifest dict（同时写入 data/manifest.json）
    """
    data_dir = Path(__file__).parent
    manifest: Dict = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "tracked_dirs": TRACKED_DIRS,
        "files": {},
    }

    total_files = 0
    for rel_dir in TRACKED_DIRS:
        d = data_dir / rel_dir
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            key = f"{rel_dir}/{p.name}"
            manifest["files"][key] = _file_stats(p)
            total_files += 1

    manifest["total_files"] = total_files

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"数据清单已生成: {MANIFEST_PATH}（{total_files} 个文件）")
    return manifest


def verify_manifest() -> Dict:
    """校验当前缓存与清单是否一致。

    Returns
        {'status': 'ok'|'drift'|'no_manifest',
         'drifted': [...], 'missing': [...], 'new': [...]}
    """
    if not MANIFEST_PATH.exists():
        return {"status": "no_manifest", "drifted": [], "missing": [], "new": []}

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    data_dir = Path(__file__).parent
    drifted: List[str] = []
    missing: List[str] = []
    new: List[str] = []

    # 1. 检查清单中的文件是否变化/丢失
    for key, meta in manifest["files"].items():
        p = data_dir / key
        if not p.exists():
            missing.append(key)
            continue
        current = _file_stats(p)
        if current["sha256"] != meta["sha256"]:
            drifted.append(key)

    # 2. 检查是否有清单未记录的新文件
    recorded = set(manifest["files"].keys())
    for rel_dir in TRACKED_DIRS:
        d = data_dir / rel_dir
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                key = f"{rel_dir}/{p.name}"
                if key not in recorded:
                    new.append(key)

    status = "ok" if not (drifted or missing or new) else "drift"
    return {
        "status": status,
        "drifted": drifted,
        "missing": missing,
        "new": new,
    }


if __name__ == "__main__":
    build_manifest()
    report = verify_manifest()
    print(f"\n校验状态: {report['status']}")
    if report["drifted"]:
        print(f"  漂移文件 ({len(report['drifted'])}):")
        for k in report["drifted"][:10]:
            print(f"    - {k}")
    if report["missing"]:
        print(f"  缺失文件 ({len(report['missing'])}):")
        for k in report["missing"][:10]:
            print(f"    - {k}")
    if report["new"]:
        print(f"  新增文件 ({len(report['new'])}):")
        for k in report["new"][:10]:
            print(f"    - {k}")
