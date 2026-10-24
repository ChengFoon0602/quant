"""扫描 factors.py 中的 MAX/MIN 用法，生成/校验公式语义登记表。

为什么要这张表：
  国泰君安 Alpha191 原文里 `MAX(A, B)` 语义双关——
  既可能是「逐元素取大」（B 是常数或序列），也可能是「B 日窗口滚动最大」。
  实现必须在每处明确选择一种，否则同一份代码里两种解读会混用，
  且后人无法判断哪处是有意、哪处是笔误（审计发现 alpha010 的
  `np.maximum(val, 5)` 就是笔误：把窗口 5 当成了逐元素下界）。

用法:
    python tests/audit_alpha191_semantics.py            # 生成表格
    python tests/audit_alpha191_semantics.py --check    # 校验：新用法未登记则失败
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

FACTORS_PY = PROJECT / "signals" / "alpha191" / "factors.py"
TABLE = PROJECT / "docs" / "alpha191_公式语义表.csv"

FUNC_RE = re.compile(r"^def (factor_(\d{3}))")

# 每类实现的判定依据（写入 CSV 的 note 列）
NOTE = {
    "elementwise_zero": "正负部截断 max(x,0)/min(x,0)，原文语义明确",
    "elementwise_const": "第二参数为正常数，疑似把窗口当逐元素下界——需人工确认",
    "elementwise": "第二参数为序列或无法判定，按逐元素处理",
    "rolling_window": "外层窗口算子：n 日滚动最大/最小（研报 MAX(X,n) 的另一解读）",
    "formula_text": "研报原文（docstring，非实现）",
}
HIT_RE = re.compile(r"(\bMAX\(|\bMIN\(|np\.maximum|np\.minimum|np\.fmax|np\.fmin)")


def _classify(code: str) -> tuple[str, str]:
    """返回 (impl_semantics, risk)。用 AST 判断 numpy 逐元素调用的第二参数。"""
    if re.search(r"(?<![a-z_])MAX\(|(?<![a-z_])MIN\(", code):
        return "rolling_window", "ok"

    try:
        tree = ast.parse("def _f():\n    " + code)
    except SyntaxError:
        # 多行表达式被截断（如 _where 嵌套），无法判定；不报假警报
        return "elementwise", "ok"

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("maximum", "minimum", "fmax", "fmin")
                and len(node.args) == 2):
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, (int, float)):
                if second.value == 0:
                    return "elementwise_zero", "ok"      # 正负部截断，语义明确
                return "elementwise_const", "SUSPECT"    # 常数 > 0：疑似把窗口当常数
    return "elementwise", "ok"


def scan() -> list[dict]:
    lines = FACTORS_PY.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    current = None
    for i, line in enumerate(lines, 1):
        m = FUNC_RE.match(line)
        if m:
            current = (m.group(1), int(m.group(2)))
            continue
        if current is None or not HIT_RE.search(line):
            continue
        code = line.strip()
        if code.startswith("#"):
            continue
        if '"""' in code:
            # docstring：记录研报原文，不代表实现语义
            sem, risk = "formula_text", "ok"
        else:
            sem, risk = _classify(code)
        rows.append({
            "factor_id": current[0],
            "line": i,
            "code": code,
            "impl_semantics": sem,
            "risk": risk,
            "note": NOTE[sem],
        })
    return rows


def write_table(rows: list[dict]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["factor_id", "line", "code", "impl_semantics", "risk", "note"])
        for r in rows:
            w.writerow([r["factor_id"], r["line"], r["code"],
                        r["impl_semantics"], r["risk"], r.get("note", "")])
    print(f"已写入 {TABLE}（{len(rows)} 条）")

    suspects = [r for r in rows if r["risk"] == "SUSPECT"]
    if suspects:
        print(f"\n⚠ 疑似把窗口参数当常数下界（{len(suspects)} 处，需人工确认）:")
        for r in suspects:
            print(f"  {r['factor_id']}:{r['line']}  {r['code']}")


def check(rows: list[dict]) -> int:
    if not TABLE.exists():
        print(f"缺少 {TABLE}，先不带 --check 运行一次")
        return 1
    with TABLE.open(encoding="utf-8-sig") as f:
        registered = {(r["factor_id"], r["code"]) for r in csv.DictReader(f)}
    missing = [r for r in rows if (r["factor_id"], r["code"]) not in registered]
    if missing:
        print(f"❌ {len(missing)} 处 MAX/MIN 用法未登记：")
        for r in missing:
            print(f"  {r['factor_id']}:{r['line']}  {r['code']}")
        return 1
    print(f"✅ 全部 {len(rows)} 处 MAX/MIN 用法已登记")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows = scan()
    return check(rows) if args.check else (write_table(rows) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
