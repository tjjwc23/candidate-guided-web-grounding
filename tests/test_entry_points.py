"""入口模块的安全约束：`--help` 必须只打印帮助，不能执行工作。

2026-09-12 出现过两次同一类事故：对没有 argparse 的入口执行 `python -m ... --help`，
模块直接开始干活（一次触发了 vLLM decode 分析，一次触发了 GPU SFT 训练流水线）。
本文件固定一条通用规则：任何带 `if __name__ == '__main__':` 的模块都必须 import
argparse，避免 `python -m ... --help` 意外执行训练或评测。
"""

import ast
from pathlib import Path

STEPENGINE = Path(__file__).resolve().parents[1]
SRC = STEPENGINE / "src"


def _entry_modules():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        has_main_block = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", None) == "__name__"
            for node in tree.body
        )
        if has_main_block:
            yield path, tree


def test_every_entry_module_parses_arguments():
    missing = []
    for path, tree in _entry_modules():
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        if "argparse" not in imported and path.name != "__main__.py":
            missing.append(str(path.relative_to(STEPENGINE)))
    assert not missing, f"这些入口没有 argparse，--help 会直接执行工作: {missing}"


import pytest
from cli import COMMANDS, main


@pytest.mark.parametrize('command', sorted(COMMANDS))
def test_registered_command_help_is_safe_and_restores_arguments(command):
    import sys
    previous = sys.argv
    with pytest.raises(SystemExit) as result:
        main([*command.split(), '--help'])
    assert result.value.code == 0 and sys.argv is previous
