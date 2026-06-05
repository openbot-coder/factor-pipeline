"""
factor-pipeline CLI — 因子检测流水线命令行入口

Usage:
    factor-pipeline run                  # 运行完整流水线
    factor-pipeline factors             # 列出可用因子
    factor-pipeline factor <name>       # 运行单个因子
    factor-pipeline doc <name>          # 因子公式说明
"""
from cli.main import main

__all__ = ["main"]
