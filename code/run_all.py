# -*- coding: utf-8 -*-
"""
run_all.py —— 一键复现全流程
============================
按 NLP 测度构建五步流水线顺序执行全部脚本。

用法：
    python code/run_all.py
    python code/run_all.py 3          # 仅执行步骤 01~03
"""
from __future__ import annotations

import runpy
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("01_corpus_acquisition.py", "第一步 语料获取"),
    ("02_preprocessing.py", "第二步 文本预处理"),
    ("03_feature_extraction.py", "第三步 特征提取"),
    ("04_measure_construction.py", "第四步 指标构建"),
    ("05_reliability_validity.py", "第五步 信效度检验"),
    ("06_regression_analysis.py", "实证分析"),
]


def main() -> None:
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else len(STEPS)
    t0 = time.time()
    for i, (script, name) in enumerate(STEPS, 1):
        if i > upto:
            break
        print("=" * 68)
        print(f"[{i}/{len(STEPS)}] {name}  ->  {script}")
        print("=" * 68)
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE.parent))
        if r.returncode != 0:
            print(f"\n!! {script} 执行失败，流程中止")
            sys.exit(r.returncode)
        print()
    print(f"全流程完成，用时 {time.time() - t0:.1f} 秒。")


if __name__ == "__main__":
    main()
