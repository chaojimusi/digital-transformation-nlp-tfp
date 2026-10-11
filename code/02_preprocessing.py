# -*- coding: utf-8 -*-
"""
02_preprocessing.py —— 第二步：文本预处理（Preprocessing）
==========================================================
对应论文"3.3 文本预处理"。

五步流水线（本脚本逐一实现）：
    ① 文档解析   —— 真实场景用 PyMuPDF 抽取 PDF 文本；本仓库语料已是纯文本
    ② 文本清洗   —— 去页眉页脚、页码、乱码与多余空白
    ③ 中文分词   —— jieba 精确模式 + 自定义词典（保证专有术语不被切碎）
    ④ 去停用词   —— 通用停用词 + 财经领域停用词 + 自定义停用词
    ⑤ 词形归一   —— 全角转半角、简称统一（如 AI → 人工智能）

输出：
    data/processed/mda_clean.csv        清洗后的文本与词数
    data/processed/top_words_overall.csv 全样本高频词表
"""
from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")
import nlp_utils

# ------------------------------------------------------------------
# 0. 初始化：自定义词典 + 三维停用词表
# ------------------------------------------------------------------
userdict = nlp_utils.build_userdict(cfg.PATH_DIGITAL_DICT, cfg.PATH_USERDICT)
nlp_utils.init_jieba(userdict)
STOPWORDS = nlp_utils.load_stopwords(
    cfg.PATH_STOP_GENERAL, cfg.PATH_STOP_FINANCE
)
print(f"[步骤2] 自定义词典 {sum(1 for _ in userdict.open(encoding='utf-8'))} 词；"
      f"停用词 {len(STOPWORDS)} 个")

# ------------------------------------------------------------------
# 1. 逐份处理
# ------------------------------------------------------------------
corpus = pd.read_csv(cfg.PATH_CORPUS, encoding="utf-8-sig")

records, counter = [], Counter()
for row in corpus.itertuples(index=False):
    t = nlp_utils.normalize_text(row.text)          # ⑤ 词形归一（先归一便于清洗）
    t = nlp_utils.clean_text(t)                     # ①② 解析 + 清洗
    toks = nlp_utils.tokenize(t)                    # ③ 中文分词
    toks = nlp_utils.remove_stopwords(toks, STOPWORDS)   # ④ 去停用词
    counter.update(toks)
    records.append({
        "firm_id": row.firm_id,
        "year": row.year,
        "text_clean": t,
        "n_chars": len(t),
        "n_tokens": len(toks),
        "n_sent": t.count("。") + t.count("！") + t.count("？"),
    })

clean = pd.DataFrame(records)
clean.to_csv(cfg.PATH_CLEAN, index=False, encoding="utf-8-sig")

top = pd.DataFrame(counter.most_common(300), columns=["word", "freq"])
top["ratio_permille"] = (top["freq"] / top["freq"].sum() * 1000).round(3)
top.to_csv(cfg.DIR_PROC / "top_words_overall.csv", index=False, encoding="utf-8-sig")

print(f"[步骤2] 清洗后文本：{cfg.PATH_CLEAN.name}  观测 {len(clean)} 条")
print(f"[步骤2] 平均保留词数 {clean['n_tokens'].mean():.1f} 词/份，"
      f"平均句数 {clean['n_sent'].mean():.1f} 句/份")
print(f"[步骤2] 全样本去停用词后共有 {len(counter)} 个不同词")
print("[步骤2] 高频词 Top10：" + "、".join(w for w, _ in counter.most_common(10)))
