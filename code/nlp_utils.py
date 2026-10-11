# -*- coding: utf-8 -*-
"""
nlp_utils.py —— 文本处理公共函数
================================
被 02_preprocessing.py / 03_feature_extraction.py 共同调用，避免逻辑重复。

包含：
    * build_userdict      由数字化词典生成 jieba 自定义词典
    * load_stopwords      加载"通用 + 财经领域"三维停用词表
    * normalize_text      全角转半角、简称归一（词形归一）
    * clean_text          去除页眉页脚、页码、乱码与多余空白
    * tokenize            jieba 精确模式分词
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import jieba
import pandas as pd

# 词形归一：把同一构念的不同写法映射到统一词形
SYNONYM_MAP = {
    "AI": "人工智能",
    "ai": "人工智能",
    "A.I.": "人工智能",
    "5G": "第五代移动通信",
    "区块链技术": "区块链",
    "大数据技术": "大数据",
    "云计算技术": "云计算",
    "物联网技术": "物联网",
    "工业互联网平台": "工业互联网",
}

# 页眉页脚 / 页码 / 公告套话等噪声模式
NOISE_PATTERNS = [
    re.compile(r"第\s*[0-9一二三四五六七八九十百]+\s*页"),
    re.compile(r"共\s*[0-9一二三四五六七八九十百]+\s*页"),
    re.compile(r"\d{4}\s*年\s*年度报告"),
    re.compile(r"证券代码[:：]?\s*\d+"),
    re.compile(r"证券简称[:：]?\s*[\u4e00-\u9fa5A-Za-z0-9]+"),
    re.compile(r"公告编号[:：]?\s*[\w\-]+"),
]

# 保留：中文、字母、数字、百分号、小数点、以及中文常用标点
KEEP_CHARS = re.compile(
    r"[^\u4e00-\u9fa5A-Za-z0-9%\.\+"
    r"\uff0c\u3002\uff1b\uff1a\u3001\uff08\uff09\u201c\u201d\uff01\uff1f\u2014\u2026\uff05]"
)


def build_userdict(dict_csv: Path, out_txt: Path) -> Path:
    """把数字化词典写入 jieba 自定义词典，保证专有术语不被切碎。"""
    df = pd.read_csv(dict_csv, encoding="utf-8")
    lines = [f"{w} 5000 n" for w in df["word"].astype(str).tolist()]
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return out_txt


def init_jieba(userdict: Path) -> None:
    jieba.initialize()
    jieba.load_userdict(str(userdict))


def load_stopwords(*paths: Path) -> set[str]:
    sw: set[str] = set()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                sw.add(w)
    return sw


def normalize_text(text: str) -> str:
    """全角转半角 + 简称归一。"""
    text = unicodedata.normalize("NFKC", text)
    for k, v in SYNONYM_MAP.items():
        text = text.replace(k, v)
    return text


def clean_text(text: str) -> str:
    """文档解析后的清洗：去页眉页脚、页码、乱码与多余空白。"""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    for pat in NOISE_PATTERNS:
        t = pat.sub("", t)
    # 去掉噪声字符，但保留句读标点，避免相邻句子拼接产生"伪词"
    t = KEEP_CHARS.sub("", t)
    t = re.sub(r"\s+", "", t)
    return t


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词。"""
    return [w for w in jieba.lcut(text) if w.strip()]


def remove_stopwords(tokens: Iterable[str], stopwords: set[str]) -> list[str]:
    return [w for w in tokens
            if w not in stopwords and len(w) > 1 and not w.isdigit()]


# ------------------------------------------------------------------
# 词典命中统计（最长匹配优先，避免长短词重复计数）
# ------------------------------------------------------------------
def build_first_bucket(terms: Iterable[str]) -> dict[str, list[str]]:
    """按首字分桶并按词长降序排列，用于高效的最长匹配扫描。"""
    bucket: dict[str, list[str]] = defaultdict(list)
    for w in terms:
        bucket[w[0]].append(w)
    for k in bucket:
        bucket[k].sort(key=len, reverse=True)
    return bucket


def count_terms(text: str, by_first: dict[str, list[str]]) -> Counter:
    """在文本中统计词典命中次数（最长匹配优先、非重叠计数）。"""
    cnt: Counter = Counter()
    i, n = 0, len(text)
    while i < n:
        cands = by_first.get(text[i])
        if cands:
            for w in cands:
                if text.startswith(w, i):
                    cnt[w] += 1
                    i += len(w)
                    break
            else:
                i += 1
        else:
            i += 1
    return cnt
