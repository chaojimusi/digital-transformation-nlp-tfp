# -*- coding: utf-8 -*-
"""
03_feature_extraction.py —— 第三步：特征提取（Feature Extraction）
================================================================
对应论文"3.4 特征提取：三条技术路线"。

本脚本同时实现课程给出的三条路线，主测度取路线一，另两条用于稳健性检验：

    路线一  词典法    词典命中词频（最透明、可复现，本科论文首选）
    路线二  词嵌入    基于 PPMI + 截断 SVD 的自建词向量，文档向量与
                      "数字化语义中心"的余弦相似度
    路线三  主题模型  LDA 主题比例中"数字化相关主题"的占比

输出：
    data/processed/word_freq_digital.csv   词典命中明细（企业-年度-词）
    data/processed/doc_features.csv        三路线文档级特征
"""
from __future__ import annotations

import importlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")
import nlp_utils

rng = np.random.default_rng(cfg.SEED)

# ------------------------------------------------------------------
# 0. 载入清洗后文本与词典
# ------------------------------------------------------------------
clean = pd.read_csv(cfg.PATH_CLEAN, encoding="utf-8-sig")
dictionary = pd.read_csv(cfg.PATH_DIGITAL_DICT, encoding="utf-8")
TERMS = dictionary["word"].astype(str).tolist()
DIM = dict(zip(dictionary["word"], dictionary["dimension"]))

# 按首字分桶 + 长度降序，实现"最长匹配优先"的高效词典扫描
BY_FIRST = nlp_utils.build_first_bucket(TERMS)


def count_dict_terms(text: str) -> Counter:
    """最长匹配优先的词典命中统计（避免 '数字化' 与 '数字化转型' 重复计数）。"""
    return nlp_utils.count_terms(text, BY_FIRST)


# ------------------------------------------------------------------
# 路线一：词典法
# ------------------------------------------------------------------
freq_rows, doc_rows = [], []
for row in clean.itertuples(index=False):
    c = count_dict_terms(row.text_clean)
    hits_A = sum(v for k, v in c.items() if DIM[k] == "A")
    hits_B = sum(v for k, v in c.items() if DIM[k] == "B")
    for w, v in c.items():
        freq_rows.append({"firm_id": row.firm_id, "year": row.year,
                          "word": w, "dimension": DIM[w], "count": v})
    doc_rows.append({"firm_id": row.firm_id, "year": row.year,
                     "n_tokens": row.n_tokens, "n_sent": row.n_sent,
                     "hits_total": hits_A + hits_B,
                     "hits_A": hits_A, "hits_B": hits_B})

freq = pd.DataFrame(freq_rows).sort_values(["firm_id", "year", "count"], ascending=[True, True, False])
freq.to_csv(cfg.PATH_WORDFREQ, index=False, encoding="utf-8-sig")
feat = pd.DataFrame(doc_rows)
print(f"[步骤3] 路线一（词典法）：命中明细 {len(freq)} 行，"
      f"平均命中 {feat['hits_total'].mean():.2f} 词/份")

# ------------------------------------------------------------------
# 路线二：词嵌入（PPMI 共现矩阵 + 截断 SVD 自建词向量）
# ------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

userdict = nlp_utils.build_userdict(cfg.PATH_DIGITAL_DICT, cfg.PATH_USERDICT)
nlp_utils.init_jieba(userdict)
STOPWORDS = nlp_utils.load_stopwords(cfg.PATH_STOP_GENERAL, cfg.PATH_STOP_FINANCE)

docs_tokens = [nlp_utils.remove_stopwords(nlp_utils.tokenize(t), STOPWORDS)
               for t in clean["text_clean"].tolist()]

vocab_counter = Counter(w for d in docs_tokens for w in d)
VOCAB = [w for w, _ in vocab_counter.most_common(3000)]
vocab_idx = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)

# 2.1 共现矩阵（窗口 ±3）
WIN = 3
cooc = np.zeros((V, V), dtype=np.float32)
for d in docs_tokens:
    idx = [vocab_idx[w] for w in d if w in vocab_idx]
    for a in range(len(idx)):
        for b in range(a + 1, min(a + WIN + 1, len(idx))):
            cooc[idx[a], idx[b]] += 1
            cooc[idx[b], idx[a]] += 1
cooc += cooc.T  # 对称化

# 2.2 PPMI 加权
total = cooc.sum()
row_sum = cooc.sum(axis=1, keepdims=True)
col_sum = cooc.sum(axis=0, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((cooc * total) / np.maximum(row_sum * col_sum, 1e-12))
ppmi = np.maximum(pmi, 0)
ppmi[np.isnan(ppmi)] = 0

# 2.3 截断 SVD 得到词向量（LSA 式词嵌入）
svd = TruncatedSVD(n_components=100, random_state=cfg.SEED)
word_vec = svd.fit_transform(ppmi)
word_vec = normalize(word_vec)

# 2.4 数字化语义中心 & 文档向量余弦相似度
dict_in_vocab = [w for w in TERMS if w in vocab_idx]
seed = word_vec[[vocab_idx[w] for w in dict_in_vocab]].mean(axis=0)
seed = seed / (np.linalg.norm(seed) + 1e-12)

doc_vec, cos_digital = [], []
for d in docs_tokens:
    ids = [vocab_idx[w] for w in d if w in vocab_idx]
    v = word_vec[ids].mean(axis=0) if ids else np.zeros(word_vec.shape[1])
    v = v / (np.linalg.norm(v) + 1e-12)
    doc_vec.append(v)
    cos_digital.append(float(np.dot(v, seed)))
feat["cos_digital"] = cos_digital
print(f"[步骤3] 路线二（词嵌入）：词表 {V} 词，向量维度 {word_vec.shape[1]}；"
      f"余弦相似度均值 {np.mean(cos_digital):.3f}")

# ------------------------------------------------------------------
# 路线三：主题模型 LDA
# ------------------------------------------------------------------
cv = CountVectorizer(vocabulary=VOCAB, tokenizer=lambda x: x, preprocessor=lambda x: x)
X = cv.fit_transform(docs_tokens)
K = 10
lda = LatentDirichletAllocation(n_components=K, random_state=cfg.SEED,
                                learning_method="batch", max_iter=30)
doc_topic = lda.fit_transform(X)

term_names = np.array(cv.get_feature_names_out())
digital_idx = {vocab_idx[w] for w in TERMS if w in vocab_idx}
is_dict_word = np.array([1.0 if j in digital_idx else 0.0 for j in range(V)])

# 每个主题的"数字化载荷" = 该主题词分布中落在词典词上的概率质量占比
topic_components = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
topic_dict_load = topic_components @ is_dict_word          # 形状 (K,)

# 文档的数字化主题得分 = 以主题数字化载荷为权重的主题比例加权和
denom = topic_dict_load.sum() + 1e-12
feat["topic_digital"] = (doc_topic @ topic_dict_load) / denom
feat["topic_digital_load"] = topic_dict_load[np.argmax(doc_topic, axis=1)]

# 判定"数字化主题"：数字化载荷显著高于全主题平均（相对阈值，避免绝对阈值在
# 不同语料规模下失效）
thr_load = max(float(topic_dict_load.mean() * 1.5), 0.05)
is_digital_topic = (topic_dict_load >= thr_load).astype(int)

lda_report = pd.DataFrame({
    "topic_id": range(K),
    "dict_load": topic_dict_load.round(4),
    "dict_share_top30": [round(float(np.mean(is_dict_word[np.argsort(lda.components_[k])[::-1][:30]])), 3)
                         for k in range(K)],
    "is_digital_topic": is_digital_topic,
    "top10_words": ["、".join(term_names[np.argsort(lda.components_[k])[::-1][:10]]) for k in range(K)],
})
lda_report.to_csv(cfg.DIR_PROC / "lda_topics.csv", index=False, encoding="utf-8-sig")
n_digital_topic = int(is_digital_topic.sum())
print(f"[步骤3] 路线三（主题模型）：K={K}，其中 {n_digital_topic} 个主题被判定为数字化主题"
      f"（载荷阈值 {thr_load:.3f}）；最高载荷 {topic_dict_load.max():.3f}")

feat.to_csv(cfg.DIR_PROC / "doc_features.csv", index=False, encoding="utf-8-sig")
print(f"[步骤3] 文档级特征已输出：{cfg.DIR_PROC / 'doc_features.csv'}  ({len(feat)} 行)")
