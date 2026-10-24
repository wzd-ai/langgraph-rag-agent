# 混合检索 + RRF + 重排
"""
在线检索模块：对外只暴露 retrieve_docs(query, top_k)。
内部完成：向量检索 + BM25 检索 + RRF 融合 + CrossEncoder 重排。
"""
import os
from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import bm25s
import pkuseg

from src.config import CHROMA_PERSIST_DIR, BM25_INDEX_PATH

load_dotenv()

# 分词器
seg = pkuseg.pkuseg()

# 懒加载,避免import 就加载
_vectorstore = None
_bm25_retriever = None
_cross_model = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        _vectorstore = Chroma(
            collection_name="knowledge_base",
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    return _vectorstore


def _get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=True)
    return _bm25_retriever

import threading

_cross_model_lock = threading.Lock()
def _get_cross_model():
    global _cross_model
    if _cross_model is None:
        with _cross_model_lock:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _cross_model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", device=device)
    return _cross_model


#  向量稠密检索
_chroma_lock = threading.Lock()
def _vector_search(query: str, k: int):
    vs = _get_vectorstore()
    total = vs._collection.count()
    k = min(k, total)
    if k <= 0:
        return []
    # ★ 加锁，防止并发读取冲突
    with _chroma_lock:
        docs = vs.similarity_search(query, k=k)
    return [{"id": doc.id, "content": doc.page_content} for doc in docs]

_bm25_lock = threading.Lock()
# BM25 稀疏检索
def _bm25_search(query: str, k: int):
    retriever = _get_bm25_retriever()
    query_tokens = [seg.cut(query)]
    k = min(k, len(retriever.corpus))
    with _bm25_lock:
        results, scores = retriever.retrieve(query_tokens, k=k)
    out = []
    for i in range(results.shape[1]):
        doc = results[0, i]
        out.append({"id": doc["id"], "content": doc["content"]})
    return out


#  RRF 融合
def _reciprocal_rank_fusion(ranked_lists, k: int = 60):
    rrf_scores, results = {}, {}
    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            results[doc_id] = doc
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [results[doc_id] for doc_id, _ in sorted_docs]


# CrossEncoder 重排
def _cross_encoder_rerank(query: str, docs, top_k: int = 5):
    if not docs:
        return []
    model = _get_cross_model()
    scores = model.predict([(query, doc["content"]) for doc in docs])
    for i, s in enumerate(scores):
        docs[i]["score"] = float(s)

    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    positive_docs = [doc for doc in sorted_docs if doc["score"] > -1.0]

    print(f"[rerank] 分数分布: {[round(float(s), 3) for s in scores]}")

    # ★ 核心兜底：如果都被阈值过滤了，强制保留前 top_k 个，防止上下文为空
    if not positive_docs:
        positive_docs = sorted_docs

    return positive_docs[:min(top_k, len(positive_docs))]


# 对外唯一接口
def retrieve_docs(query: str, top_k: int = 5):
    """
    输入 query，返回重排后的 top_k 文档列表。
    每个元素形如：{"id": ..., "content": ..., "score": ...}
    """
    vector_results = _vector_search(query, k=top_k)
    bm25_results = _bm25_search(query, k=top_k)
    if not vector_results or not bm25_results:
        return []
    rrf_results = _reciprocal_rank_fusion([vector_results, bm25_results])
    return _cross_encoder_rerank(query, rrf_results, top_k)