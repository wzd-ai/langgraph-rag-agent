# RAG Agent（你的 search_knowledge_base 工具
"""
RAG Agent：把检索封装成一个 tool，供 LangChain Agent 调用。
"""
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from dotenv import load_dotenv

import threading
_agent_lock = threading.Lock()

from src.rag_engine.retriever import retrieve_docs

load_dotenv()

# 用于存评估时的 contexts（生产环境建议用 thread_id 隔离）
contexts_store = {}


@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库，获取教育学、名人等的知识。需要查找资料时使用。"""
    final_docs = retrieve_docs(query, top_k=5)
    if not final_docs:
        return "未找到相关文档"

    # 存 context 供评估使用
    cs = contexts_store.get(query, [])
    cs.extend([doc["content"] for doc in final_docs])
    contexts_store[query] = list(set(cs))

    return "\n\n".join(doc["content"] for doc in final_docs)


# ===== 初始化 RAG Agent =====
rag_agent = create_agent(
    model=ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    ),
    tools=[search_knowledge_base],
    system_prompt="""你是一个电商答题助手。你必须严格遵守以下规则：
1. 有关电商的问题**只能**调用`search_knowledge_base`工具来查阅相关文档，根据文档信息回答问题
2. **禁止**添加任何你自己的知识、理解或推断
3. **禁止**扩展、举例或给出建议
4. 如果参考文档中没有相关信息，直接回复"根据提供的资料，无法回答此问题"
5. 回答要简洁，直接引用原文，不要改写或总结
""",
)