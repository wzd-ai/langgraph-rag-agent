"""
所有节点函数：classify / router / tools / rag / escalate / aggregate / generate / grade。
"""
from src.config import llm
from src.state import OverallState, QueryAnalysis
from src.utils import logger, query_order_api
from src.rag_engine.agent import rag_agent, _agent_lock
from pydantic import BaseModel,Field
import threading

# classify 节点：任务拆解 + 问题重写
def classify_node(state: OverallState):
    """用结构化输出同时完成：问题重写 + 意图分类"""
    query = state["user_query"]
    logger.info(f"[classify] 原始 query: {query}")
    structured_llm = llm.with_structured_output(QueryAnalysis)
    result: QueryAnalysis = structured_llm.invoke(f"""你是一个意图分类和查询重写助手。
请分析下面用户的问题，并提取出三个字段：

用户问题：{query}

字段要求：
1. rewritten_query: 如果用户问题是口语化的（含"咋""啥"等），请重写为标准检索式查询；如果问题已经很清晰，直接返回原问题。不要添加未提及的信息。
2. intents: 从 ["tools", "rag", "escalate"] 中选择一个或多个。tools=查订单/查状态，rag=查知识库/常识问答，escalate=转人工/投诉。
3. reasoning: 用一句话说明你做出这个判断的理由。

请以 JSON 格式输出。
""")

    logger.info(f"[classify] 意图: {result.intents}, 重写: {result.rewritten_query}")
    return {
        "user_query": result.rewritten_query,
        "intents": result.intents,
        "need_escalate": "escalate" in result.intents,
    }

#问题拆解节点
from src.state import SubQueries


def query_decompose_node(state: OverallState):
    """判断是否需要拆解 Query"""
    query = state["user_query"]
    logger.info(f"[decompose] 检查是否拆解: {query}")
    decomposer = llm.with_structured_output(SubQueries)
    # 强化 Prompt
    result = decomposer.invoke(f"""请分析下面的用户问题。
如果问题包含多个独立主体、对比（如"孟子和荀子的区别"）或连词（如"和"、"以及"），请将其拆解为多个独立子问题。
如果只是一个简单问题（例如"孔子是谁"），则 is_complex 为 False，sub_queries 为空列表。
问题：{query}
请以 JSON 格式输出。
""")

    logger.info(f"[decompose] LLM 原始返回: is_complex={result.is_complex}, sub_queries={result.sub_queries}")

    if result.is_complex and result.sub_queries:
        logger.info(f"[decompose] 成功拆解为: {result.sub_queries}")
        return {"sub_queries": result.sub_queries}

    logger.info("[decompose] 判定为简单问题，不拆解")
    return {"sub_queries": [query]}



#  router 节点：条件路由 + Send 并行分发

def router_condition(state: OverallState):
    from langgraph.types import Send
    intents = state["intents"]

    # 转人工逻辑不变
    if state.get("need_escalate") or "escalate" in intents:
        return "escalate"

    sends = []
    # ★ 针对 RAG 意图，按子查询并行分发
    sub_queries = state.get("sub_queries") or [state["user_query"]]
    if "rag" in intents:
        for sub_q in sub_queries:
            if "订单" not in sub_q or "思想" in sub_q or "政策" in sub_q:
                sends.append(Send("rag", {"user_query": sub_q}))

    if "tools" in intents:
        sub_queries = state.get("sub_queries") or [state["user_query"]]
        for sub_q in sub_queries:
            import re
            if "订单" in sub_q or re.search(r"\d{5,}", sub_q):
                sends.append(Send("tools", {"user_query": sub_q}))
    if not sends:
        sends.append(Send("rag", {"user_query": state["user_query"]}))

    return sends


 # tools 节点：查订单/查状态
_tools_llm_lock = threading.Lock()
from src.state import OrderQueryParams
def tools_node(state: OverallState):
    """调用订单查询 API（使用 Function Calling 提取参数）"""
    query = state["user_query"]
    logger.info(f"[tools] 准备查询: {query}")

    # 1. 用 LLM 提取订单号
    extractor_llm = llm.with_structured_output(OrderQueryParams)
    try:
        with _tools_llm_lock:
            params = extractor_llm.invoke(f"""请从以下用户的输入中提取订单号。如果找不到，请将 order_id 设为 null。
            注意：订单号通常是纯数字或字母组合。
            例如：用户说“帮我查订单12345”，你应该提取出 "12345"。
            请以 JSON 格式输出。

            用户输入：{query}
            请用 JSON 格式输出。""")
            order_id = params.order_id
    except Exception as e:
        logger.error(f"[tools] 参数提取失败: {e}")
        order_id = None

    logger.info(f"[tools] LLM 提取到的订单号: {order_id}")

    if not order_id:
        logger.warning("[tools] 未提取到订单号，直接返回提示")
        return {"results": ["[tools] 抱歉，请提供准确的订单号，例如：12345"]}


    data = query_order_api(order_id=order_id)

    if "error" in data:
        result_text = f"[tools] 订单查询失败：{data['error']}"
    else:
        result_text = (
            f"[tools] 订单 {data['order_id']} 状态：{data['status']}，"
            f"物流：{data['logistics_company']}，运单号：{data['tracking_no']}"
        )
    return {"results": [result_text]}


# rag 节点：知识库问答

def rag_node(state: OverallState):
    """调用 RAG Agent 生成答案"""
    query = state["user_query"]
    logger.info(f"[rag] 检索: {query}")

    #加锁
    with _agent_lock:
        try:
            response = rag_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
            )
            answer = response["messages"][-1].content
        except Exception as e:
            logger.error(f"[rag] Agent 调用失败: {e}")
            answer = f"检索发生错误: {e}"
    return {"results": [f"[rag] {answer}"]}


#  escalate 节点：转人工

def escalate_node(state: OverallState):
    """标记转人工，写入聚合结果"""
    logger.info("[escalate] 触发转人工")
    return {"results": ["[escalate] 已为您转接人工客服，请稍候..."]}



#  aggregate 节点：汇总所有并行结果

def aggregate_node(state: OverallState):
    """把 results 列表拼成一段上下文"""
    combined = "\n".join(state.get("results", []))
    logger.info(f"[aggregate] 汇总 {len(state.get('results', []))} 条结果")
    return {"final_answer": combined}



# generate 节点：统一生成最终回复

def generate_node(state: OverallState):
    """用 LLM 把聚合结果润色成自然语言"""
    original_query = state.get("original_query", "")
    context = state.get("final_answer", "")

    prompt = f"""用户原始问题：{original_query}

检索/查询到的信息：
{context}

请用简洁、自然的中文回答用户问题。要求：
1. 只依据上面提供的信息，不要编造
2. 如果信息包含订单查询结果，如实说明订单状态
3. 如果信息包含知识库内容，直接引用核心内容
4. 如果两者都有，分点清晰回答
"""
    history = state.get("messages", [])
    response = llm.invoke(history + [("user", prompt)])
    return {"final_answer": response.content,
            "messages": [("assistant", response.content)]}



# 8. grade 节点：答案质检
from langgraph.types import Command
from src.state import GradeResult
def grade_node(state: OverallState):
    """质检节点：调用 LLM 判断回答质量，不合格则重新检索"""
    question = state.get("original_query", "")
    answer = state.get("final_answer", "")
    contexts = state.get("results", [])  # results 里存的就是检索结果
    intents = state.get("intents", [])

    retry_count = state.get("retry_count", 0)

    # 防止死循环：如果重试超过 2 次，哪怕不好也放行
    if retry_count >= 2:
        logger.warning("[grade] 重试次数过多，强制通过")
        return Command(goto="__end__")

    # 调用 LLM 质检
    grader = llm.with_structured_output(GradeResult)
    prompt = f"""你是一个质检员。请判断下面的回答是否满足以下两点：
1. 回答是否完全基于提供的资料？（没有瞎编）
2. 回答是否完整回答了用户的问题？
如果满足，is_good 为 True；如果不满足，is_good 为 False。

用户问题：{question}
提供的资料：{" ".join(contexts)}
生成的回答：{answer}
请用 JSON 格式输出，例如：
{
{"is_good": True, "reason": "回答完全基于提供的资料，并完整回答了用户问题"}}
"""
    result = grader.invoke(prompt)

    if result.is_good:
        logger.info("[grade] 质检通过")
        return Command(goto="__end__")  # 直接结束
    else:
        logger.warning(f"[grade] 质检不通过，准备重新检索。原因: {result.reason}")
        # 根据意图决定退回哪里
        if "tools" in intents and "rag" not in intents:
            # 如果是纯查订单失败，退回 tools 重试（而不是去查知识库）
            next_node = "tools"
        else:
            # 如果是知识库问题，退回 rag
            next_node = "rag"
        logger.info(f"[grade] 触发重试，退回节点: {next_node}")
        return Command(
            goto=next_node,
            update={"final_answer": "", "retry_count": retry_count + 1}
        )
