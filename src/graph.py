# 核心：组装节点，构建 LangGraph
"""
组装 LangGraph：把节点连接成图。
"""
from langgraph.graph import StateGraph, START, END

from src.state import OverallState
from src.nodes import (
    classify_node,
    router_condition,
    tools_node,
    rag_node,
    escalate_node,
    aggregate_node,
    generate_node,
    grade_node,
    query_decompose_node,
)


def build_graph():
    workflow = StateGraph(OverallState)

    # ===== 添加节点 =====
    workflow.add_node("classify", classify_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("decompose", query_decompose_node)

    # ===== 入口 =====
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "decompose")
    # ===== 条件路由：classify → router → (tools | rag | escalate) =====
    workflow.add_conditional_edges(
        "decompose",
        router_condition,
        {
            "tools": "tools",
            "rag": "rag",
            "escalate": "escalate",
        },
    )

    # ===== 三条分支都汇入 aggregate =====
    workflow.add_edge("tools", "aggregate")
    workflow.add_edge("rag", "aggregate")
    workflow.add_edge("escalate", "aggregate")

    # ===== 后续串行 =====
    workflow.add_edge("aggregate", "generate")
    workflow.add_edge("generate", "grade")


    workflow.add_conditional_edges(
        "grade",
        lambda state: "__end__",  # 实际运行会被 Command 覆盖，这里只为了画图
        {
            "__end__": END,
            "rag": "rag",
            "tools": "tools"
        }
    )
    return workflow.compile()
app = build_graph()
if __name__ == "__main__":
    # 直接打印 Mermaid 代码
    print(app.get_graph().draw_mermaid())