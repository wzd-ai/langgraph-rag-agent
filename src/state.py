from langgraph.types import TypedDict
from pydantic import BaseModel,Field
from typing import Annotated,List,Literal
from langgraph.graph.message import add_messages
import operator

#结构化输出
class QueryAnalysis(BaseModel):
    """classify 节点的结构化输出"""
    rewritten_query: str = Field(description="将口语化查询重写为精准的检索式查询，保持用户原意不变")
    intents: List[Literal["tools", "rag", "escalate"]] = Field(description="用户意图列表，可多选。tools=查订单，rag=知识问答，escalate=转人工")
    reasoning: str = Field(description="简要说明分类依据")

#拆解问题
class SubQueries(BaseModel):
    """用于判断是否需要拆解 query"""
    is_complex: bool = Field(description="如果问题包含多个实体或对比（如'孟子和荀子的区别'），则为 True")
    sub_queries: List[str] = Field(default_factory=list, description="拆解后的子问题列表，如果 is_complex 为 False，则为空")


#大模型提取订单号
class OrderQueryParams(BaseModel):
    order_id: str | None = Field(default=None, description="用户的订单号，如果没有提取到则为空")

#自我修正
class GradeResult(BaseModel):
    is_good: bool = Field(description="回答是否完整且没有幻觉")
    reason: str = Field(description="如果回答不好，说明原因")


#全局state
class OverallState(TypedDict):
    #输入
    user_query: str           #重写后的
    original_query:str        #原始问题
    sub_queries: List[str]    #存放拆解后的子查询
    # 决策层
    retry_count: int          #记录重试次数
    intents: List[str]
    need_escalate: bool       #是否需要转人工
    # 执行层（并行节点写入，用 operator.add 合并）
    results: Annotated[List[str], operator.add]  #并行节点写入，用 operator.add 合并
    # 输出层
    final_answer: str
    messages: Annotated[list, add_messages]

