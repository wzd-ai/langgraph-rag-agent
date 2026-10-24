"""
RAG 评估脚本：用 RAGAS 量化检索和生成质量。
运行方式：
    cd knowledge_base_agent
    python -m evaluation.run_ragas
"""
import os
os.environ["RAGAS_CONCURRENCY"] = "2"  # 强制限制并发数为 2，牺牲速度换稳定

import json
import time
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from src.config import llm
# ===== RAGAS =====
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
)
from openai import AsyncOpenAI

# ===== 项目内模块 =====
from src.config import BASE_DIR
from src.graph import app

# ===== 加载环境变量 =====
load_dotenv(BASE_DIR / ".env")

# ===== 路径配置 =====
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DATASET_PATH = EVAL_DIR / "eval_dataset.json"
EXPERIMENTS_DIR = EVAL_DIR / "reports"


# ============================================================
# 1. 构建评估数据集：调 Agent + 收集 contexts
# ============================================================
def create_dataset():
    """跑一遍评估集，收集每条问题的 response 和 retrieved_contexts"""
    if EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
        print(f"已清空旧实验目录：{EXPERIMENTS_DIR}")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        name="rag_eval",
        backend="local/csv",
        root_dir=str(EXPERIMENTS_DIR),
    )

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"加载了 {len(eval_data)} 个评估问题")

    for i, item in enumerate(eval_data):
        query = item["question"]
        print(f"\n[{i+1}/{len(eval_data)}] 正在评估: {query}")
        # 从 contexts_store 里取检索片段
        # 注意：key 是 rag_node 里传给 rag_agent 的 query（可能是重写后的）
        # 这里用原始 query 兜底
        from src.rag_engine.retriever import retrieve_docs
        contexts = [doc["content"] for doc in retrieve_docs(query, top_k=8)]
        context_text = "\n\n".join(contexts)
        try:
            prompt = f"""你是一个电商客服答题助手。请严格根据以下资料回答问题。
如果资料中没有答案，直接回复“根据提供的资料，无法回答此问题”。

要求：在回答时，请完整列出资料中提到的所有具体数字、时间、条件（如：30分钟、48小时、95折、100积分等），不要遗漏。

资料：
{context_text}

问题：{query}
"""
            answer = llm.invoke(prompt).content
        except Exception as e:
            print(f"  [错误] LLM 生成失败: {e}")
            answer = ""
        dataset.append({
            "user_input": query,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        })
        print(f"  ✓ response: {answer[:50]}... | contexts: {len(contexts)} 条")

        # 限流：避免密集请求
        if i < len(eval_data) - 1:
            time.sleep(1)

    dataset.save()
    print(f"\n评估数据集已保存到: {EXPERIMENTS_DIR}")
    return dataset


# ============================================================
# 2. 初始化评估模型
# ============================================================
def build_evaluator():
    """初始化 RAGAS 的评估 LLM 和 Embedding"""
    llm_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        timeout=120.0,
        max_retries=5,
    )
    #给噪声准备的特定裁判
    qwen_client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        timeout=120.0,
        max_retries=5,
    )
    embedding_client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        timeout=120.0,
        max_retries=5,
    )

    # 专门给 CER 指标配一个 Qwen 裁判
    cer_evaluator_llm = llm_factory(
        model="qwen-plus",
        client=qwen_client,
        max_tokens=8192,
    )

    evaluator_llm = llm_factory(
        model="deepseek-chat",
        client=llm_client,
        max_tokens=8192,
    )
    evaluator_embeddings = embedding_factory(
        provider="openai",
        model="text-embedding-v3",
        client=embedding_client,
    )
    print("评估模型已就绪")
    return evaluator_llm, evaluator_embeddings,cer_evaluator_llm


# ============================================================
# 3. 批量评估
# ============================================================
async def run_evaluation(dataset, evaluator_llm, evaluator_embeddings, cer_evaluator_llm):
    """用 @experiment 装饰器批量评估"""

    # 初始化所有指标（按需取舍）
    faith_metric = Faithfulness(llm=evaluator_llm)
    ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    cp_metric = ContextPrecision(llm=evaluator_llm)
    cr_metric = ContextRecall(llm=evaluator_llm)
    cer_metric = ContextEntityRecall(llm=cer_evaluator_llm)
    ns_metric = NoiseSensitivity(llm=evaluator_llm)

    @experiment()
    async def rag_evaluation(row):
        """对单条数据，依次调用所有指标打分"""
        user_input = row["user_input"]
        response = row["response"]
        contexts = row["retrieved_contexts"]
        reference = row["reference"]
        
        cp_res = await cp_metric.ascore(
            user_input=user_input, reference=reference, retrieved_contexts=contexts
        )
        cr_res = await cr_metric.ascore(
            user_input=user_input, reference=reference, retrieved_contexts=contexts
        )
        cer_res = await cer_metric.ascore(
            reference=reference, retrieved_contexts=contexts
        )
        faith_res = await faith_metric.ascore(
            user_input=user_input, response=response, retrieved_contexts=contexts
        )
        ar_res = await ar_metric.ascore(
            user_input=user_input, response=response
        )
        ns_res = await ns_metric.ascore(
            user_input=user_input,
            response=response,
            reference=reference,
            retrieved_contexts=contexts,
        )

        return {
            "context_precision": cp_res.value,
            "context_recall": cr_res.value,
            "context_entity_recall": cer_res.value,
            "faithfulness": faith_res.value,
            "answer_relevancy": ar_res.value,
            "noise_sensitivity": ns_res.value,
        }

    print("\n开始批量评估（使用 @experiment）...")
    result = await rag_evaluation.arun(dataset, name="rag_eval_batch")
    print("评估完成！")
    return result


# ============================================================
# 4. 输出结果
# ============================================================
def report(result):
    """打印表格 + 保存 CSV + 绘制雷达图"""
    import pandas as pd

    df = result.to_pandas()
    print("\n========== 评估明细 ==========")
    print(df)

    # 保存 CSV
    csv_path = EXPERIMENTS_DIR / "ragas_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {csv_path}")

    # 平均分
    summary = df.mean(numeric_only=True)
    print("\n========== 指标平均分 ==========")
    for k, v in summary.items():
        if k == "noise_sensitivity":
            level = "优秀（抗噪强）" if v <= 0.2 else ("及格" if v <= 0.4 else "偏低（易受噪声干扰）")
        else:
            level = "优秀" if v >= 0.8 else ("及格" if v >= 0.6 else "偏低，需要排查")
        print(f"{k:25} = {v:.3f}  ({level})")

    # 雷达图
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 中文显示
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        labels = list(summary.index)
        values = list(summary.values)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("RAG 评估雷达图", size=14, pad=20)

        radar_path = EXPERIMENTS_DIR / "radar.png"
        plt.savefig(radar_path, dpi=150, bbox_inches="tight")
        print(f"雷达图已保存: {radar_path}")
        plt.close()
    except Exception as e:
        print(f"[警告] 雷达图绘制失败: {e}")

    # 优化建议
    print("""
========== 排查建议 ==========
- Faithfulness 低     → 模型在瞎编，加强"只依据上下文回答"的约束
- ContextRecall 低    → 该检索到的没检索到，改 chunk 策略 / 换 embedding
- ContextPrecision 低 → 检索回来一堆没用的，加 RRF / 重排
- AnswerRelevancy 低  → 回答跑题，改 prompt
- NoiseSensitivity 低 → 对噪声不敏感，检查检索噪声
""")


# ============================================================
# 5. 主入口
# ============================================================
async def main():
    # Step 1: 构建数据集
    dataset = create_dataset()

    # Step 2: 初始化评估模型（★ 注意这里是 3 个变量接收）
    evaluator_llm, cer_evaluator_llm, evaluator_embeddings = build_evaluator()

    # Step 3: 批量评估（★ 记得把 cer_evaluator_llm 传进去）
    result = await run_evaluation(dataset, evaluator_llm, cer_evaluator_llm, evaluator_embeddings)

    # Step 4: 输出报告
    report(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())