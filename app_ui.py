import streamlit as st
import uuid
from src.graph import app

# 1. 页面基本配置
st.set_page_config(page_title="Agentic RAG 智能助手", page_icon="🧠", layout="wide")
st.title("🧠 Agentic RAG 智能客服")
st.caption("支持多轮记忆、多意图拆解、并发检索与自我纠错 (LangGraph)")

# 2. 初始化 Session State（用来存聊天记录和用户 ID）
if "messages" not in st.session_state:
    st.session_state.messages = []

# ★ 核心：为每一个浏览器窗口生成一个唯一的 thread_id
# 这样不同的人打开网页，记忆是互相隔离的
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 侧边栏加个按钮，方便你随时清空记忆
with st.sidebar:
    st.write(f"当前会话 ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("🧹 清空当前对话记忆"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 3. 遍历显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. 接收用户输入
if prompt := st.chat_input("请提问，例如：帮我查订单12345，它发货没？"):
    # 把用户的话存进历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. 调用 LangGraph 核心引擎
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考与并发检索中..."):

            # ★ 严格按照你 OverallState 定义的字段初始化
            # 注意：我们需要把 Streamlit 的历史记录转换成 LangGraph 需要的格式
            initial_state = {
                "user_query": prompt,
                "original_query": prompt,
                "intents": [],
                "need_escalate": False,
                "results": [],
                "final_answer": "",
                "sub_queries": [],
                "retry_count": 0,
                # 把前面的历史记录传进去
                "messages": [("user", m["content"]) if m["role"] == "user" else ("assistant", m["content"]) for m in
                             st.session_state.messages[:-1]] + [("user", prompt)]
            }

            # ★ 核心：带上 thread_id 配置，LangGraph 的 MemorySaver 就会自动接管记忆
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                # 运行图，传入 config
                result = app.invoke(initial_state, config=config)
                answer = result.get("final_answer", "抱歉，系统未生成有效回复。")

                # 把 Agent 的回复存进历史并显示
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Agent 调用失败: {e}")