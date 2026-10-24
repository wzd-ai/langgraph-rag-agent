import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#根目录
BASE_DIR = Path(__file__).resolve().parents[1]
#根目录加载.env文件
load_dotenv(BASE_DIR/".env")
#向量数据库持久化目录
CHROMA_PERSIST_DIR = str(
    BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/vector_store/chroma_db")
)
#BM25索引文件目录
BM25_INDEX_PATH = str(
    BASE_DIR / os.getenv("BM25_INDEX_PATH", "data/vector_store/bm25_index")
)
#原始文档目录
RAW_DOC_PATH = str(
    BASE_DIR / os.getenv("RAW_DOC_PATH", "data/raw/中二知识笔记.md")
)
#日志目录
LOGGING_LOGGER= str(
    BASE_DIR / os.getenv("LOGGING_LOGGER", "data/logs/file_log")
)


#确保文件存在,没有则建立
Path(CHROMA_PERSIST_DIR).parent.mkdir(parents=True,exist_ok=True)
Path(BM25_INDEX_PATH).parent.mkdir(parents=True,exist_ok=True)
Path(LOGGING_LOGGER).parent.mkdir(parents=True,exist_ok=True)

#初始化模型
llm=init_chat_model(
    model="qwen-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

#开机自检
def print_config():
    """打印关键配置，用于调试"""
    print("=" * 50)
    print(f"[config] BASE_DIR          = {BASE_DIR}")
    print(f"[config] RAW_DOC_PATH      = {RAW_DOC_PATH}")
    print(f"[config] CHROMA_PERSIST_DIR = {CHROMA_PERSIST_DIR}")
    print(f"[config] BM25_INDEX_PATH   = {BM25_INDEX_PATH}")
    print(f"[config] LOGGING_LOGGER    = {LOGGING_LOGGER}")
    print(f"知识库文档是否存在      = {Path(RAW_DOC_PATH).exists()}")

if __name__=="__main__":
    print_config()