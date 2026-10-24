from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import CHROMA_PERSIST_DIR,BM25_INDEX_PATH,RAW_DOC_PATH

#返回切分后的文档
def load_and_split(md_path: str):
    """加载 md 文档并按 Markdown 标题 + 递归字符切分"""
    with open(md_path,"r",encoding="utf-8") as f :
        text=f.read()
    split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    mark_splitter=MarkdownHeaderTextSplitter(split_on)
    chunks=mark_splitter.split_text(text)

    recursive_splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。"]
    )
    docs=[]
    idx=1
    for doc in chunks:
        if len(doc.page_content)>300:
            sub_docs=recursive_splitter.split_documents([doc])
            for s in sub_docs:
                s.page_content=f"### {s.metadata.get('Header 3', '')}\n{s.page_content}"
                s.id=f"doc_{idx}"
                idx+=1
                docs.append(s)
        else:
            doc.page_content=f"### {doc.metadata.get('Header 3', '')}\n{doc.page_content}"
            doc.id=f"doc_{idx}"
            idx+=1
            docs.append(doc)
    return docs


#构建chroma向量库
def build_vector_store(docs):
    """构建 Chroma 向量库"""
    embedding=DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v3"
    )
    os.makedirs(CHROMA_PERSIST_DIR,exist_ok=True)
    vectorstore=Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="knowledge_base",
        embedding_function=embedding
    )
    try:
        existing=vectorstore.get()
        if existing["ids"]:
            vectorstore.delete(existing["ids"])
    except Exception:
        pass
    vectorstore.add_documents(docs)
    print(f"[build_index] 已写入 {len(docs)} 条向量到 {CHROMA_PERSIST_DIR}")
    return vectorstore




import shutil
import pkuseg
import bm25s
seg=pkuseg.pkuseg()


#构建bm25索引
def build_bm25_index(docs):
    """构建 BM25 索引并保存到磁盘"""
    if os.path.exists(BM25_INDEX_PATH):
        shutil.rmtree(BM25_INDEX_PATH)
    os.makedirs(BM25_INDEX_PATH,exist_ok=True)
    d_docs=[{"id":doc.id,"content":doc.page_content}for doc in docs]
    tokens=[seg.cut(doc['content']) for doc in d_docs]
    retriever=bm25s.BM25(k1=1.5, b=0.75,corpus=d_docs)
    retriever.index(tokens)
    retriever.save(BM25_INDEX_PATH)
    print(f"[build_index] BM25 索引已保存到 {BM25_INDEX_PATH}")

def main():
    print("[build_index] 开始构建索引...")
    docs = load_and_split(RAW_DOC_PATH)
    print(f"[build_index] 切分后共 {len(docs)} 条文档")
    build_vector_store(docs)
    build_bm25_index(docs)
    print("[build_index] 索引构建完成！")


if __name__=="__main__":
    main()









