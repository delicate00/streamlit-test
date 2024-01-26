import json
import os
import logging
import sys

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

from dashscope import TextEmbedding
import dashscope                    # 这里是调用阿里的包
from llama_index import (
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index import download_loader
from langchain_community.llms import Tongyi # 这里正在调用通义千问大模型
import os
from langchain import OpenAI        # OpenAI大模型

from llama_index import load_index_from_storage, load_indices_from_storage
from llama_hub.youtube_transcript import YoutubeTranscriptReader
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from llama_index import SimpleDirectoryReader, KnowledgeGraphIndex, ServiceContext, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex

from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from typing import List
from CustomRetriever import *
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever

from llama_index.node_parser import SimpleNodeParser
import faiss
from llama_index.vector_stores import FaissVectorStore
from llama_index import StorageContext

from neo4j import GraphDatabase

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore, Neo4jGraphStore
from llama_index.vector_stores import Neo4jVectorStore


os.environ["OPENAI_API_KEY"] = "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"
os.environ["DASHSCOPE_API_KEY"] = "sk-e2f85e4067304e53b510b3084897c915"



# host = "124.220.147.101"
# port = 7687
# username = "neo4j"
# password = "123456"
host = "127.0.0.1"
port = 7687
username = "neo4j"
password = "neo4j123"

url = f"bolt://{host}:{port}"
embed_dim = 1536

# 创建Neo4j图存储
graph_store = Neo4jGraphStore(
    url=f"bolt://{host}:{port}",
    username=username,
    password=password,
    database="",
    node_label="nit"    # 节点标签
)

# 创建Neo4j向量存储
vector_store = Neo4jVectorStore(username, password, url, embed_dim)

llm = Tongyi()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)  # 使用默认配置创建一个服务上下文对象

local_persist_path = "./doc"

docs_name = []


def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name)


def load_new_documents1(file_path):  # 方便加载不同文件，保存用不同的名字
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()    # 文档读取
    print("加载成功！准备更新数据库...")

    # 从存储中加载一个知识图谱索引
    new_kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(graph_store=graph_store),  # 存储上下文对象，用于指定存储的配置和位置
        service_context=service_context,
        max_triplets_per_chunk=15,
        verbose=True,
    )
    new_kg_index.storage_context.persist(persist_dir='./storage_neo4j_KG/nit')

    # 从存储中加载向量索引
    new_vector_index = VectorStoreIndex.from_documents(                       # 向量检索 嵌入文档/加载
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )
    new_vector_index.storage_context.persist(persist_dir='./storage_neo4j_Vector/nit')

    # file_name = os.path.basename(file_path)
    # docs_name.append(file_name)
    # print("file_path", file_path)
    # print("docs_name", docs_name)


# 上传文件
def load_new_documents(file_path):  # 方便加载不同的PDF文件，保存用不同的名字
    print("file_path", file_path)
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()    # 文档读取
    print("加载成功！准备更新数据库...")

    # 从存储中加载一个知识图谱索引
    new_kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(graph_store=graph_store),  # 存储上下文对象，用于指定存储的配置和位置
        service_context=service_context,
        max_triplets_per_chunk=15,
        verbose=True,
    )
    new_kg_index.storage_context.persist(persist_dir='./storage_neo4j_KG/nit')

    # 从存储中加载向量索引
    new_vector_index = VectorStoreIndex.from_documents(                       # 向量检索 嵌入文档/加载
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )
    new_vector_index.storage_context.persist(persist_dir='./storage_neo4j_Vector/nit')

    file_name = os.path.basename(file_path)
    docs_name.append(file_name)
    print("file_path", file_path)
    print("docs_name", docs_name)


# 知识图谱的回答（向量+关键词）
def query_KG(question):
    # 从存储中加载一个知识图谱索引
    storage_context = StorageContext.from_defaults(persist_dir="./storage_neo4j_KG/nit", graph_store=graph_store)
    kg_index = load_index_from_storage(
        storage_context=storage_context,    # 存储上下文对象，用于指定存储的配置和位置
        service_context=service_context,    # 服务上下文（全局已定义）
        max_triplets_per_chunk=15,          # 指定每个数据块（chunk）中的最大三元组数量
        verbose=True,                       # 是否打印详细信息
    )
    # 混合索引
    hybrid_query_engine = kg_index.as_query_engine(
        include_text=True,                  # 查询结果包含文本信息
        response_mode="tree_summarize",     # 查询结果以树状结构进行汇总和展示
        embedding_mode="hybrid",            # 指定嵌入模式。使用混合嵌入模式
        similarity_top_k=3,                 # 查询结果中返回与查询相关性最高的前3个结果
        explore_global_knowledge=True,      # 探索全局知识，即在知识图谱中查找相关的知识
    )
    response = hybrid_query_engine.query(question)
    print("response", response)
    return f"{response}"


# 原生向量的回答（简单的向量检索）
def query_Vector(question):                 # 通过加载持久化索引来避免重新加载和重新索引数据
    vector_index = load_index_from_storage(
        storage_context=StorageContext.from_defaults(persist_dir='./storage_neo4j_Vector/nit', vector_store=vector_store)
    )
    vector_query_engine = vector_index.as_query_engine()
    response = vector_query_engine.query(question)
    print("response", response)
    return f"{response}"


# 知识图谱和原生向量的结合
def query_KG_and_Vector(question):
    kg_index = load_index_from_storage(       # 单一 load_index_from_storage
        storage_context=StorageContext.from_defaults(persist_dir='./storage_neo4j_KG/nit', graph_store=graph_store),
        service_context=service_context,
        max_triplets_per_chunk=15,
        verbose=True,
    )
    vector_index = load_index_from_storage(
        storage_context=StorageContext.from_defaults(persist_dir='./storage_neo4j_Vector/nit', vector_store=vector_store),  # 存储上下文对象，用于指定存储的配置和位置
    )

    vector_retrievers = VectorIndexRetriever(index=vector_index)     # 创建向量索引检索器对象

    kg_retrievers = KGTableRetriever(                                   # 知识图谱表格检索器对象
        index=kg_index,
        retriever_mode="keyword",
        include_text=False  # 为 "keyword" 表示使用关键词检索模式
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize"  # 生成树状摘要的响应
    )

    custom_retriever = CustomRetriever(vector_retrievers, kg_retrievers)

    custom_query_engine = RetrieverQueryEngine(  # 默认的 mode OR 保证了两种搜索结果的并集，结果是包含了这两个搜索方式的结果，且进行了结果去重
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    response = custom_query_engine.query(question)
    print("response", response)
    return f"{response}"


# RAG检索
def query_RAG(question):
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=StorageContext.from_defaults(persist_dir='./storage_neo4j_KG/nit', graph_store=graph_store),
        service_context=service_context,
        llm=llm,
        verbose=True,
    )
    kg_rag_query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever, service_context=service_context
    )
    response = kg_rag_query_engine.query(question)
    print("response", response)
    return f"{response}"