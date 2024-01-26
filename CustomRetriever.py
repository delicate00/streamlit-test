from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from typing import List

import logging

# logging.basicConfig(level=logging.DEBUG)
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""
    # 自定义检索器，可以同时执行向量搜索和知识图谱搜索
    # 定义了一个名为 CustomRetriever 的类，该类继承自 BaseRetriever 类
    def __init__(
        self,
        vector_retrievers: VectorIndexRetriever,     # 执行向量搜索
        kg_retrievers: KGTableRetriever,             # 执行知识图谱搜索
        mode: str = "OR",       # 搜索模式，默认为 "OR"
    ) -> None:
        """Init params."""

        self._vector_retrievers = vector_retrievers
        self._kg_retrievers = kg_retrievers
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    # 用于执行节点检索操作
    # 接受一个 QueryBundle 类型的参数，并返回一个 List[NodeWithScore] 类型的结果
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # 检索给定查询的节点
        kg_nodes = self._kg_retrievers.retrieve(query_bundle)
        vector_nodes = self._vector_retrievers.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}             # 通过集合推导式，从向量搜索结果中提取节点的 ID，并存储在 vector_ids 集合中
        kg_ids = {n.node.node_id for n in kg_nodes}                     # 通过集合推导式，从知识图谱搜索结果中提取节点的 ID，并存储在 kg_ids 集合中

        combined_dict = {n.node.node_id: n for n in vector_nodes}       # 使用字典推导式，将向量搜索结果中的节点 ID 作为键，节点对象作为值，构建一个字典 combined_dict
        combined_dict.update({n.node.node_id: n for n in kg_nodes})     # 使用字典推导式，将知识图谱搜索结果中的节点 ID 作为键，节点对象作为值，更新 combined_dict 字典

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)              # 取向量搜索结果和知识图谱搜索结果的交集，作为最终的检索结果的节点 ID
        else:
            retrieve_ids = vector_ids.union(kg_ids)                     # 取向量搜索结果和知识图谱搜索结果的并集，作为最终的检索结果的节点 ID

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]   # 根据检索结果的节点 ID，从 combined_dict 字典中获取对应的节点对象，并将它们存储在 retrieve_nodes 列表中
        return retrieve_nodes

