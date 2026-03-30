"""
测试：向量数据库
"""

import os
import tempfile
import pytest

from langchain_core.documents import Document
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents, embed_query


@pytest.fixture
def temp_vector_store():
    """
    创建临时向量存储实例。

    每个测试使用独立的集合名称和持久化目录。
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = f"test_collection_{id(temp_dir)}"

    store = VectorStore(
        collection_name=collection_name,
        persist_dir=temp_dir
    )

    yield store

    # 清理
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """
    提供测试用的 Document 列表。
    """
    return [
        Document(
            page_content="人工智能（AI）是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。",
            metadata={
                "source": "ai_intro.txt",
                "category": "technology",
            }
        ),
        Document(
            page_content="深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程。",
            metadata={
                "source": "deep_learning.txt",
                "category": "technology",
            }
        ),
        Document(
            page_content="自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
            metadata={
                "source": "nlp_intro.txt",
                "category": "technology",
            }
        ),
        Document(
            page_content="量子计算利用量子力学原理来执行计算，有望解决传统计算机难以处理的复杂问题。",
            metadata={
                "source": "quantum_computing.txt",
                "category": "physics",
            }
        ),
        Document(
            page_content="CRISPR-Cas9 是一种基因编辑技术，能够精确修改 DNA 序列。",
            metadata={
                "source": "crispr.txt",
                "category": "biology",
            }
        ),
    ]


class TestVectorStoreCRUD:
    """测试向量数据库的基本 CRUD 操作"""

    def test_upsert_documents(self, temp_vector_store, sample_documents):
        """测试插入文档"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        count = temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        assert count == len(sample_documents), f"应插入 {len(sample_documents)} 个文档，实际插入 {count}"

        # 验证文档数量
        assert temp_vector_store.count() == len(sample_documents), "数据库中文档数量不匹配"

    def test_upsert_empty_list(self, temp_vector_store):
        """测试插入空列表"""
        count = temp_vector_store.upsert([], [], metadatas=[])

        assert count == 0, "空列表应返回 0"
        assert temp_vector_store.count() == 0, "数据库应为空"

    def test_similarity_search(self, temp_vector_store, sample_documents):
        """测试相似度搜索"""
        # 先插入文档
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 搜索相关文档
        query = "什么是人工智能？"
        query_embedding = embed_query(query)
        results = temp_vector_store.similarity_search(query_embedding, top_k=2)

        assert len(results) == 2, f"应返回 2 个结果，实际返回 {len(results)}"
        assert all(isinstance(doc, Document) for doc in results), "每个结果应为 Document 对象"

        # 检查结果内容相关
        result_contents = [doc.page_content for doc in results]
        assert any("人工智能" in content or "AI" in content for content in result_contents), "结果应包含相关内容"

    def test_similarity_search_with_filter(self, temp_vector_store, sample_documents):
        """测试带过滤条件的相似度搜索"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 搜索特定类别的文档
        query = "技术"
        query_embedding = embed_query(query)
        results = temp_vector_store.similarity_search(
            query_embedding,
            top_k=10,
            filter={"category": "technology"}
        )

        assert len(results) > 0, "应返回结果"

        # 验证所有结果都属于指定类别
        for doc in results:
            assert doc.metadata.get("category") == "technology", f"文档类别不匹配: {doc.metadata.get('category')}"

    def test_similarity_search_no_results(self, temp_vector_store):
        """测试空数据库的搜索"""
        query_embedding = embed_query("随机查询")
        results = temp_vector_store.similarity_search(query_embedding, top_k=5)

        assert len(results) == 0, "空数据库应返回空列表"

    def test_similarity_search_with_score(self, temp_vector_store, sample_documents):
        """测试带分数的相似度搜索"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        query_embedding = embed_query("机器学习算法")
        results = temp_vector_store.similarity_search_with_score(
            query_embedding,
            top_k=3
        )

        assert len(results) == 3, f"应返回 3 个结果，实际返回 {len(results)}"

        # 检查每个结果是 (Document, score) 元组
        for doc, score in results:
            assert isinstance(doc, Document), "结果第一个元素应为 Document"
            assert isinstance(score, (int, float)), "结果第二个元素应为数字（分数）"
            assert score >= 0, "相似度分数应为非负数"

        # 检查分数按降序排列（更相似的分数更小）
        scores = [score for _, score in results]
        assert scores == sorted(scores), "结果应按相似度排序"

    def test_count(self, temp_vector_store):
        """测试文档计数"""
        assert temp_vector_store.count() == 0, "新数据库应为空"

        docs = [
            Document(page_content="文档 1"),
            Document(page_content="文档 2"),
            Document(page_content="文档 3"),
        ]

        embedded_pairs = embed_documents(docs)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)
        assert temp_vector_store.count() == 3, "计数不匹配"

    def test_reset_collection(self, temp_vector_store, sample_documents):
        """测试重置集合"""
        # 先插入文档
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)
        assert temp_vector_store.count() > 0, "数据库应不为空"

        # 重置集合
        temp_vector_store.reset_collection()

        # 验证集合已清空
        assert temp_vector_store.count() == 0, "重置后数据库应为空"


class TestVectorStoreOperations:
    """测试向量数据库的高级操作"""

    def test_upsert_after_reset(self, temp_vector_store, sample_documents):
        """测试重置后重新插入"""
        # 插入、重置、再插入
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)
        temp_vector_store.reset_collection()
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        count = temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        assert count == len(sample_documents), "重置后应能正常插入"
        assert temp_vector_store.count() == len(sample_documents), "计数应匹配"

    def test_multiple_searches(self, temp_vector_store, sample_documents):
        """测试多次搜索"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        queries = [
            "量子计算应用",
            "基因编辑技术",
            "深度学习模型"
        ]

        for query in queries:
            query_embedding = embed_query(query)
            results = temp_vector_store.similarity_search(query_embedding, top_k=1)
            assert len(results) == 1, f"查询 '{query}' 应返回 1 个结果"
            assert results[0].page_content, "结果应有内容"

    def test_search_after_multiple_upserts(self, temp_vector_store):
        """测试多次插入后的搜索"""
        # 第一批
        docs1 = [
            Document(page_content="第一批文档 1"),
            Document(page_content="第一批文档 2"),
        ]
        embedded_pairs1 = embed_documents(docs1)
        texts1 = [doc.page_content for doc, _ in embedded_pairs1]
        metadatas1 = [doc.metadata for doc, _ in embedded_pairs1]
        embeddings1 = [embedding for _, embedding in embedded_pairs1]
        temp_vector_store.upsert(texts1, embeddings1, metadatas=metadatas1)

        # 第二批
        docs2 = [
            Document(page_content="第二批文档 1"),
            Document(page_content="第二批文档 2"),
        ]
        embedded_pairs2 = embed_documents(docs2)
        texts2 = [doc.page_content for doc, _ in embedded_pairs2]
        metadatas2 = [doc.metadata for doc, _ in embedded_pairs2]
        embeddings2 = [embedding for _, embedding in embedded_pairs2]
        temp_vector_store.upsert(texts2, embeddings2, metadatas=metadatas2)

        # 验证总数
        assert temp_vector_store.count() == 4, "应有 4 个文档"

        # 搜索应能找到所有文档
        query_embedding = embed_query("文档")
        results = temp_vector_store.similarity_search(query_embedding, top_k=10)
        assert len(results) >= 2, "应能找到多个相关文档"

    def test_metadata_preservation(self, temp_vector_store):
        """测试元数据保留"""
        test_metadata = {
            "source": "test.txt",
            "page": 1,
            "author": "测试作者",
            "custom_field": "自定义值",
        }

        doc = Document(
            page_content="测试内容",
            metadata=test_metadata
        )

        embedded_pairs = embed_documents([doc])
        texts = [d.page_content for d, _ in embedded_pairs]
        metadatas = [d.metadata for d, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 检索并验证元数据
        query_embedding = embed_query("测试")
        results = temp_vector_store.similarity_search(query_embedding, top_k=1)
        assert len(results) == 1, "应返回 1 个结果"

        retrieved_metadata = results[0].metadata

        # 验证关键元数据字段存在
        assert "source" in retrieved_metadata, "应保留 source 字段"
        assert retrieved_metadata["source"] == test_metadata["source"], "source 值应匹配"

    def test_large_document_handling(self, temp_vector_store):
        """测试大文档处理"""
        # 创建大文档
        large_content = "这是一个很长的文档。" * 100  # 约 2000 字符

        doc = Document(
            page_content=large_content,
            metadata={"source": "large.txt"}
        )

        embedded_pairs = embed_documents([doc])
        texts = [d.page_content for d, _ in embedded_pairs]
        metadatas = [d.metadata for d, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 搜索
        query_embedding = embed_query("长文档")
        results = temp_vector_store.similarity_search(query_embedding, top_k=1)

        assert len(results) == 1, "应返回 1 个结果"
        assert len(results[0].page_content) >= 1000, "检索到的内容应保持完整"


class TestVectorStoreEdgeCases:
    """测试边界情况"""

    def test_query_special_characters(self, temp_vector_store, sample_documents):
        """测试包含特殊字符的查询"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 包含特殊字符的查询
        query = "人工智能 & AI !@#$%^&*()"
        query_embedding = embed_query(query)
        results = temp_vector_store.similarity_search(query_embedding, top_k=1)

        # 不应抛出异常
        assert isinstance(results, list), "应返回列表"

    def test_empty_query(self, temp_vector_store, sample_documents):
        """测试空查询"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 空查询不应崩溃
        query_embedding = embed_query("")
        results = temp_vector_store.similarity_search(query_embedding, top_k=3)
        assert isinstance(results, list), "应返回列表"

    def test_nonexistent_filter(self, temp_vector_store, sample_documents):
        """测试不存在的过滤条件"""
        embedded_pairs = embed_documents(sample_documents)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]
        temp_vector_store.upsert(texts, embeddings, metadatas=metadatas)

        # 使用不存在的过滤条件
        query_embedding = embed_query("查询")
        results = temp_vector_store.similarity_search(
            query_embedding,
            top_k=5,
            filter={"nonexistent_field": "value"}
        )

        # 应返回空结果或抛出异常（取决于实现）
        assert isinstance(results, list), "应返回列表"
        assert len(results) == 0, "不存在的过滤条件应返回空结果"