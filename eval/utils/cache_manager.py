"""
缓存管理模块

管理前半段输出的持久化和加载，支持TEST_PLAN.md定义的Schema。
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.documents import Document


class FirstHalfOutput:
    """前半段输出数据结构，符合TEST_PLAN.md定义的Schema"""

    def __init__(
        self,
        query: str,
        tasks: List[str],
        documents: List[Document],
        timestamp: Optional[str] = None,
        search_provider: str = "tavily",
        cache_version: str = "1.0"
    ):
        self.query = query
        self.tasks = tasks
        self.documents = documents
        self.timestamp = timestamp or datetime.now().isoformat()
        self.search_provider = search_provider
        self.cache_version = cache_version

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于JSON序列化"""
        return {
            "query": self.query,
            "tasks": self.tasks,
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ],
            "timestamp": self.timestamp,
            "search_provider": self.search_provider,
            "cache_version": self.cache_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstHalfOutput":
        """从字典创建实例"""
        documents = [
            Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            )
            for doc in data.get("documents", [])
        ]
        return cls(
            query=data["query"],
            tasks=data.get("tasks", []),
            documents=documents,
            timestamp=data.get("timestamp"),
            search_provider=data.get("search_provider", "tavily"),
            cache_version=data.get("cache_version", "1.0")
        )

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        return {
            "query": self.query,
            "num_tasks": len(self.tasks),
            "num_documents": len(self.documents),
            "timestamp": self.timestamp
        }


class CacheManager:
    """
    缓存管理器

    管理前半段输出的持久化和加载，支持TEST_PLAN.md定义的Schema
    """

    def __init__(self, cache_dir: str = "eval/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"

    def _get_cache_file_path(self, query_id: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{query_id}.json"

    def check_cache_exists(self, query_id: str) -> bool:
        """检查指定查询的缓存是否存在"""
        return self._get_cache_file_path(query_id).exists()

    def save_first_half_output(
        self,
        query_id: str,
        output: FirstHalfOutput
    ) -> str:
        """
        保存前半段输出到JSON文件

        Args:
            query_id: 查询ID（如"query_001"）
            output: 前半段输出

        Returns:
            保存的文件路径
        """
        cache_file = self._get_cache_file_path(query_id)

        data = output.to_dict()
        data["query_id"] = query_id  # 额外添加query_id字段

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 更新元数据
        self._update_metadata(query_id, output)

        return str(cache_file)

    def load_first_half_output(self, query_id: str) -> FirstHalfOutput:
        """
        从JSON文件加载前半段输出

        Args:
            query_id: 查询ID

        Returns:
            FirstHalfOutput实例

        Raises:
            FileNotFoundError: 如果缓存文件不存在
        """
        cache_file = self._get_cache_file_path(query_id)

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return FirstHalfOutput.from_dict(data)

    def load_all_cached_queries(self) -> List[FirstHalfOutput]:
        """
        加载所有已缓存的查询结果

        Returns:
            FirstHalfOutput列表
        """
        outputs = []

        for cache_file in sorted(self.cache_dir.glob("*.json")):
            if cache_file.name == "metadata.json":
                continue
            try:
                query_id = cache_file.stem
                output = self.load_first_half_output(query_id)
                outputs.append(output)
            except Exception as e:
                print(f"[警告] 加载缓存失败 {cache_file}: {e}")

        return outputs

    def load_all_cached_queries_with_ids(self) -> List[tuple[str, FirstHalfOutput]]:
        """
        加载所有已缓存的查询结果（包含query_id）

        Returns:
            (query_id, FirstHalfOutput) 列表
        """
        outputs = []

        for cache_file in sorted(self.cache_dir.glob("*.json")):
            if cache_file.name == "metadata.json":
                continue
            try:
                query_id = cache_file.stem
                output = self.load_first_half_output(query_id)
                outputs.append((query_id, output))
            except Exception as e:
                print(f"[警告] 加载缓存失败 {cache_file}: {e}")

        return outputs

    def _update_metadata(self, query_id: str, output: FirstHalfOutput):
        """更新元数据文件"""
        metadata = self._load_metadata()

        if "queries" not in metadata:
            metadata["queries"] = {}

        metadata["queries"][query_id] = {
            "query": output.query,
            "num_tasks": len(output.tasks),
            "num_documents": len(output.documents),
            "timestamp": output.timestamp
        }

        metadata["total_queries"] = len(metadata["queries"])
        metadata["last_updated"] = datetime.now().isoformat()

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据文件"""
        if not self.metadata_file.exists():
            return {
                "total_queries": 0,
                "creation_date": datetime.now().isoformat()
            }

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return self._load_metadata()

    def save_metadata(self, data: Dict[str, Any]):
        """保存运行级元数据"""
        metadata = self._load_metadata()
        metadata["run"] = data
        metadata["last_updated"] = datetime.now().isoformat()

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def delete_cache(self, query_id: str) -> bool:
        """
        删除指定查询的缓存

        Args:
            query_id: 查询ID

        Returns:
            是否成功删除
        """
        cache_file = self._get_cache_file_path(query_id)

        if cache_file.exists():
            cache_file.unlink()

            # 更新元数据
            metadata = self._load_metadata()
            if "queries" in metadata and query_id in metadata["queries"]:
                del metadata["queries"][query_id]
                metadata["total_queries"] = len(metadata["queries"])
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            return True

        return False

    def clear_all_cache(self):
        """清除所有缓存"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

        if self.metadata_file.exists():
            self.metadata_file.unlink()
