from typing import List, Dict, Optional, Any, Union
import numpy as np
import pyarrow as pa
import lancedb
from loguru import logger

from .credentials import LanceDBCredentials
from .schemas import (
    VectorData,
    TableConfig,
    SearchConfig,
    BatchConfig,
    IndexConfig,
    SearchResult,
    BatchResult,
)


class LanceDBService:
    """Service for interacting with LanceDB embedded database"""

    def __init__(self, credentials: Optional[LanceDBCredentials] = None):
        """Initialize the LanceDB service"""
        self.credentials = credentials or LanceDBCredentials.from_env()
        self.db = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to LanceDB"""
        try:
            self.db = lancedb.connect(uri=self.credentials.database_uri)
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            raise

    async def create_table(self, config: TableConfig) -> bool:
        """
        Create a new table in the database

        Args:
            config: TableConfig instance with table configuration

        Returns:
            bool: True if successful
        """
        try:
            # Create schema with vector column and metadata
            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), config.vector_dim)),
                    pa.field(
                        "metadata",
                        pa.struct(
                            [pa.field("id", pa.int64()), pa.field("text", pa.string())]
                        ),
                    ),
                ]
            )

            if config.mode == "create":
                if config.name in self.db.table_names():
                    raise ValueError(f"Table {config.name} already exists")
                self.db.create_table(config.name, schema=schema)
            elif config.mode == "overwrite":
                self.db.create_table(config.name, schema=schema, mode="overwrite")
            else:
                raise ValueError(f"Unsupported table creation mode: {config.mode}")

            return True
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}")
            raise

    async def add_vectors(
        self, table_name: str, vectors: List[VectorData]
    ) -> BatchResult:
        """
        Add vectors to a table

        Args:
            table_name: Name of the target table
            vectors: List of VectorData instances

        Returns:
            BatchResult: Result of the operation
        """
        try:
            table = self.db[table_name]
            # Convert to list of dictionaries
            data = [{"vector": v.vector, "metadata": v.metadata} for v in vectors]

            table.add(data)
            return BatchResult(
                success=True, affected_rows=len(vectors), error_message=None
            )
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return BatchResult(success=False, affected_rows=0, error_message=str(e))

    async def search_vectors(self, config: SearchConfig) -> List[SearchResult]:
        """
        Search for similar vectors

        Args:
            config: SearchConfig instance with search parameters

        Returns:
            List[SearchResult]: List of search results
        """
        try:
            table = self.db[config.table_name]
            query = table.search(config.query_vector)

            if config.metric:
                query = query.metric(config.metric)
            if config.filter_expr:
                query = query.filter(config.filter_expr)

            results = query.limit(config.k).to_list()

            return [
                SearchResult(
                    vector=result["vector"],
                    distance=result["_distance"],
                    metadata=result.get("metadata"),
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise

    async def create_index(self, config: IndexConfig) -> bool:
        """
        Create an index on a table

        Args:
            config: IndexConfig instance with index parameters

        Returns:
            bool: True if successful
        """
        try:
            table = self.db[config.table_name]
            logger.info(f"Creating index on table: {config.table_name}")
            logger.info(f"config: {config}")

            # Create index with parameters
            index_params = {}
            if config.params:
                # Remove metric from params if present to avoid duplicate
                index_params = {k: v for k, v in config.params.items() if k != "metric"}

            table.create_index()
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    async def delete_table(self, table_name: str) -> bool:
        """
        Delete a table from the database

        Args:
            table_name: Name of the table to delete

        Returns:
            bool: True if successful
        """
        try:
            if table_name in self.db.table_names():
                self.db[table_name].drop()
            return True
        except Exception as e:
            logger.error(f"Failed to delete table: {str(e)}")
            raise

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table

        Args:
            table_name: Name of the table

        Returns:
            Dict[str, Any]: Table information
        """
        try:
            table = self.db[table_name]
            return {
                "name": table_name,
                "schema": table.schema,
                "num_rows": len(table),
                "has_index": table.has_index(),
            }
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            raise

    def __del__(self):
        """Cleanup when the service is destroyed"""
        if self.db:
            try:
                self.db.close()
            except:
                pass
