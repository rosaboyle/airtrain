from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import numpy as np
import pyarrow as pa


class VectorData(BaseModel):
    """Schema for vector data"""

    vector: List[float] = Field(..., description="Vector data as list of floats")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata associated with the vector"
    )


class TableConfig(BaseModel):
    """Configuration for table creation"""

    name: str = Field(..., description="Name of the table")
    vector_dim: int = Field(..., description="Dimension of vectors to be stored")
    schema: Optional[Dict[str, str]] = Field(
        default=None, description="Optional schema definition for additional columns"
    )
    mode: str = Field(
        default="create",
        description="Table creation mode: 'create', 'overwrite', or 'append'",
    )


class SearchConfig(BaseModel):
    """Configuration for vector search"""

    table_name: str = Field(..., description="Name of the table to search in")
    query_vector: List[float] = Field(
        ..., description="Query vector for similarity search"
    )
    metric: str = Field(
        default="cosine", description="Distance metric: 'cosine', 'l2', or 'dot'"
    )
    k: int = Field(default=10, description="Number of results to return")
    filter_expr: Optional[str] = Field(
        default=None, description="Optional filter expression"
    )


class BatchConfig(BaseModel):
    """Configuration for batch operations"""

    table_name: str = Field(..., description="Name of the table")
    batch_size: int = Field(default=1000, description="Size of batches for processing")


class IndexConfig(BaseModel):
    """Configuration for index creation"""

    table_name: str = Field(..., description="Name of the table")
    index_type: str = Field(
        default="hnsw", description="Type of index: 'hnsw' or 'ivf_pq'"
    )
    metric: str = Field(default="cosine", description="Distance metric for index")
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional index parameters (should not include metric)",
    )


class SearchResult(BaseModel):
    """Schema for search results"""

    vector: List[float] = Field(..., description="Retrieved vector")
    distance: float = Field(..., description="Distance from query vector")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Associated metadata"
    )


class BatchResult(BaseModel):
    """Schema for batch operation results"""

    success: bool = Field(..., description="Whether the operation was successful")
    affected_rows: int = Field(..., description="Number of rows affected")
    error_message: Optional[str] = Field(
        default=None, description="Error message if operation failed"
    )
