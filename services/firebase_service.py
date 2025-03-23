from typing import Any, Dict, List, Optional, Union
from firebase_admin import firestore  # type: ignore
from loguru import logger


class FirebaseService:
    def __init__(self):
        self.db = firestore.client()

    async def create_document(
        self, collection: str, data: Dict[str, Any], document_id: Optional[str] = None
    ) -> str:
        """Create a new document in specified collection"""
        try:
            if document_id:
                doc_ref = self.db.collection(collection).document(document_id)
                doc_ref.set(data)
                return document_id
            else:
                doc_ref = self.db.collection(collection).add(data)
                return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise

    async def get_document(
        self, collection: str, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document"""
        try:
            doc_ref = self.db.collection(collection).document(document_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            raise

    async def update_document(
        self, collection: str, document_id: str, data: Dict[str, Any]
    ) -> bool:
        """Update an existing document"""
        try:
            doc_ref = self.db.collection(collection).document(document_id)
            doc_ref.update(data)
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    async def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete a document"""
        try:
            self.db.collection(collection).document(document_id).delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    async def query_documents(
        self,
        collection: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Complex query with multiple filters, ordering, and limit

        filters format: [
            {"field": "age", "op": ">=", "value": 18},
            {"field": "city", "op": "==", "value": "New York"}
        ]

        order_by format: [
            {"field": "created_at", "direction": "DESCENDING"},
            {"field": "name", "direction": "ASCENDING"}
        ]
        """
        try:
            query = self.db.collection(collection)

            # Apply filters
            if filters:
                for filter_dict in filters:
                    query = query.where(
                        filter_dict["field"], filter_dict["op"], filter_dict["value"]
                    )

            # Apply ordering
            if order_by:
                for order_dict in order_by:
                    direction = (
                        firestore.Query.DESCENDING
                        if order_dict["direction"] == "DESCENDING"
                        else firestore.Query.ASCENDING
                    )
                    query = query.order_by(order_dict["field"], direction=direction)

            # Apply limit
            if limit:
                query = query.limit(limit)

            # Execute query
            docs = query.stream()
            return [doc.to_dict() | {"id": doc.id} for doc in docs]

        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise

    async def batch_create(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Create multiple documents in a batch"""
        try:
            batch = self.db.batch()
            doc_refs = []

            for doc in documents:
                doc_ref = self.db.collection(collection).document()
                batch.set(doc_ref, doc)
                doc_refs.append(doc_ref)

            batch.commit()
            return [ref.id for ref in doc_refs]
        except Exception as e:
            logger.error(f"Error in batch creation: {e}")
            raise

    async def aggregate_query(
        self,
        collection: str,
        group_by_field: str,
        aggregate_field: str,
        operation: str = "COUNT",
    ) -> List[Dict[str, Any]]:
        """
        Perform aggregation queries
        Supported operations: COUNT, SUM, AVG, MIN, MAX
        """
        try:
            docs = self.db.collection(collection).stream()
            results = {}

            for doc in docs:
                data = doc.to_dict()
                group_value = data.get(group_by_field)
                agg_value = data.get(aggregate_field, 0)

                if group_value not in results:
                    results[group_value] = {"count": 0, "sum": 0, "values": []}

                results[group_value]["count"] += 1
                results[group_value]["sum"] += agg_value
                results[group_value]["values"].append(agg_value)

            # Process results based on operation
            final_results = []
            for group, data in results.items():
                result = {
                    group_by_field: group,
                }

                if operation == "COUNT":
                    result["value"] = data["count"]
                elif operation == "SUM":
                    result["value"] = data["sum"]
                elif operation == "AVG":
                    result["value"] = data["sum"] / data["count"]
                elif operation == "MIN":
                    result["value"] = min(data["values"])
                elif operation == "MAX":
                    result["value"] = max(data["values"])

                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Error in aggregation query: {e}")
            raise
