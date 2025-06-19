"""
API routes for document batch insert and batch delete.
"""
from typing import List, Optional, Any
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Body
from pydantic import BaseModel, Field, field_validator
from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import DocStatus
from .document_routes import InsertResponse, ClearDocumentsResponse
import traceback
from ..config import global_args

router = APIRouter(
    prefix="/api/documents",
    tags=["api_documents"],
)

class DocumentItem(BaseModel):
    text: str = Field(min_length=1, description="The text to insert")
    source: Optional[str] = Field(default=None, description="File Source")
    id: str = Field(min_length=1, description="Document ID")

    @field_validator("text", mode="after")
    @classmethod
    def strip_text_after(cls, text: str) -> str:
        return text.strip()

    @field_validator("source", mode="after")
    @classmethod
    def strip_source_after(cls, source: str) -> str:
        return source.strip() if source else source

    @field_validator("id", mode="after")
    @classmethod
    def strip_id_after(cls, id_: str) -> str:
        return id_.strip()

class BatchInsertRequest(BaseModel):
    documents: List[DocumentItem] = Field(..., description="List of documents to insert")

class BatchDeleteRequest(BaseModel):
    ids: List[str] = Field(..., description="List of document IDs to delete")

def create_document_api_routes(rag: LightRAG, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("", response_model=InsertResponse, dependencies=[Depends(combined_auth)])
    async def batch_insert_documents(
        request: BatchInsertRequest, background_tasks: BackgroundTasks
    ):
        """
        批量插入文档。
        Body: { documents: [ {text, source, id}, ... ] }
        """
        try:
            texts = [doc.text for doc in request.documents]
            sources = [doc.source for doc in request.documents]
            ids = [doc.id for doc in request.documents]
            async def do_insert():
                await rag.apipeline_enqueue_documents(input=texts, file_paths=sources, ids=ids)
                await rag.apipeline_process_enqueue_documents()
            background_tasks.add_task(do_insert)
            return InsertResponse(
                status="success",
                message="Text successfully received. Processing will continue in background.",
            )
        except Exception as e:
            import lightrag.utils as utils
            utils.logger.error(f"Error /api/documents: {str(e)}")
            utils.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("", response_model=ClearDocumentsResponse, dependencies=[Depends(combined_auth)])
    async def batch_delete_documents(
        request: BatchDeleteRequest = Body(...)
    ):
        """
        批量删除指定ID的文档。
        Body: { ids: [id1, id2, ...] }
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        async with pipeline_status_lock:
            if pipeline_status.get("busy", False):
                return ClearDocumentsResponse(
                    status="busy",
                    message="Cannot clear documents while pipeline is busy",
                )
            pipeline_status.update(
                {
                    "busy": True,
                    "job_name": "Deleting Documents",
                    "job_start": __import__('datetime').datetime.now().isoformat(),
                    "docs": len(request.ids),
                    "batchs": 1,
                    "cur_batch": 0,
                    "request_pending": False,
                    "latest_message": "Starting document deletion process",
                }
            )
            del pipeline_status["history_messages"][:]
            pipeline_status["history_messages"].append(
                f"Starting deletion of {len(request.ids)} documents"
            )

        try:
            deleted_count = 0
            failed_count = 0
            errors = []
            for doc_id in request.ids:
                try:
                    await rag.adelete_by_doc_id(doc_id)
                    deleted_count += 1
                except Exception as e:
                    failed_count += 1
                    errors.append(f"{doc_id}: {str(e)}")
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(
                    f"Deleted {deleted_count} documents, failed to delete {failed_count}"
                )
            if failed_count == 0:
                status = "success"
                message = f"All {deleted_count} documents deleted successfully."
            elif deleted_count > 0:
                status = "partial_success"
                message = f"Deleted {deleted_count} documents, failed to delete {failed_count}. Errors: {errors}"
            else:
                status = "fail"
                message = f"Failed to delete any documents. Errors: {errors}"
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(message)
            return ClearDocumentsResponse(status=status, message=message)
        except Exception as e:
            import lightrag.utils as utils
            error_msg = f"Error deleting documents: {str(e)}"
            utils.logger.error(error_msg)
            utils.logger.error(traceback.format_exc())
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(error_msg)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                completion_msg = "Document deletion process completed"
                pipeline_status["latest_message"] = completion_msg
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(completion_msg)

    return router 