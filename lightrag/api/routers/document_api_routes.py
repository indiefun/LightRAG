"""
API routes for document batch insert and batch delete.
"""
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Body, HTTPException
from pydantic import BaseModel, Field, field_validator
from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger
from .document_routes import InsertResponse, ClearDocumentsResponse
import asyncio
import json
import threading
import time
import redis.asyncio as aioredis
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DATABASE = int(os.getenv("REDIS_DATABASE", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

QUEUE_KEY = "pipeline_queue"
STATUS_KEY = "pipeline_status"
WORKER_LOCK_KEY = "pipeline_worker_lock"

if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DATABASE}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DATABASE}"

# 延迟初始化 Redis 连接
redis_client = None
worker_task = None

def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.from_url(REDIS_URL)
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise HTTPException(status_code=503, detail="Redis connection failed")
    return redis_client

async def check_redis_connection():
    """检查 Redis 连接是否可用"""
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection check failed: {e}")
        return False

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

async def update_pipeline_status(status: str, message: str = ""):
    try:
        client = get_redis_client()
        await client.hset(STATUS_KEY, mapping={"status": status, "message": message, "timestamp": str(time.time())})
    except Exception as e:
        logger.error(f"Failed to update pipeline status: {e}")

async def pipeline_worker(rag: LightRAG):
    client = get_redis_client()
    await update_pipeline_status("idle", "Pipeline worker started")
    while True:
        try:
            task = await client.blpop(QUEUE_KEY, timeout=5)
            if not task:
                await update_pipeline_status("idle", "No tasks in queue")
                await asyncio.sleep(1)
                continue
            _, task_data = task
            task_obj = json.loads(task_data)
            await update_pipeline_status("busy", f"Processing {task_obj['type']} task")
            
            if task_obj["type"] == "insert":
                await process_insert(rag, task_obj["data"])
            elif task_obj["type"] == "delete":
                await process_delete(rag, task_obj["data"])
            else:
                await update_pipeline_status("error", f"Unknown task type: {task_obj['type']}")
                continue
                
            await update_pipeline_status("idle", f"Finished {task_obj['type']} task")
        except Exception as e:
            logger.error(f"Pipeline worker error: {e}")
            await update_pipeline_status("error", f"Worker error: {str(e)}")
            await asyncio.sleep(2)

async def process_insert(rag: LightRAG, data):
    # data: {"texts": [...], "sources": [...], "ids": [...]}
    try:
        texts = data.get("texts")
        if not texts:
            raise ValueError("texts is required")
            
        await rag.ainsert(
            input=texts,
            ids=data.get("ids"),
            file_paths=data.get("sources"),
        )
    except Exception as e:
        error_details = {
            "task_type": "insert",
            "task_data": data,
            "error": str(e),
            "error_type": type(e).__name__
        }
        logger.error(f"Insert task failed: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
        await update_pipeline_status("error", f"Insert error: {str(e)}")
        raise  # 重新抛出异常，让上层处理

async def process_delete(rag: LightRAG, data):
    # data: {"ids": [...]}
    try:
        ids = data.get("ids")
        if not ids:
            raise ValueError("ids is required")
            
        for doc_id in ids:
            await rag.adelete_by_doc_id(doc_id)
    except Exception as e:
        error_details = {
            "task_type": "delete",
            "task_data": data,
            "error": str(e),
            "error_type": type(e).__name__
        }
        logger.error(f"Delete task failed: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
        await update_pipeline_status("error", f"Delete error: {str(e)}")
        raise  # 重新抛出异常，让上层处理

async def ensure_worker(rag: LightRAG):
    """使用 Redis 锁确保全局只有一个 worker 运行"""
    global worker_task
    
    # 如果已经有 worker 任务在运行，直接返回
    if worker_task and not worker_task.done():
        return
    
    client = get_redis_client()
    
    # 尝试获取 worker 锁，超时时间 300 秒（5分钟）
    lock_acquired = await client.set(WORKER_LOCK_KEY, "1", ex=300, nx=True)
    
    if lock_acquired:
        # 成功获取锁，启动 worker
        try:
            worker_task = asyncio.create_task(pipeline_worker(rag))
            await worker_task
        except Exception as e:
            logger.error(f"Worker task failed: {e}")
        finally:
            # 释放锁
            try:
                await client.delete(WORKER_LOCK_KEY)
            except Exception as e:
                logger.error(f"Failed to release worker lock: {e}")
    else:
        # 锁被其他进程持有，等待一下再检查
        await asyncio.sleep(1)

def create_document_api_routes(rag: LightRAG, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("", response_model=InsertResponse, dependencies=[Depends(combined_auth)])
    async def batch_insert_documents(
        request: BatchInsertRequest
    ):
        """
        批量插入文档。
        Body: { documents: [ {text, source, id}, ... ] }
        """
        # 检查 Redis 连接
        if not await check_redis_connection():
            raise HTTPException(status_code=503, detail="Redis service unavailable")
        
        # 异步启动 worker（如果还没有运行）
        asyncio.create_task(ensure_worker(rag))
        
        try:
            client = get_redis_client()
            texts = [doc.text for doc in request.documents]
            sources = [doc.source for doc in request.documents]
            ids = [doc.id for doc in request.documents]
            task = {"type": "insert", "data": {"texts": texts, "sources": sources, "ids": ids}}
            await client.rpush(QUEUE_KEY, json.dumps(task))
            return InsertResponse(
                status="success",
                message="Text successfully enqueued. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Failed to enqueue insert task: {e}")
            raise HTTPException(status_code=500, detail="Failed to enqueue task")

    @router.delete("", response_model=ClearDocumentsResponse, dependencies=[Depends(combined_auth)])
    async def batch_delete_documents(
        request: BatchDeleteRequest = Body(...)
    ):
        """
        批量删除指定ID的文档。
        Body: { ids: [id1, id2, ...] }
        """
        # 检查 Redis 连接
        if not await check_redis_connection():
            raise HTTPException(status_code=503, detail="Redis service unavailable")
        
        # 异步启动 worker（如果还没有运行）
        asyncio.create_task(ensure_worker(rag))
        
        try:
            client = get_redis_client()
            task = {"type": "delete", "data": {"ids": request.ids}}
            await client.rpush(QUEUE_KEY, json.dumps(task))
            return ClearDocumentsResponse(
                status="success",
                message="Delete request enqueued. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Failed to enqueue delete task: {e}")
            raise HTTPException(status_code=500, detail="Failed to enqueue task")

    @router.get("/pipeline_status")
    async def get_pipeline_status():
        try:
            client = get_redis_client()
            status = await client.hgetall(STATUS_KEY)
            return status
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            raise HTTPException(status_code=503, detail="Failed to get pipeline status")

    return router 