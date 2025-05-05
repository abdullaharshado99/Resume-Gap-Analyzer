import os
import asyncio
import logging
import traceback
from typing import List, Any
from jose import jwt, JWTError
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


load_dotenv()

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(sync_app: FastAPI):
    job_stores = {"default": MemoryJobStore()}
    executors = {"default": ThreadPoolExecutor(15)}
    job_defaults = {"coalesce": False, "max_instances": 10}
    scheduler = BackgroundScheduler(jobstores=job_stores, executors=executors, job_defaults=job_defaults)

    print("Starting the scheduler...")
    scheduler.start()

    sync_app.state.scheduler = scheduler

    yield

    print("Stopping the scheduler...")
    scheduler.shutdown(wait=False)


def get_scheduler():
    return app.state.scheduler


app = FastAPI(lifespan=lifespan)

SECRET_KEY = os.getenv("JWT_SECRET", "default_secret_key")
ALGORITHM = "HS256"


class QueryResponse(BaseModel):
    file_ids: List[str]


class QueryInput(BaseModel):
    query: str


class DatabaseName(BaseModel):
    name: str


class FileId(BaseModel):
    id: str


class DelFileId(BaseModel):
    del_id: List[str]


class UserId(BaseModel):
    id: str


class FileUrl(BaseModel):
    url: str


class FileName(BaseModel):
    name: str


security = HTTPBearer()


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> tuple[dict[str, Any], str]:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload, token
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")



@app.get("/")
async def welcome_message():
    return {"message": "Welcome to the Resume Gap Analyzer API!"}


@app.post("/run-data-pipeline/")
async def run_data_pipeline(
        file_url: FileUrl,
        file_name: FileName,
        file_id: FileId,
        db_name: DatabaseName,
        user_id: UserId,
        auth_tuple: tuple[dict[str, Any], str] = Depends(verify_jwt_token)
):
    user, token = auth_tuple

    asyncio.create_task(
        run_pipeline_background(file_url.url, file_name.name, file_id.id, db_name.name, user_id.id, token))

    return {"message": "Pipeline request received. Execution will start when a step is available."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)