import os
import shutil
import zipfile

from fastapi import APIRouter, UploadFile, File, Path, Body
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.responses import JSONResponse

from src.adapter.vo.ai_chat_model import ChatInputVO
from src.app.ai_chat_service import chat_handler

from src.utils.SnowFlake import Snowflake

router = APIRouter(
    prefix="/aichat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)
sf = Snowflake(worker_id=0, datacenter_id=0)

# 使用相对路径
DATA_DIR = "data"
MODULE_DIR = "data/model"
DATASET_DIR = "data/dataset"
os.makedirs(MODULE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
# 文件保存路径
DATA_ZIP_PATH = os.path.join(DATA_DIR, "data.zip")
MODEL_ZIP_PATH = os.path.join(DATA_DIR, "model.zip")


@router.post("/chat", summary="会话列表")
async def chat(request: ChatInputVO) -> EventSourceResponse:
    return EventSourceResponse(
        content=chat_handler(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        ping_message_factory=lambda: ServerSentEvent(
            event="ping", retry=15000, data=""
        ),
    )


@router.post("/upload/data", summary="上传 data.zip")
async def upload_data(data_file: UploadFile = File(..., description="数据包ZIP文件")) -> JSONResponse:
    # 保存 data_file 到 data/data.zip
    with open(DATA_ZIP_PATH, "wb") as f:
        f.write(await data_file.read())

    return JSONResponse(content={"message": "Data file uploaded successfully.", "path": DATA_ZIP_PATH})


@router.post("/upload/model", summary="上传 model.zip")
async def upload_model(model_file: UploadFile = File(..., description="模型包ZIP文件")) -> JSONResponse:
    # 保存 model_file 到 data/model.zip
    with open(MODEL_ZIP_PATH, "wb") as f:
        f.write(await model_file.read())

    return JSONResponse(content={"message": "Model file uploaded successfully.", "path": MODEL_ZIP_PATH})


@router.post("/upload/{mod}/{upload_type}", summary="上传模型或数据集")
async def upload(
    mod: str = Path(..., description="模型或数据集['model'|'dataset']"),
    upload_type: str = Path(..., description="上传方式['zip'|'folder'|'files']"),
    filename: str = Body(..., description="文件路径"),
    file: UploadFile = File(..., description="模型或数据文件")
) -> JSONResponse:
    print(mod, upload_type, filename)
    base_dir = os.path.join(DATA_DIR, mod)
    if upload_type == 'zip':
        zip_filepath = os.path.join(DATA_DIR, f'{mod}.zip')
        with open(zip_filepath, "wb") as f:
            f.write(await file.read())
        shutil.rmtree(base_dir, ignore_errors=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    elif upload_type == 'folder':
        if '/' in filename:
            os.makedirs(os.path.join(base_dir, filename[:filename.rfind('/')]), exist_ok=True)
        with open(os.path.join(base_dir, filename), "wb") as f:
            f.write(await file.read())
    elif upload_type == 'files':
        with open(os.path.join(base_dir, filename), "wb") as f:
            f.write(await file.read())
    return JSONResponse(content={"message": "File uploaded successfully.", "path": base_dir})

