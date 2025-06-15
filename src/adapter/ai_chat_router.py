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

# 文件上传路径
ZIP_INPUT_DIR = os.getenv('ZIP_INPUT_DIR', 'data/input/zip')
CODE_DIR = os.getenv('CODE.INPUT_DIR', 'data/input/code')
DATASET_DIR = os.getenv('DATASET.INPUT_DIR', 'data/input/dataset')
# 文件保存路径
DATASET_ZIP_PATH = os.path.join(ZIP_INPUT_DIR, "data.zip")
CODE_ZIP_PATH = os.path.join(ZIP_INPUT_DIR, "code.zip")


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

@router.post("/upload/{mod}/{upload_type}", summary="上传模型或数据集")
async def upload(
    mod: str = Path(..., description="模型或数据集['model'|'dataset']"),
    upload_type: str = Path(..., description="上传方式['zip'|'folder'|'files']"),
    filename: str = Body(..., description="文件路径"),
    file: UploadFile = File(..., description="模型或数据文件")
) -> JSONResponse:
    print(mod, upload_type, filename)
    if mod in ['model', 'code']:
        base_dir = CODE_DIR
        zip_path = CODE_ZIP_PATH
    elif mod == 'dataset':
        base_dir = DATASET_DIR
        zip_path = DATASET_ZIP_PATH
    else:
        raise ValueError('mod must be model/code or dataset')
    os.makedirs(base_dir, exist_ok=True)
    if upload_type == 'zip':
        os.makedirs(ZIP_INPUT_DIR, exist_ok=True)
        with open(zip_path, "wb") as f:
            f.write(await file.read())
        shutil.rmtree(base_dir, ignore_errors=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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

