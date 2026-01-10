import os
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
import json
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from services.background_removal import BackgroundRemovalService
from services.video_matting import VideoMattingService
from services.sam2_video_matting import SAM2VideoMattingService
from services.video_processor import VideoProcessor

app = FastAPI(title="YouTube Shorts Editor API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ディレクトリ設定
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"

# ディレクトリ作成
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
BACKGROUNDS_DIR.mkdir(exist_ok=True)

# 静的ファイル配信
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# === 背景除去サービス選択 ===
# True: SAM 2 (高品質・低速) - ギター演奏動画に最適
# False: RVM (中品質・高速) - 一般的な動画向け
USE_SAM2 = True

if USE_SAM2:
    # SAM 2: Segment Anything Model 2
    # - ポイント指定型追跡（人物+ギター）
    # - Hough変換によるギター指板保護
    # - オプティカルフローによる振動検知
    print("Using SAM 2 for background removal (high quality mode)")
    bg_removal_service = SAM2VideoMattingService(
        model_size="large"  # tiny, small, base_plus, large
    )
    video_processor = VideoProcessor(frame_skip=1)  # SAM 2は毎フレーム処理（品質重視）
else:
    # RVM: Robust Video Matting
    # - ビデオ専用設計、時間的一貫性
    print("Using RVM for background removal (fast mode)")
    bg_removal_service = VideoMattingService(
        downsample_ratio=0.4
    )
    video_processor = VideoProcessor(frame_skip=2)

# 進捗管理
progress_store: Dict[str, dict] = {}
# WebSocket接続管理
websocket_connections: Dict[str, WebSocket] = {}
# スレッドプール
executor = ThreadPoolExecutor(max_workers=2)


@app.get("/")
async def root():
    return {"message": "YouTube Shorts Editor API", "status": "running"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """動画ファイルをアップロード"""
    allowed_extensions = {".mp4", ".webm", ".mov", ".avi"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file format. Allowed: {allowed_extensions}")

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    filepath = UPLOAD_DIR / filename

    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)

    video_info = video_processor.get_video_info(str(filepath))

    return {
        "file_id": file_id,
        "filename": filename,
        "original_name": file.filename,
        "path": str(filepath),
        "info": video_info,
    }


@app.get("/api/backgrounds")
async def list_backgrounds():
    """利用可能な背景一覧を取得"""
    backgrounds = []
    categories = {"funny": "ネタ系", "stylish": "おしゃれ系", "realistic": "実写系"}

    for category_dir in BACKGROUNDS_DIR.iterdir():
        if category_dir.is_dir():
            category_name = categories.get(category_dir.name, category_dir.name)
            for bg_file in category_dir.iterdir():
                if bg_file.suffix.lower() in {".mp4", ".webm", ".gif", ".jpg", ".png"}:
                    backgrounds.append({
                        "id": bg_file.stem,
                        "name": bg_file.stem.replace("_", " ").title(),
                        "category": category_name,
                        "category_id": category_dir.name,
                        "path": f"/backgrounds/{category_dir.name}/{bg_file.name}",
                        "type": "video" if bg_file.suffix.lower() in {".mp4", ".webm", ".gif"} else "image",
                    })

    return {"backgrounds": backgrounds}


@app.post("/api/preview")
async def generate_preview(
    file_id: str = Form(...),
    background_id: str = Form(None),
    background_file: UploadFile = File(None),
    frame_time: float = Form(0.0),
):
    """1フレームのプレビューを生成"""
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    frame = video_processor.extract_frame(str(video_path), frame_time)
    fg_frame = bg_removal_service.remove_background(frame)

    if background_file:
        bg_content = await background_file.read()
        bg_frame = video_processor.load_background_from_bytes(bg_content, frame.shape[:2])
    elif background_id:
        bg_path = find_background_by_id(background_id)
        if not bg_path:
            raise HTTPException(404, "Background not found")
        bg_frame = video_processor.load_background(str(bg_path), frame.shape[:2])
    else:
        import numpy as np
        bg_frame = np.zeros_like(frame)
        bg_frame[:] = (0, 177, 64)

    composite = video_processor.composite_frames(fg_frame, bg_frame)

    preview_id = str(uuid.uuid4())
    preview_path = OUTPUT_DIR / f"preview_{preview_id}.jpg"
    video_processor.save_frame(composite, str(preview_path))

    return {"preview_id": preview_id, "preview_url": f"/outputs/preview_{preview_id}.jpg"}


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """進捗をWebSocketで送信"""
    await websocket.accept()
    websocket_connections[task_id] = websocket

    try:
        while True:
            # 進捗を確認
            if task_id in progress_store:
                progress = progress_store[task_id]
                await websocket.send_json(progress)

                if progress.get("status") == "完了" or progress.get("status") == "エラー":
                    break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in websocket_connections:
            del websocket_connections[task_id]


@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """進捗をポーリングで取得"""
    if task_id not in progress_store:
        return {"current": 0, "total": 0, "status": "待機中", "percent": 0}

    progress = progress_store[task_id]
    return progress


def process_video_task(
    task_id: str,
    video_path: str,
    output_path: str,
    bg_path: str = None,
    bg_bytes: bytes = None,
):
    """バックグラウンドで動画処理を実行"""
    def progress_callback(current: int, total: int, status: str):
        percent = int((current / total) * 100) if total > 0 else 0
        progress_store[task_id] = {
            "current": current,
            "total": total,
            "status": status,
            "percent": percent,
        }

    try:
        video_processor.process_video(
            input_path=video_path,
            output_path=output_path,
            bg_removal_service=bg_removal_service,
            background_path=bg_path,
            background_bytes=bg_bytes,
            progress_callback=progress_callback,
        )
        progress_store[task_id] = {
            "current": 100,
            "total": 100,
            "status": "完了",
            "percent": 100,
        }
    except Exception as e:
        progress_store[task_id] = {
            "current": 0,
            "total": 0,
            "status": f"エラー: {str(e)}",
            "percent": 0,
        }


@app.post("/api/process")
async def process_video(
    file_id: str = Form(...),
    background_id: str = Form(None),
    background_file: UploadFile = File(None),
):
    """動画全体を処理して背景を差し替え（非同期）"""
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    bg_path = None
    bg_bytes = None

    if background_file:
        bg_bytes = await background_file.read()
    elif background_id:
        bg_path = find_background_by_id(background_id)
        if not bg_path:
            raise HTTPException(404, "Background not found")
        bg_path = str(bg_path)

    output_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"output_{output_id}.mp4"
    task_id = output_id

    # 進捗初期化
    progress_store[task_id] = {
        "current": 0,
        "total": 0,
        "status": "開始中",
        "percent": 0,
    }

    # バックグラウンドで処理開始
    executor.submit(
        process_video_task,
        task_id,
        str(video_path),
        str(output_path),
        bg_path,
        bg_bytes,
    )

    return {
        "task_id": task_id,
        "output_id": output_id,
        "output_url": f"/outputs/output_{output_id}.mp4",
        "download_url": f"/api/download/{output_id}",
    }


@app.get("/api/download/{output_id}")
async def download_video(output_id: str):
    """処理済み動画をダウンロード"""
    output_path = OUTPUT_DIR / f"output_{output_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"edited_{output_id}.mp4",
    )


def find_background_by_id(background_id: str) -> Path | None:
    """IDから背景ファイルを検索"""
    for category_dir in BACKGROUNDS_DIR.iterdir():
        if category_dir.is_dir():
            for bg_file in category_dir.iterdir():
                if bg_file.stem == background_id:
                    return bg_file
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
