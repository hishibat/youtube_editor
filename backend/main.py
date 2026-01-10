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
from services.pro_matting_service import ProMattingService
from services.video_processor import VideoProcessor
from services.hybrid_matting_service import HybridMattingService
from services.subtitle_service import SubtitleService, SubtitleEntry
from services.highlight_service import HighlightService
from services.robust_video_segmentation import RobustVideoSegmentation, KeyframeAnnotation

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

# === 背景除去サービス ===
# デフォルト: RVM（高速モード）
# ユーザーがストローク指定した場合: ProMattingService（VFXグレード）

print("Initializing background removal services...")

# RVM: 高速モード（デフォルト）
rvm_service = VideoMattingService(downsample_ratio=0.4)

# ProMattingService: VFXグレードの高精度マッティング
# SAM 2 Large + Trimap + Deep Matting + Temporal Smoothing
pro_matting_service = None  # 遅延初期化

def get_pro_matting_service():
    """ProMattingService を遅延初期化"""
    global pro_matting_service
    if pro_matting_service is None:
        print("Initializing Pro Matting Service (first use)...")
        print("  - SAM 2 Large model")
        print("  - 1024x1024 high-resolution inference")
        print("  - Deep Matting Refinement (Trimap)")
        print("  - Temporal Smoothing")
        print("  - FP16 acceleration")
        pro_matting_service = ProMattingService(
            model_size="large",
            inference_resolution=1024,
            use_fp16=True
        )
    return pro_matting_service

# デフォルトはRVMを使用
bg_removal_service = rvm_service
video_processor = VideoProcessor(frame_skip=1)  # 高精度モードではスキップなし

# ハイブリッドサービス（SAM 2 + RVM）- 商用レベル高速処理
hybrid_service = None

def get_hybrid_service():
    """HybridMattingService を遅延初期化"""
    global hybrid_service
    if hybrid_service is None:
        print("Initializing Hybrid Matting Service...")
        print("  - SAM 2 Large for keyframe")
        print("  - RVM for fast frame processing")
        print("  - FFmpeg audio preservation")
        hybrid_service = HybridMattingService()
    return hybrid_service

# 字幕サービス
subtitle_service = SubtitleService()

# ハイライト検出サービス
highlight_service = HighlightService()

# 堅牢なビデオセグメンテーション（SAM 2 VideoPredictor + 時間的一貫性）
robust_segmentation = None

def get_robust_segmentation():
    """RobustVideoSegmentation を遅延初期化"""
    global robust_segmentation
    if robust_segmentation is None:
        print("Initializing Robust Video Segmentation...")
        print("  - SAM 2 VideoPredictor for temporal consistency")
        print("  - Morphological post-processing")
        print("  - Temporal smoothing")
        robust_segmentation = RobustVideoSegmentation(model_size="large")
    return robust_segmentation

print("Default: RVM (fast mode)")

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


@app.get("/api/frame/{file_id}")
async def get_raw_frame(file_id: str, time: float = 0.0):
    """処理なしの生フレームを取得（ポイント選択用）"""
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    # 生フレームを抽出（背景除去なし）
    frame = video_processor.extract_frame(str(video_path), time)

    # JPEGとして保存
    frame_id = f"raw_{file_id}_{int(time * 1000)}"
    frame_path = OUTPUT_DIR / f"{frame_id}.jpg"
    video_processor.save_frame(frame, str(frame_path))

    return {"frame_url": f"/outputs/{frame_id}.jpg"}


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
    strokes: str = Form(None),
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

    # ストロークを解析してポイントに変換
    foreground_points = None
    background_points = None

    if strokes:
        try:
            strokes_data = json.loads(strokes)
            foreground_points = []
            background_points = []
            h, w = frame.shape[:2]

            for stroke in strokes_data:
                stroke_type = stroke.get('type', 'foreground')
                points = stroke.get('points', [])

                # ストロークの全ポイントを追加（密な空間情報）
                for point in points:
                    px = int(point['x'] * w / 100)
                    py = int(point['y'] * h / 100)

                    if stroke_type == 'foreground':
                        foreground_points.append((px, py))
                    else:
                        background_points.append((px, py))

            print(f"Strokes parsed: {len(foreground_points)} FG points, {len(background_points)} BG points")
        except Exception as e:
            print(f"Failed to parse strokes: {e}")

    # ストローク指定がある場合はProMattingを使用、なければRVM（高速）
    if foreground_points or background_points:
        print("Preview: Using Pro Matting Service (VFX grade)")
        print(f"  - FG points: {len(foreground_points) if foreground_points else 0}")
        print(f"  - BG points: {len(background_points) if background_points else 0}")
        service = get_pro_matting_service()

        # ストロークデータを渡す
        strokes_data_list = None
        if strokes:
            try:
                strokes_data_list = json.loads(strokes)
            except:
                pass

        fg_frame = service.process_single_frame(
            frame,
            foreground_points=foreground_points,
            background_points=background_points,
            strokes=strokes_data_list,
            apply_temporal_smoothing=False
        )
    else:
        print("Preview: Using RVM (fast mode)")
        fg_frame = rvm_service.remove_background(frame)

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
    strokes_json: str = None,
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
        # ストローク指定がある場合はProMattingを使用、なければRVM（高速）
        if strokes_json:
            print("Using Pro Matting Service (VFX grade) - user specified strokes")
            service = get_pro_matting_service()
        else:
            print("Using RVM (fast mode) - no strokes specified")
            service = rvm_service

        video_processor.process_video(
            input_path=video_path,
            output_path=output_path,
            bg_removal_service=service,
            background_path=bg_path,
            background_bytes=bg_bytes,
            progress_callback=progress_callback,
            strokes_json=strokes_json,
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
    strokes: str = Form(None),
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
        strokes,
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


# === ハイライト検出 API ===

@app.post("/api/detect-highlights")
async def detect_highlights(file_id: str = Form(...)):
    """動画からハイライトを自動検出"""
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    try:
        highlights = highlight_service.detect_highlights(
            str(video_path),
            min_duration=2.0,
            max_duration=5.0,
            top_k=3
        )

        return {
            "highlights": [
                {
                    "start_time": h.start_time,
                    "end_time": h.end_time,
                    "score": h.score,
                    "type": h.type
                }
                for h in highlights
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Highlight detection failed: {str(e)}")


@app.post("/api/restructure-video")
async def restructure_video(
    file_id: str = Form(...),
    hook_start: float = Form(...),
    hook_end: float = Form(...),
):
    """動画をフック構造に再構成"""
    from services.highlight_service import HighlightSegment

    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    output_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"restructured_{output_id}.mp4"

    highlight = HighlightSegment(
        start_time=hook_start,
        end_time=hook_end,
        score=1.0,
        type="manual"
    )

    try:
        success = highlight_service.restructure_video_with_hook(
            str(video_path),
            str(output_path),
            highlight,
            hook_duration=hook_end - hook_start
        )

        if not success:
            raise HTTPException(500, "Video restructure failed")

        return {
            "output_id": output_id,
            "output_url": f"/outputs/restructured_{output_id}.mp4",
            "download_url": f"/api/download-restructured/{output_id}",
        }
    except Exception as e:
        raise HTTPException(500, f"Video restructure failed: {str(e)}")


@app.get("/api/download-restructured/{output_id}")
async def download_restructured_video(output_id: str):
    """再構成済み動画をダウンロード"""
    output_path = OUTPUT_DIR / f"restructured_{output_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"restructured_{output_id}.mp4",
    )


# === 字幕 API ===

@app.post("/api/add-subtitles")
async def add_subtitles(
    file_id: str = Form(...),
    subtitles_json: str = Form(...),
):
    """動画に字幕を追加"""
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    try:
        subtitles_data = json.loads(subtitles_json)
        subtitles = [
            SubtitleEntry(
                start_time=s["start_time"],
                end_time=s["end_time"],
                text=s["text"],
                style=s.get("style", "default")
            )
            for s in subtitles_data
        ]
    except Exception as e:
        raise HTTPException(400, f"Invalid subtitles format: {str(e)}")

    output_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"subtitled_{output_id}.mp4"

    try:
        success = subtitle_service.burn_subtitles(
            str(video_path),
            str(output_path),
            subtitles
        )

        if not success:
            raise HTTPException(500, "Subtitle burn failed")

        return {
            "output_id": output_id,
            "output_url": f"/outputs/subtitled_{output_id}.mp4",
            "download_url": f"/api/download-subtitled/{output_id}",
        }
    except Exception as e:
        raise HTTPException(500, f"Subtitle burn failed: {str(e)}")


@app.get("/api/download-subtitled/{output_id}")
async def download_subtitled_video(output_id: str):
    """字幕付き動画をダウンロード"""
    output_path = OUTPUT_DIR / f"subtitled_{output_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"subtitled_{output_id}.mp4",
    )


# === 高速処理 API（ハイブリッド） ===

def process_video_fast_task(
    task_id: str,
    video_path: str,
    output_path: str,
    fg_points: list = None,
    bg_points: list = None,
    bg_path: str = None,
    bg_bytes: bytes = None,
):
    """高速動画処理タスク（SAM 2 + RVM + FFmpeg音声保持）"""
    def progress_callback(current: int, total: int, status: str, eta: float):
        percent = int((current / total) * 100) if total > 0 else 0
        progress_store[task_id] = {
            "current": current,
            "total": total,
            "status": status,
            "percent": percent,
            "eta": eta,
        }

    try:
        service = get_hybrid_service()

        service.process_video(
            input_path=video_path,
            output_path=output_path,
            foreground_points=fg_points,
            background_points=bg_points,
            background_path=bg_path,
            background_bytes=bg_bytes,
            progress_callback=progress_callback,
        )

        progress_store[task_id] = {
            "current": 100,
            "total": 100,
            "status": "完了",
            "percent": 100,
            "eta": 0,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_store[task_id] = {
            "current": 0,
            "total": 0,
            "status": f"エラー: {str(e)}",
            "percent": 0,
            "eta": 0,
        }


@app.post("/api/process-fast")
async def process_video_fast(
    file_id: str = Form(...),
    background_id: str = Form(None),
    background_file: UploadFile = File(None),
    strokes: str = Form(None),
):
    """
    高速動画処理（SAM 2 + RVM + FFmpeg音声保持）

    - キーフレームはSAM 2で高精度処理
    - 後続フレームはRVMで高速処理
    - 音声は完全保持
    """
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    # ポイントを解析
    fg_points = None
    bg_points = None

    if strokes:
        try:
            strokes_data = json.loads(strokes)
            fg_points = []
            bg_points = []

            video_info = video_processor.get_video_info(str(video_path))
            w, h = video_info["width"], video_info["height"]

            for stroke in strokes_data:
                stroke_type = stroke.get('type', 'foreground')
                points = stroke.get('points', [])

                for point in points:
                    px = int(point['x'] * w / 100)
                    py = int(point['y'] * h / 100)

                    if stroke_type == 'foreground':
                        fg_points.append((px, py))
                    else:
                        bg_points.append((px, py))

            print(f"Fast processing: {len(fg_points)} FG points, {len(bg_points)} BG points")
        except Exception as e:
            print(f"Failed to parse strokes: {e}")

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
    output_path = OUTPUT_DIR / f"fast_output_{output_id}.mp4"
    task_id = output_id

    # 進捗初期化
    progress_store[task_id] = {
        "current": 0,
        "total": 0,
        "status": "開始中",
        "percent": 0,
        "eta": 0,
    }

    # バックグラウンドで処理開始
    executor.submit(
        process_video_fast_task,
        task_id,
        str(video_path),
        str(output_path),
        fg_points,
        bg_points,
        bg_path,
        bg_bytes,
    )

    return {
        "task_id": task_id,
        "output_id": output_id,
        "output_url": f"/outputs/fast_output_{output_id}.mp4",
        "download_url": f"/api/download-fast/{output_id}",
    }


@app.get("/api/download-fast/{output_id}")
async def download_fast_video(output_id: str):
    """高速処理済み動画をダウンロード"""
    output_path = OUTPUT_DIR / f"fast_output_{output_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"edited_{output_id}.mp4",
    )


# === 堅牢なビデオセグメンテーション API ===

def process_robust_video_task(
    task_id: str,
    video_path: str,
    output_path: str,
    keyframe_annotations: list,
    bg_path: str = None,
    bg_bytes: bytes = None,
):
    """堅牢なビデオセグメンテーション処理タスク"""
    def progress_callback(current: int, total: int, status: str, eta: float):
        percent = int((current / total) * 100) if total > 0 else 0
        progress_store[task_id] = {
            "current": current,
            "total": total,
            "status": status,
            "percent": percent,
            "eta": eta,
        }

    try:
        service = get_robust_segmentation()

        # KeyframeAnnotationオブジェクトに変換
        annotations = [
            KeyframeAnnotation(
                frame_idx=ann["frame_idx"],
                foreground_points=[tuple(p) for p in ann.get("foreground_points", [])],
                background_points=[tuple(p) for p in ann.get("background_points", [])]
            )
            for ann in keyframe_annotations
        ]

        service.process_video_with_propagation(
            input_path=video_path,
            output_path=output_path,
            keyframe_annotations=annotations,
            background_path=bg_path,
            background_bytes=bg_bytes,
            progress_callback=progress_callback,
        )

        progress_store[task_id] = {
            "current": 100,
            "total": 100,
            "status": "完了",
            "percent": 100,
            "eta": 0,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_store[task_id] = {
            "current": 0,
            "total": 0,
            "status": f"エラー: {str(e)}",
            "percent": 0,
            "eta": 0,
        }


@app.post("/api/process-robust")
async def process_video_robust(
    file_id: str = Form(...),
    background_id: str = Form(None),
    background_file: UploadFile = File(None),
    keyframes_json: str = Form(None),
    strokes: str = Form(None),
):
    """
    堅牢なビデオセグメンテーション（SAM 2 VideoPredictor + 時間的一貫性）

    - 複数キーフレームでのポイント指定
    - Morphological後処理（ギターの細いパーツを保護）
    - Temporal Smoothing（フリッカー除去）
    - 音声完全保持
    """
    video_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == file_id:
            video_path = f
            break

    if not video_path:
        raise HTTPException(404, "Video not found")

    # キーフレームアノテーションを解析
    keyframe_annotations = []

    video_info = video_processor.get_video_info(str(video_path))
    w, h = video_info["width"], video_info["height"]

    if keyframes_json:
        try:
            keyframes_data = json.loads(keyframes_json)
            for kf in keyframes_data:
                fg_points = []
                bg_points = []

                for point in kf.get("foreground_points", []):
                    px = int(point["x"] * w / 100) if isinstance(point, dict) else int(point[0] * w / 100)
                    py = int(point["y"] * h / 100) if isinstance(point, dict) else int(point[1] * h / 100)
                    fg_points.append([px, py])

                for point in kf.get("background_points", []):
                    px = int(point["x"] * w / 100) if isinstance(point, dict) else int(point[0] * w / 100)
                    py = int(point["y"] * h / 100) if isinstance(point, dict) else int(point[1] * h / 100)
                    bg_points.append([px, py])

                keyframe_annotations.append({
                    "frame_idx": kf.get("frame_idx", 0),
                    "foreground_points": fg_points,
                    "background_points": bg_points
                })

            print(f"Parsed {len(keyframe_annotations)} keyframe annotations")
        except Exception as e:
            print(f"Failed to parse keyframes_json: {e}")

    # strokes から keyframe_annotations に変換（後方互換性）
    if not keyframe_annotations and strokes:
        try:
            strokes_data = json.loads(strokes)
            fg_points = []
            bg_points = []

            for stroke in strokes_data:
                stroke_type = stroke.get('type', 'foreground')
                points = stroke.get('points', [])

                for point in points:
                    px = int(point['x'] * w / 100)
                    py = int(point['y'] * h / 100)

                    if stroke_type == 'foreground':
                        fg_points.append([px, py])
                    else:
                        bg_points.append([px, py])

            if fg_points or bg_points:
                keyframe_annotations.append({
                    "frame_idx": 0,
                    "foreground_points": fg_points,
                    "background_points": bg_points
                })

            print(f"Converted strokes to keyframe: {len(fg_points)} FG, {len(bg_points)} BG points")
        except Exception as e:
            print(f"Failed to parse strokes: {e}")

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
    output_path = OUTPUT_DIR / f"robust_output_{output_id}.mp4"
    task_id = output_id

    # 進捗初期化
    progress_store[task_id] = {
        "current": 0,
        "total": 0,
        "status": "開始中",
        "percent": 0,
        "eta": 0,
    }

    # バックグラウンドで処理開始
    executor.submit(
        process_robust_video_task,
        task_id,
        str(video_path),
        str(output_path),
        keyframe_annotations,
        bg_path,
        bg_bytes,
    )

    return {
        "task_id": task_id,
        "output_id": output_id,
        "output_url": f"/outputs/robust_output_{output_id}.mp4",
        "download_url": f"/api/download-robust/{output_id}",
    }


@app.get("/api/download-robust/{output_id}")
async def download_robust_video(output_id: str):
    """堅牢処理済み動画をダウンロード"""
    output_path = OUTPUT_DIR / f"robust_output_{output_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"edited_{output_id}.mp4",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
