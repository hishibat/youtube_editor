"""動画処理サービス - OpenCVを使用した動画操作（高速化版）"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import subprocess
import shutil
from typing import Callable, Optional
import asyncio


class VideoProcessor:
    """OpenCVを使用した動画処理サービス（高速化版）"""

    def __init__(self, frame_skip: int = 2):
        """
        初期化

        Args:
            frame_skip: 処理するフレームの間隔（2=1フレームおきに処理）
        """
        self.frame_skip = frame_skip

    def get_video_info(self, filepath: str) -> dict:
        """動画ファイルの情報を取得"""
        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {filepath}")

        try:
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                if cap.get(cv2.CAP_PROP_FPS) > 0
                else 0,
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            return info
        finally:
            cap.release()

    def extract_frame(self, video_path: str, frame_time: float) -> np.ndarray:
        """指定時間のフレームを抽出"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Cannot read frame at time {frame_time}")
            return frame
        finally:
            cap.release()

    def load_background(self, path: str, target_size: tuple[int, int]) -> np.ndarray:
        """背景画像/動画を読み込み"""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".mp4", ".webm", ".gif"}:
            cap = cv2.VideoCapture(str(path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Cannot read background video: {path}")
            bg = frame
        else:
            bg = cv2.imread(str(path))
            if bg is None:
                raise ValueError(f"Cannot read background image: {path}")

        bg = cv2.resize(bg, (target_size[1], target_size[0]))
        return bg

    def load_background_from_bytes(
        self, content: bytes, target_size: tuple[int, int]
    ) -> np.ndarray:
        """バイトデータから背景を読み込み"""
        pil_image = Image.open(io.BytesIO(content))
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        bg = np.array(pil_image)[:, :, ::-1]
        bg = cv2.resize(bg, (target_size[1], target_size[0]))
        return bg

    def composite_frames(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray
    ) -> np.ndarray:
        """前景（アルファ付き）と背景を合成"""
        alpha = fg_frame[:, :, 3:4].astype(float) / 255.0
        fg_bgr = fg_frame[:, :, :3].astype(float)
        bg_float = bg_frame.astype(float)
        composite = (fg_bgr * alpha + bg_float * (1 - alpha)).astype(np.uint8)
        return composite

    def save_frame(self, frame: np.ndarray, path: str) -> None:
        """フレームを画像として保存"""
        cv2.imwrite(path, frame)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        bg_removal_service,
        background_path: str = None,
        background_bytes: bytes = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """
        動画全体を処理して背景を差し替え（高速化版）

        Args:
            input_path: 入力動画パス
            output_path: 出力動画パス
            bg_removal_service: BackgroundRemovalServiceインスタンス
            background_path: 背景ファイルパス
            background_bytes: 背景バイトデータ
            progress_callback: 進捗コールバック (current, total, status)
        """
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # RVMの時間的状態をリセット（新しい動画の開始）
        if hasattr(bg_removal_service, 'reset_temporal_state'):
            bg_removal_service.reset_temporal_state()

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 出力動画設定
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 背景を読み込み
            if background_bytes:
                bg_frame = self.load_background_from_bytes(
                    background_bytes, (height, width)
                )
            elif background_path:
                bg_frame = self.load_background(background_path, (height, width))
            else:
                bg_frame = np.zeros((height, width, 3), dtype=np.uint8)
                bg_frame[:] = (0, 177, 64)

            # 背景動画の場合
            bg_cap = None
            if background_path and Path(background_path).suffix.lower() in {
                ".mp4", ".webm", ".gif"
            }:
                bg_cap = cv2.VideoCapture(background_path)

            frame_count = 0
            processed_count = 0
            last_fg_frame = None

            if progress_callback:
                progress_callback(0, total_frames, "処理開始")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 背景動画の場合、フレームを更新
                if bg_cap is not None:
                    bg_ret, bg_video_frame = bg_cap.read()
                    if not bg_ret:
                        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        _, bg_video_frame = bg_cap.read()
                    bg_frame = cv2.resize(bg_video_frame, (width, height))

                # フレームスキップ: N フレームごとに背景除去を実行
                if frame_count % self.frame_skip == 0:
                    # 背景除去（高速版を使用）
                    fg_frame = bg_removal_service.remove_background(frame)
                    last_fg_frame = fg_frame
                    processed_count += 1
                else:
                    # スキップしたフレームは前のフレームのマスクを再利用
                    if last_fg_frame is not None:
                        # 現在のフレームに前のマスクを適用
                        alpha = last_fg_frame[:, :, 3:4]
                        fg_frame = np.dstack([frame, alpha[:, :, 0]])
                    else:
                        fg_frame = bg_removal_service.remove_background(frame)
                        last_fg_frame = fg_frame

                # 合成
                composite = self.composite_frames(fg_frame, bg_frame)
                out.write(composite)

                frame_count += 1

                # 進捗通知（10フレームごと）
                if progress_callback and frame_count % 10 == 0:
                    progress_callback(
                        frame_count,
                        total_frames,
                        f"処理中: {frame_count}/{total_frames}"
                    )

            out.release()
            if bg_cap:
                bg_cap.release()

            if progress_callback:
                progress_callback(total_frames, total_frames, "H.264変換中")

            # ffmpegでH.264に変換
            self._convert_to_h264(output_path)

            if progress_callback:
                progress_callback(total_frames, total_frames, "完了")

            print(f"Video processing complete: {output_path}")
            print(f"Processed {processed_count} frames (skipped {frame_count - processed_count})")

        finally:
            cap.release()

    def _convert_to_h264(self, video_path: str) -> None:
        """動画をH.264コーデックに変換"""
        if not shutil.which("ffmpeg"):
            print("Warning: ffmpeg not found, skipping H.264 conversion")
            return

        temp_path = video_path + ".temp.mp4"

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                Path(video_path).unlink()
                Path(temp_path).rename(video_path)
                print(f"Converted to H.264: {video_path}")
            else:
                print(f"ffmpeg conversion failed: {result.stderr}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

        except Exception as e:
            print(f"H.264 conversion error: {e}")
            if Path(temp_path).exists():
                Path(temp_path).unlink()
