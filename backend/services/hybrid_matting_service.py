"""
ハイブリッドマッティングサービス - 商用レベル高速処理

アーキテクチャ:
1. SAM 2 Large - キーフレーム（最初の1フレーム）の高精度セグメンテーション
2. RVM - 後続フレームの高速処理（30fps以上）
3. FFmpeg - 音声保持と最終エンコード

処理時間: 60秒動画を約2-3分で処理（GPU使用時）
"""

import cv2
import numpy as np
import torch
import os
import tempfile
import shutil
import subprocess
import time
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path


def guided_filter_fallback(guide: np.ndarray, src: np.ndarray, radius: int = 16, eps: float = 1e-4) -> np.ndarray:
    """
    純粋なcv2関数のみで実装したGuided Filter（フォールバック用）

    cv2.ximgprocが利用できない場合に使用

    Args:
        guide: ガイド画像（グレースケール）
        src: ソース画像（フィルタリング対象）
        radius: ウィンドウ半径
        eps: 正則化パラメータ

    Returns:
        フィルタリング結果
    """
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)

    # Box filterを使用（cv2.blurで代用）
    ksize = 2 * radius + 1

    mean_g = cv2.blur(guide, (ksize, ksize))
    mean_s = cv2.blur(src, (ksize, ksize))
    corr_g = cv2.blur(guide * guide, (ksize, ksize))
    corr_gs = cv2.blur(guide * src, (ksize, ksize))

    var_g = corr_g - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.blur(a, (ksize, ksize))
    mean_b = cv2.blur(b, (ksize, ksize))

    return mean_a * guide + mean_b


def try_guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 16, eps: float = 1e-4) -> np.ndarray:
    """
    Guided Filterを適用（ximgproc利用可能ならそちらを使用、なければフォールバック）
    """
    try:
        # opencv-contrib-pythonがあれば使用
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps)
    except AttributeError:
        # フォールバック実装を使用
        return guided_filter_fallback(guide, src, radius, eps)


class HybridMattingService:
    """
    SAM 2 + RVM ハイブリッドマッティングサービス

    - キーフレーム: SAM 2 Large で高精度マスク生成
    - 後続フレーム: RVM で高速処理（30fps以上）
    - 結果: 商用レベルの品質と速度を両立
    """

    def __init__(
        self,
        device: Optional[str] = None,
        rvm_downsample_ratio: float = 0.4
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rvm_downsample_ratio = rvm_downsample_ratio

        # SAM 2 Predictor（遅延初期化）
        self.sam2_predictor = None

        # RVM（遅延初期化）
        self.rvm_model = None
        self.rvm_rec = None  # RVMの時間的状態

        # FFmpeg
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe"

        print(f"HybridMattingService initialized (device: {self.device})")

    def _load_sam2(self):
        """SAM 2 Large を読み込み"""
        if self.sam2_predictor is not None:
            return

        print("Loading SAM 2 Large for keyframe processing...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download

            repo_id = "facebook/sam2-hiera-large"
            ckpt_name = "sam2_hiera_large.pt"
            config_name = "sam2_hiera_l.yaml"

            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

            import sam2
            sam2_path = Path(sam2.__file__).parent
            config_path = sam2_path / "configs" / "sam2" / config_name

            if not config_path.exists():
                config_path = sam2_path / "sam2_configs" / config_name

            sam2_model = build_sam2(
                str(config_path),
                checkpoint_path,
                device=self.device
            )

            if self.device == "cuda":
                sam2_model = sam2_model.half()

            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM 2 Large loaded successfully")

        except Exception as e:
            print(f"Failed to load SAM 2: {e}")
            raise

    def _load_rvm(self):
        """RVM を読み込み"""
        if self.rvm_model is not None:
            return

        print("Loading RVM for fast frame processing...")

        try:
            from huggingface_hub import hf_hub_download

            # RVMモデルをダウンロード
            model_path = hf_hub_download(
                repo_id="briaai/RMBG-1.4",
                filename="model.pth",
            )

            # 代替: RVM専用モデル
            # model_path = hf_hub_download(
            #     repo_id="PeterL1n/RobustVideoMatting",
            #     filename="rvm_mobilenetv3.pth"
            # )

            # RVMは実際にはtorch.hub経由で読み込み
            self.rvm_model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3",
                trust_repo=True
            )

            self.rvm_model = self.rvm_model.to(self.device)
            self.rvm_model.eval()

            if self.device == "cuda":
                self.rvm_model = self.rvm_model.half()

            print("RVM loaded successfully")

        except Exception as e:
            print(f"Failed to load RVM from torch.hub: {e}")
            print("Trying alternative loading method...")

            try:
                # 代替方法: VideoMattingServiceを使用
                from services.video_matting import VideoMattingService
                self._rvm_service = VideoMattingService(
                    downsample_ratio=self.rvm_downsample_ratio
                )
                self.rvm_model = "external"
                print("Using VideoMattingService as RVM backend")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise

    def reset_temporal_state(self):
        """RVMの時間的状態をリセット"""
        self.rvm_rec = None
        if hasattr(self, '_rvm_service'):
            self._rvm_service.reset_temporal_state()

    def segment_keyframe(
        self,
        frame: np.ndarray,
        foreground_points: List[Tuple[int, int]],
        background_points: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        SAM 2でキーフレームをセグメント

        Args:
            frame: BGR画像
            foreground_points: 前景ポイント
            background_points: 背景ポイント

        Returns:
            アルファマスク（0-255）
        """
        self._load_sam2()

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ポイントとラベルを準備
        all_points = []
        all_labels = []

        for point in foreground_points:
            all_points.append(point)
            all_labels.append(1)

        for point in (background_points or []):
            all_points.append(point)
            all_labels.append(0)

        if not all_points:
            # ポイントがない場合は中央を前景として推定
            all_points = [(w // 2, h // 2)]
            all_labels = [1]

        points_array = np.array(all_points)
        labels_array = np.array(all_labels)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    self.sam2_predictor.set_image(frame_rgb)
                    masks, scores, _ = self.sam2_predictor.predict(
                        point_coords=points_array,
                        point_labels=labels_array,
                        multimask_output=True
                    )
            else:
                self.sam2_predictor.set_image(frame_rgb)
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=points_array,
                    point_labels=labels_array,
                    multimask_output=True
                )

        # 最高スコアのマスクを選択
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # Guided Filterで精緻化（フォールバック対応）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_float = mask.astype(np.float32)

        refined = try_guided_filter(gray, mask_float, radius=16, eps=1e-4)

        return (np.clip(refined, 0, 1) * 255).astype(np.uint8)

    def process_frame_rvm(self, frame: np.ndarray) -> np.ndarray:
        """
        RVMで単一フレームを処理

        Args:
            frame: BGR画像

        Returns:
            BGRA画像
        """
        self._load_rvm()

        # 外部サービスを使用
        if self.rvm_model == "external":
            return self._rvm_service.remove_background(frame)

        # 直接RVMを使用
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # テンソルに変換
        src = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255

        if self.device == "cuda":
            src = src.half().cuda()

        with torch.inference_mode():
            fgr, pha, *self.rvm_rec = self.rvm_model(
                src,
                *self.rvm_rec if self.rvm_rec else [None] * 4,
                downsample_ratio=self.rvm_downsample_ratio
            )

        # アルファマスクを取得
        alpha = pha[0, 0].cpu().numpy()
        alpha = cv2.resize(alpha, (w, h))
        alpha = (alpha * 255).astype(np.uint8)

        # BGRA画像を作成
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        return bgra

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        互換性用インターフェース
        """
        return self.process_frame_rvm(frame)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        foreground_points: Optional[List[Tuple[int, int]]] = None,
        background_points: Optional[List[Tuple[int, int]]] = None,
        background_path: Optional[str] = None,
        background_bytes: Optional[bytes] = None,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None
    ) -> None:
        """
        動画全体を処理（SAM 2 + RVM + FFmpeg音声保持）

        Args:
            input_path: 入力動画パス
            output_path: 出力動画パス
            foreground_points: 前景ポイント
            background_points: 背景ポイント
            background_path: 背景ファイルパス
            background_bytes: 背景バイトデータ
            progress_callback: 進捗コールバック (current, total, status, eta_seconds)
        """
        print(f"=== Hybrid Video Processing ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # 時間的状態をリセット
        self.reset_temporal_state()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

            # 一時ディレクトリ
            temp_dir = tempfile.mkdtemp()
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            temp_audio = os.path.join(temp_dir, "temp_audio.aac")

            try:
                # 音声抽出
                has_audio = self._has_audio(input_path)
                if has_audio:
                    print("Extracting audio...")
                    self._extract_audio(input_path, temp_audio)
                else:
                    print("No audio track found")

                # 出力動画設定
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

                # 背景を読み込み
                bg_frame = self._load_background(
                    width, height, background_path, background_bytes
                )

                # 背景動画の場合
                bg_cap = None
                if background_path and Path(background_path).suffix.lower() in {".mp4", ".webm", ".gif"}:
                    bg_cap = cv2.VideoCapture(background_path)

                frame_count = 0
                start_time = time.time()
                last_report_time = start_time
                keyframe_mask = None

                if progress_callback:
                    progress_callback(0, total_frames, "処理開始", 0)

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

                    if frame_count == 0 and foreground_points:
                        # キーフレーム: SAM 2で高精度処理
                        print(f"Keyframe: Using SAM 2 with {len(foreground_points)} FG points")
                        keyframe_mask = self.segment_keyframe(
                            frame, foreground_points, background_points
                        )

                        # 最初のフレームは SAM 2 のマスクを使用
                        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                        bgra[:, :, 3] = keyframe_mask
                    else:
                        # 後続フレーム: RVMで高速処理
                        bgra = self.process_frame_rvm(frame)

                    # 合成
                    composite = self._composite_frames(bgra, bg_frame)
                    out.write(composite)

                    frame_count += 1

                    # 進捗通知（100msごと）
                    current_time = time.time()
                    if progress_callback and (current_time - last_report_time >= 0.1 or frame_count == total_frames):
                        elapsed = current_time - start_time
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        remaining_frames = total_frames - frame_count
                        eta = remaining_frames / fps_actual if fps_actual > 0 else 0

                        progress_callback(
                            frame_count,
                            total_frames,
                            f"処理中: {frame_count}/{total_frames} ({fps_actual:.1f}fps)",
                            eta
                        )
                        last_report_time = current_time

                out.release()
                if bg_cap:
                    bg_cap.release()

                # H.264変換
                if progress_callback:
                    progress_callback(total_frames, total_frames, "H.264変換中...", 0)

                h264_video = os.path.join(temp_dir, "h264_video.mp4")
                self._convert_to_h264(temp_video, h264_video)

                # 音声結合
                if has_audio and os.path.exists(temp_audio):
                    if progress_callback:
                        progress_callback(total_frames, total_frames, "音声結合中...", 0)
                    self._merge_video_audio(h264_video, temp_audio, output_path)
                else:
                    shutil.copy(h264_video, output_path)

                if progress_callback:
                    progress_callback(total_frames, total_frames, "完了", 0)

                total_time = time.time() - start_time
                print(f"Processing complete: {total_time:.1f}s ({total_frames/total_time:.1f}fps)")

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        finally:
            cap.release()

    def _has_audio(self, filepath: str) -> bool:
        """動画に音声トラックがあるか確認"""
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return "audio" in result.stdout.lower()
        except Exception:
            return False

    def _extract_audio(self, video_path: str, output_path: str) -> bool:
        """動画から音声を抽出"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "copy",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return False

    def _merge_video_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """動画と音声を結合"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Video-audio merge failed: {e}")
            return False

    def _convert_to_h264(self, input_path: str, output_path: str) -> bool:
        """H.264に変換"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"H.264 conversion failed: {e}")
            return False

    def _load_background(
        self,
        width: int,
        height: int,
        background_path: Optional[str] = None,
        background_bytes: Optional[bytes] = None
    ) -> np.ndarray:
        """背景を読み込み"""
        if background_bytes:
            from PIL import Image
            import io
            pil_image = Image.open(io.BytesIO(background_bytes))
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")
            bg = np.array(pil_image)[:, :, ::-1]
            return cv2.resize(bg, (width, height))

        if background_path:
            path = Path(background_path)
            suffix = path.suffix.lower()

            if suffix in {".mp4", ".webm", ".gif"}:
                cap = cv2.VideoCapture(str(path))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.resize(frame, (width, height))
            else:
                bg = cv2.imread(str(path))
                if bg is not None:
                    return cv2.resize(bg, (width, height))

        # デフォルト: グリーン
        bg_frame = np.zeros((height, width, 3), dtype=np.uint8)
        bg_frame[:] = (0, 177, 64)
        return bg_frame

    def _composite_frames(self, fg_frame: np.ndarray, bg_frame: np.ndarray) -> np.ndarray:
        """前景（アルファ付き）と背景を合成"""
        alpha = fg_frame[:, :, 3:4].astype(float) / 255.0
        fg_bgr = fg_frame[:, :, :3].astype(float)
        bg_float = bg_frame.astype(float)
        composite = (fg_bgr * alpha + bg_float * (1 - alpha)).astype(np.uint8)
        return composite
