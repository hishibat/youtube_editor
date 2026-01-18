"""
堅牢なビデオセグメンテーションサービス

特徴:
1. SAM 2 VideoPredictor による時間的一貫性のあるマスク伝播
2. マルチキーフレームサポート（任意のフレームでポイント追加可能）
3. Morphological後処理（ギターの細いパーツを保護）
4. Temporal Smoothing（フリッカー除去）
5. 接続成分解析（人物と接続された物体を保護）
"""

import cv2
import numpy as np
import torch
import os
import tempfile
import shutil
import subprocess
import time
from typing import Optional, Tuple, List, Dict, Callable, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field


@dataclass
class KeyframeAnnotation:
    """キーフレームのアノテーション"""
    frame_idx: int
    foreground_points: List[Tuple[int, int]] = field(default_factory=list)
    background_points: List[Tuple[int, int]] = field(default_factory=list)


class MaskPostProcessor:
    """
    マスク後処理クラス

    - Morphological Operations（細いパーツを保護）
    - Connected Component Analysis（接続成分を保護）
    - Temporal Smoothing（フリッカー除去）
    - Guided Filter（エッジ精緻化）
    """

    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.mask_buffer: deque = deque(maxlen=buffer_size)
        # Guided Filter が使えるか確認
        self.has_ximgproc = hasattr(cv2, 'ximgproc')
        if self.has_ximgproc:
            print("  Guided Filter: available (opencv-contrib)")
        else:
            print("  Guided Filter: NOT available (using fallback)")

    def apply_morphological_ops(
        self,
        mask: np.ndarray,
        close_kernel_size: int = 15,  # 増加: ギター指板の穴を埋める
        dilate_kernel_size: int = 5,   # 増加: 細いパーツ保護
        dilate_iterations: int = 2     # 増加: より強い保護
    ) -> np.ndarray:
        """
        Morphological操作を適用

        - Closing: 小さな穴を埋める（ギターの弦の隙間、フレット間など）
        - Dilation: エッジを少し膨張（細いパーツが消えるのを防ぐ）
        """
        # 複数回のClosing（穴埋め強化）
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # 小さい穴も埋める
        small_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_close_kernel, iterations=1)

        # 軽いDilation（細いパーツ保護）
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)

        return mask

    def apply_guided_filter(
        self,
        mask: np.ndarray,
        guide_image: Optional[np.ndarray] = None,
        radius: int = 8,
        eps: float = 0.01
    ) -> np.ndarray:
        """
        Guided Filterでエッジを精緻化

        Args:
            mask: 入力マスク (0-255)
            guide_image: ガイド画像（元フレーム、Noneの場合は自己ガイド）
            radius: フィルタ半径
            eps: 正則化パラメータ
        """
        if not self.has_ximgproc:
            # フォールバック: Bilateral Filter
            return cv2.bilateralFilter(mask, 9, 75, 75)

        mask_float = mask.astype(np.float32) / 255.0

        if guide_image is not None:
            # カラーガイド画像を使用
            if len(guide_image.shape) == 3:
                guide = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
            else:
                guide = guide_image
            guide_float = guide.astype(np.float32) / 255.0
        else:
            # 自己ガイド
            guide_float = mask_float

        # Guided Filter適用
        filtered = cv2.ximgproc.guidedFilter(
            guide_float, mask_float, radius, eps
        )

        return np.clip(filtered * 255, 0, 255).astype(np.uint8)

    def protect_connected_components(
        self,
        mask: np.ndarray,
        min_area_ratio: float = 0.005,  # 増加: より厳しいノイズ除去
        keep_only_largest: bool = False  # 最大成分のみを保持
    ) -> np.ndarray:
        """
        接続成分解析で小さなノイズを除去しつつ、
        主要な物体に接続された部分を保護

        Args:
            mask: 入力マスク
            min_area_ratio: 最小面積比率（これ以下は除去）
            keep_only_largest: Trueの場合、最大の連結成分のみを保持
        """
        h, w = mask.shape
        min_area = int(h * w * min_area_ratio)

        # 二値化
        binary = (mask > 127).astype(np.uint8)

        # 接続成分解析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 最大の成分を見つける（背景=0を除く）
        if num_labels <= 1:
            return mask

        # 各成分の面積を取得（背景を除く）
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return mask

        max_area = np.max(areas)
        max_label = np.argmax(areas) + 1  # +1 because background is 0

        result = np.zeros_like(mask)

        if keep_only_largest:
            # 最大の成分のみを保持（ギター演奏者のみを抽出）
            result[labels == max_label] = 255
        else:
            # 最大成分の一定割合以上の成分を保持
            # ただし、閾値を厳しくして小さな残骸を除去
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # 最大成分の20%以上、または最小面積以上
                if area >= max_area * 0.20 or (area >= min_area and area >= max_area * 0.05):
                    result[labels == i] = 255

        return result

    def temporal_smooth(self, mask: np.ndarray, weight: float = 0.7) -> np.ndarray:
        """
        時間的平滑化（移動平均）

        Args:
            mask: 現在のマスク
            weight: 現在フレームの重み（0-1、高いほど現在フレーム重視）
        """
        mask_float = mask.astype(np.float32)

        if len(self.mask_buffer) == 0:
            self.mask_buffer.append(mask_float)
            return mask

        # サイズチェック
        if self.mask_buffer[0].shape != mask_float.shape:
            self.mask_buffer.clear()
            self.mask_buffer.append(mask_float)
            return mask

        self.mask_buffer.append(mask_float)

        # 加重平均
        weights = np.array([weight ** (len(self.mask_buffer) - 1 - i)
                          for i in range(len(self.mask_buffer))])
        weights = weights / weights.sum()

        smoothed = np.zeros_like(mask_float)
        for i, buf_mask in enumerate(self.mask_buffer):
            smoothed += buf_mask * weights[i]

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def apply_edge_margin(
        self,
        mask: np.ndarray,
        margin_ratio: float = 0.15  # 端から15%をマスクから除外
    ) -> np.ndarray:
        """
        画像の端（特に右側）をマスクから除外

        ギター演奏動画では、右側に椅子/機材が映り込むことが多いため、
        画像の右端をマスクから除外する
        """
        h, w = mask.shape
        result = mask.copy()

        # 右端をフェードアウト（椅子/機材を除去）
        right_margin = int(w * margin_ratio)
        for i in range(right_margin):
            alpha = i / right_margin  # 0 -> 1 のグラデーション
            x = w - right_margin + i
            result[:, x] = (result[:, x].astype(np.float32) * alpha).astype(np.uint8)

        return result

    def process(self, mask: np.ndarray, guide_image: Optional[np.ndarray] = None) -> np.ndarray:
        """全ての後処理を適用"""
        # 1. Morphological操作（穴埋め、細部保護）
        mask = self.apply_morphological_ops(mask)

        # 2. 接続成分解析（ノイズ除去）
        mask = self.protect_connected_components(mask)

        # 3. エッジマージン（右端を除外）
        mask = self.apply_edge_margin(mask, margin_ratio=0.12)

        # 4. Guided Filter（エッジ精緻化）
        mask = self.apply_guided_filter(mask, guide_image)

        # 5. 時間的平滑化（フリッカー除去）
        mask = self.temporal_smooth(mask)

        return mask

    def reset(self):
        """状態リセット"""
        self.mask_buffer.clear()


class RobustVideoSegmentation:
    """
    堅牢なビデオセグメンテーションサービス

    SAM 2 VideoPredictor を使用した時間的一貫性のあるセグメンテーション

    改善点:
    - CPU上では軽量モデル（small）をデフォルトで使用
    - Guided Filterでエッジを滑らかに
    - 強化されたMorphological操作でギター指板の穴を埋める
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_size: str = "auto"  # "auto" = CPU: small, GPU: large
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # CPU上では軽量モデルを使用（速度優先）
        if model_size == "auto":
            if self.device == "cpu":
                self.model_size = "small"  # CPU: 軽量モデル
                print("  Note: Using 'small' model for CPU (faster)")
            else:
                self.model_size = "large"  # GPU: 高品質モデル
        else:
            self.model_size = model_size

        # SAM 2 関連
        self.sam2_predictor = None
        self.video_predictor = None

        # 後処理（強化版）
        self.post_processor = MaskPostProcessor(buffer_size=5)

        # FFmpeg
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe"

        print(f"RobustVideoSegmentation initialized")
        print(f"  Device: {self.device}")
        print(f"  Model: sam2_hiera_{self.model_size}")

    def _load_sam2_image_predictor(self):
        """SAM 2 Image Predictor を読み込み"""
        if self.sam2_predictor is not None:
            return

        print("Loading SAM 2 Image Predictor...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download

            model_configs = {
                "tiny": ("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
                "small": ("facebook/sam2-hiera-small", "sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
                "base_plus": ("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
                "large": ("facebook/sam2-hiera-large", "sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            }

            repo_id, ckpt_name, config_name = model_configs.get(
                self.model_size, model_configs["large"]
            )

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
            print("SAM 2 Image Predictor loaded")

        except Exception as e:
            print(f"Failed to load SAM 2 Image Predictor: {e}")
            raise

    def _load_video_predictor(self):
        """SAM 2 Video Predictor を読み込み"""
        if self.video_predictor is not None:
            return

        print("Loading SAM 2 Video Predictor...")

        try:
            from sam2.build_sam import build_sam2_video_predictor
            from huggingface_hub import hf_hub_download

            model_configs = {
                "tiny": ("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
                "small": ("facebook/sam2-hiera-small", "sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
                "base_plus": ("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
                "large": ("facebook/sam2-hiera-large", "sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            }

            repo_id, ckpt_name, config_name = model_configs.get(
                self.model_size, model_configs["large"]
            )

            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

            import sam2
            sam2_path = Path(sam2.__file__).parent
            config_path = sam2_path / "configs" / "sam2" / config_name

            if not config_path.exists():
                config_path = sam2_path / "sam2_configs" / config_name

            self.video_predictor = build_sam2_video_predictor(
                str(config_path),
                checkpoint_path,
                device=self.device
            )

            print("SAM 2 Video Predictor loaded")

        except Exception as e:
            print(f"Failed to load SAM 2 Video Predictor: {e}")
            raise

    def segment_single_frame(
        self,
        frame: np.ndarray,
        foreground_points: List[Tuple[int, int]],
        background_points: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        単一フレームをセグメント（プレビュー用）
        """
        self._load_sam2_image_predictor()

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ポイントを準備
        all_points = []
        all_labels = []

        for point in foreground_points:
            all_points.append(point)
            all_labels.append(1)

        for point in (background_points or []):
            all_points.append(point)
            all_labels.append(0)

        if not all_points:
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
        mask = (masks[best_idx] * 255).astype(np.uint8)

        # 後処理
        mask = self.post_processor.apply_morphological_ops(mask)
        mask = self.post_processor.protect_connected_components(mask)

        return mask

    def process_video_with_propagation(
        self,
        input_path: str,
        output_path: str,
        keyframe_annotations: List[KeyframeAnnotation],
        background_path: Optional[str] = None,
        background_bytes: Optional[bytes] = None,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None
    ) -> None:
        """
        動画全体を処理（SAM 2 VideoPredictor + 時間的伝播）

        Args:
            input_path: 入力動画パス
            output_path: 出力動画パス
            keyframe_annotations: キーフレームアノテーションのリスト
            background_path: 背景ファイルパス
            background_bytes: 背景バイトデータ
            progress_callback: 進捗コールバック (current, total, status, eta)
        """
        print(f"=== Robust Video Segmentation ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Keyframes: {len(keyframe_annotations)}")

        self._load_video_predictor()
        self.post_processor.reset()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

            # 一時ディレクトリ
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                os.makedirs(frames_dir)

                # 音声抽出
                temp_audio = os.path.join(temp_dir, "audio.aac")
                has_audio = self._extract_audio(input_path, temp_audio)

                # フレームを画像として保存
                if progress_callback:
                    progress_callback(0, total_frames, "フレーム抽出中...", 0)

                cap = cv2.VideoCapture(input_path)
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frame_idx += 1

                    if progress_callback and frame_idx % 30 == 0:
                        progress_callback(frame_idx, total_frames, f"フレーム抽出中: {frame_idx}/{total_frames}", 0)

                cap.release()

                # SAM 2 VideoPredictor で処理
                if progress_callback:
                    progress_callback(0, total_frames, "SAM 2 初期化中...", 0)

                start_time = time.time()

                with torch.inference_mode():
                    context = (torch.autocast(device_type='cuda', dtype=torch.float16)
                              if self.device == "cuda" else torch.no_grad())

                    with context:
                        # 動画状態を初期化
                        inference_state = self.video_predictor.init_state(video_path=frames_dir)

                        # キーフレームアノテーションを追加
                        for annotation in keyframe_annotations:
                            all_points = []
                            all_labels = []

                            for point in annotation.foreground_points:
                                all_points.append([point[0], point[1]])
                                all_labels.append(1)

                            for point in annotation.background_points:
                                all_points.append([point[0], point[1]])
                                all_labels.append(0)

                            if all_points:
                                points_array = np.array(all_points, dtype=np.float32)
                                labels_array = np.array(all_labels, dtype=np.int32)

                                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=annotation.frame_idx,
                                    obj_id=1,
                                    points=points_array,
                                    labels=labels_array,
                                )

                                print(f"Added annotation at frame {annotation.frame_idx}: "
                                      f"{len(annotation.foreground_points)} FG, "
                                      f"{len(annotation.background_points)} BG points")

                        # マスク伝播
                        if progress_callback:
                            progress_callback(0, total_frames, "マスク伝播中...", 0)

                        video_segments = {}
                        propagate_start = time.time()

                        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                            video_segments[out_frame_idx] = mask

                            if progress_callback and (out_frame_idx + 1) % 30 == 0:
                                elapsed = time.time() - propagate_start
                                fps_actual = (out_frame_idx + 1) / elapsed if elapsed > 0 else 0
                                remaining = total_frames - out_frame_idx - 1
                                eta = remaining / fps_actual if fps_actual > 0 else 0

                                progress_callback(
                                    out_frame_idx + 1, total_frames,
                                    f"マスク伝播中: {out_frame_idx + 1}/{total_frames} ({fps_actual:.1f}fps)",
                                    eta
                                )

                        self.video_predictor.reset_state(inference_state)

                # 背景を読み込み
                bg_frame = self._load_background(width, height, background_path, background_bytes)

                # 背景動画の場合
                bg_cap = None
                if background_path and Path(background_path).suffix.lower() in {".mp4", ".webm", ".gif"}:
                    bg_cap = cv2.VideoCapture(background_path)

                # 合成と出力
                if progress_callback:
                    progress_callback(0, total_frames, "合成中...", 0)

                temp_video = os.path.join(temp_dir, "temp_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

                cap = cv2.VideoCapture(input_path)
                composite_start = time.time()

                for frame_idx in range(total_frames):
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

                    # マスクを取得
                    if frame_idx in video_segments:
                        mask = video_segments[frame_idx]

                        # 元のサイズに戻す
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask.astype(np.float32), (width, height)) > 0.5

                        mask = (mask * 255).astype(np.uint8)

                        # 後処理（Guided Filterにフレームを渡す）
                        mask = self.post_processor.process(mask, guide_image=frame)
                    else:
                        mask = np.zeros((height, width), dtype=np.uint8)

                    # 合成（滑らかなアルファブレンド）
                    alpha = mask.astype(np.float32) / 255.0
                    # エッジを少しぼかして滑らかに
                    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                    alpha = alpha[:, :, np.newaxis]
                    composite = (frame.astype(np.float32) * alpha +
                                bg_frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
                    out.write(composite)

                    if progress_callback and (frame_idx + 1) % 30 == 0:
                        elapsed = time.time() - composite_start
                        fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                        remaining = total_frames - frame_idx - 1
                        eta = remaining / fps_actual if fps_actual > 0 else 0

                        progress_callback(
                            frame_idx + 1, total_frames,
                            f"合成中: {frame_idx + 1}/{total_frames} ({fps_actual:.1f}fps)",
                            eta
                        )

                cap.release()
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

        except Exception as e:
            print(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_audio(self, video_path: str, output_path: str) -> bool:
        """音声を抽出"""
        try:
            # まず音声があるか確認
            cmd_check = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path
            ]
            result = subprocess.run(cmd_check, capture_output=True, text=True)
            if "audio" not in result.stdout.lower():
                return False

            # 音声抽出
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

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """互換性用インターフェース（単一フレーム処理）"""
        mask = self.segment_single_frame(frame, [(frame.shape[1]//2, frame.shape[0]//2)])
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        return bgra

    def reset_temporal_state(self):
        """時間的状態をリセット"""
        self.post_processor.reset()
