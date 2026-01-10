"""背景除去サービス - rembgを使用したAI背景除去（品質・速度バランス版）"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


class BackgroundRemovalService:
    """rembgを使用して画像/フレームから背景を除去するサービス"""

    def __init__(self, model_name: str = "u2net", fast_mode: bool = False, process_height: int = 720):
        """
        初期化

        Args:
            model_name: 使用するモデル名
                - "u2net": 汎用高品質（推奨）人物全体を認識
                - "u2netp": 高速版（品質は落ちる）
                - "u2net_human_seg": 人物特化
                - "isnet-general-use": 高品質汎用
                - "silueta": 軽量（低品質）
            fast_mode: 高速モードを使用するか（alpha_mattingを無効化）
            process_height: 処理解像度（高さ）
        """
        self.model_name = model_name
        self.fast_mode = fast_mode
        self.session = None
        # 処理用の縮小解像度（高さ）
        self.process_height = process_height

    def _get_session(self):
        """セッションを遅延初期化"""
        if self.session is None:
            self.session = new_session(self.model_name)
        return self.session

    def remove_background(
        self, frame: np.ndarray, target_size: tuple[int, int] = None
    ) -> np.ndarray:
        """
        フレームから背景を除去（高速版）

        Args:
            frame: BGR形式のOpenCV画像 (numpy array)
            target_size: 出力サイズ (height, width)。Noneの場合は入力と同じ

        Returns:
            BGRA形式の画像（アルファチャンネル付き）
        """
        original_h, original_w = frame.shape[:2]
        target_size = target_size or (original_h, original_w)

        # 処理用に縮小（高速化のため）
        scale = self.process_height / original_h
        if scale < 1.0:
            process_w = int(original_w * scale)
            process_h = self.process_height
            small_frame = cv2.resize(frame, (process_w, process_h))
        else:
            small_frame = frame
            process_h, process_w = original_h, original_w

        # OpenCV BGR -> PIL RGB
        frame_rgb = small_frame[:, :, ::-1]
        pil_image = Image.fromarray(frame_rgb)

        # 背景除去（高速モードではalpha_mattingを無効化）
        if self.fast_mode:
            result = remove(
                pil_image,
                session=self._get_session(),
                alpha_matting=False,
            )
        else:
            result = remove(
                pil_image,
                session=self._get_session(),
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )

        # PIL RGBA -> numpy BGRA
        result_array = np.array(result)
        # RGBA -> BGRA
        bgra = result_array[:, :, [2, 1, 0, 3]]

        # 元のサイズに戻す
        if bgra.shape[:2] != target_size:
            bgra = cv2.resize(bgra, (target_size[1], target_size[0]))

        return bgra

    def remove_background_fast(self, frame: np.ndarray) -> np.ndarray:
        """
        最速モードで背景除去（品質は落ちるが非常に高速）

        Args:
            frame: BGR形式のOpenCV画像

        Returns:
            BGRA形式の画像
        """
        original_h, original_w = frame.shape[:2]

        # さらに小さいサイズで処理
        fast_height = 320
        scale = fast_height / original_h
        fast_w = int(original_w * scale)

        small_frame = cv2.resize(frame, (fast_w, fast_height))

        # BGR -> RGB
        frame_rgb = small_frame[:, :, ::-1]
        pil_image = Image.fromarray(frame_rgb)

        # 最速設定
        result = remove(
            pil_image,
            session=self._get_session(),
            alpha_matting=False,
        )

        # PIL RGBA -> numpy BGRA
        result_array = np.array(result)
        bgra = result_array[:, :, [2, 1, 0, 3]]

        # 元のサイズに戻す
        bgra = cv2.resize(bgra, (original_w, original_h))

        return bgra
