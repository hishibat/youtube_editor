"""MediaPipe Selfie Segmentation - 人物専用の高品質セグメンテーション"""

import cv2
import numpy as np
import mediapipe as mp


class MediaPipeSegmentation:
    """MediaPipeを使用した人物セグメンテーションサービス"""

    def __init__(self, model_selection: int = 1):
        """
        初期化

        Args:
            model_selection: モデル選択
                - 0: 一般モデル（256x256入力、高速）
                - 1: 風景モデル（256x144入力、より正確、推奨）
        """
        self.model_selection = model_selection
        self.selfie_segmentation = None

    def _get_segmenter(self):
        """セグメンターを遅延初期化"""
        if self.selfie_segmentation is None:
            self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=self.model_selection
            )
        return self.selfie_segmentation

    def remove_background(
        self, frame: np.ndarray, threshold: float = 0.2
    ) -> np.ndarray:
        """
        フレームから背景を除去（品質重視版）

        Args:
            frame: BGR形式のOpenCV画像 (numpy array)
            threshold: セグメンテーション閾値 (0.0-1.0)
                      低いほど人物領域を広く取る（デフォルト0.2で広めに）

        Returns:
            BGRA形式の画像（アルファチャンネル付き）
        """
        segmenter = self._get_segmenter()

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # セグメンテーション実行
        results = segmenter.process(frame_rgb)

        # マスクを取得 (0.0-1.0の確信度)
        mask = results.segmentation_mask

        # ソフトマスク（グラデーション）を使用して滑らかに
        # 閾値より低い部分を0、高い部分を段階的に255に
        soft_mask = np.clip((mask - threshold) / (1.0 - threshold), 0, 1)
        soft_mask = (soft_mask * 255).astype(np.uint8)

        # マスクを膨張させて人物領域を確実にカバー
        kernel = np.ones((9, 9), np.uint8)
        soft_mask = cv2.dilate(soft_mask, kernel, iterations=2)

        # エッジを滑らかにするためにガウシアンブラー
        soft_mask = cv2.GaussianBlur(soft_mask, (21, 21), 0)

        # BGRA画像を作成
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = soft_mask

        return bgra

    def remove_background_smooth(
        self, frame: np.ndarray, threshold: float = 0.3
    ) -> np.ndarray:
        """
        より滑らかなエッジで背景を除去（品質重視）

        Args:
            frame: BGR形式のOpenCV画像
            threshold: セグメンテーション閾値（低めで人物を広く取る）

        Returns:
            BGRA形式の画像
        """
        segmenter = self._get_segmenter()

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # セグメンテーション実行
        results = segmenter.process(frame_rgb)
        mask = results.segmentation_mask

        # ソフトマスク（グラデーション）を使用
        # 閾値より低い部分を0、高い部分を段階的に255に
        soft_mask = np.clip((mask - threshold) / (1.0 - threshold), 0, 1)
        soft_mask = (soft_mask * 255).astype(np.uint8)

        # マスクを膨張させて人物領域を確保
        kernel = np.ones((7, 7), np.uint8)
        soft_mask = cv2.dilate(soft_mask, kernel, iterations=2)

        # エッジを滑らかに
        soft_mask = cv2.GaussianBlur(soft_mask, (15, 15), 0)

        # BGRA画像を作成
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = soft_mask

        return bgra

    def __del__(self):
        """リソース解放"""
        if self.selfie_segmentation is not None:
            self.selfie_segmentation.close()
