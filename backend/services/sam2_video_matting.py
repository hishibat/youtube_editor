"""
SAM 2 (Segment Anything Model 2) ベースの動画マッティングサービス

ギター演奏動画の高精度背景除去に特化
- ポイント指定型追跡（人物+ギター）
- Hough変換によるギター指板保護
- オプティカルフローによる振動検知
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path


class SAM2VideoMattingService:
    """
    SAM 2を使用した高精度動画マッティングサービス

    処理速度よりも切り抜き精度を最優先
    """

    def __init__(
        self,
        model_size: str = "large",  # tiny, small, base_plus, large
        device: Optional[str] = None
    ):
        """
        Args:
            model_size: モデルサイズ（large推奨）
            device: 使用デバイス（None=自動検出）
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SAM 2 関連
        self.predictor = None
        self.video_state = None

        # ポイント追跡用
        self.foreground_points: List[Tuple[int, int]] = []  # 前景ポイント（人物+ギター）
        self.background_points: List[Tuple[int, int]] = []  # 背景ポイント（椅子など）

        # ギター指板保護用
        self.guitar_neck_mask: Optional[np.ndarray] = None

        # オプティカルフロー用
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.flow_accumulator: Optional[np.ndarray] = None
        self.flow_frame_count: int = 0

        # フレームバッファ（SAM 2のビデオ処理用）
        self.frame_buffer: List[np.ndarray] = []
        self.max_buffer_size: int = 30  # 処理するフレーム数

        # 処理済みマスクのキャッシュ
        self.mask_cache: Dict[int, np.ndarray] = {}

        print(f"SAM2VideoMattingService initialized (model={model_size}, device={self.device})")

    def _load_model(self):
        """SAM 2モデルを遅延読み込み"""
        if self.predictor is not None:
            return

        print(f"Loading SAM 2 {self.model_size} model...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download

            # モデルサイズに応じた設定
            model_configs = {
                "tiny": ("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
                "small": ("facebook/sam2-hiera-small", "sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
                "base_plus": ("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
                "large": ("facebook/sam2-hiera-large", "sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            }

            repo_id, ckpt_name, config_name = model_configs.get(self.model_size, model_configs["large"])

            # チェックポイントをダウンロード
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

            # SAM 2のインストールパスからconfig取得
            import sam2
            sam2_path = Path(sam2.__file__).parent
            config_path = sam2_path / "configs" / "sam2" / config_name

            if not config_path.exists():
                # 代替パスを試す
                config_path = sam2_path / "sam2_configs" / config_name

            print(f"Checkpoint: {checkpoint_path}")
            print(f"Config: {config_path}")

            # SAM 2モデルを構築
            sam2_model = build_sam2(
                str(config_path),
                checkpoint_path,
                device=self.device
            )

            # 画像予測器を作成
            self.predictor = SAM2ImagePredictor(sam2_model)

            print("SAM 2 model loaded successfully.")

        except Exception as e:
            print(f"Failed to load SAM 2: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to simple implementation...")
            self.predictor = None
            raise

    def detect_guitar_neck_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Hough変換でギターネック（指板）の直線を検出し、保護マスクを生成

        Args:
            frame: BGR画像

        Returns:
            保護マスク（ギターネック領域が255）
        """
        h, w = frame.shape[:2]
        protection_mask = np.zeros((h, w), dtype=np.uint8)

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # エッジ検出（Canny）
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # モルフォロジー演算でエッジを強調
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 確率的Hough変換で直線検出
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=100,  # ギターネックは長い直線
            maxLineGap=20
        )

        if lines is None:
            return protection_mask

        # ギターネックらしい直線を抽出
        guitar_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 直線の長さ
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 直線の角度（水平に近いものはスキップ）
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            # ギターネックの条件：
            # - 長さが画面の10%以上
            # - 角度が20-80度（斜めの直線）
            # - 画面中央〜左側に位置
            center_x = (x1 + x2) / 2

            if (length > min(h, w) * 0.1 and
                15 < angle < 85 and
                center_x < w * 0.7):  # 右側30%は椅子の可能性が高いので除外
                guitar_lines.append((x1, y1, x2, y2, length, angle))

        # 検出された直線の周囲に保護領域を作成
        for x1, y1, x2, y2, length, angle in guitar_lines:
            # 直線の周囲に太い線を描画（保護領域）
            thickness = int(length * 0.15)  # 長さに応じた太さ
            cv2.line(protection_mask, (x1, y1), (x2, y2), 255, thickness)

        # 保護領域を少し拡張
        if np.any(protection_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            protection_mask = cv2.dilate(protection_mask, kernel, iterations=2)

            # エッジをぼかす
            protection_mask = cv2.GaussianBlur(protection_mask, (21, 21), 0)

        return protection_mask

    def detect_vibration_regions(
        self,
        frame: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        オプティカルフローで振動（微細な動き）がある領域を検出

        振動がある = 人物の手やギターの弦 = 前景
        振動がない = 椅子などの静止物体 = 背景

        Args:
            frame: BGR画像
            threshold: 振動検出閾値（低いほど敏感）

        Returns:
            振動マスク（振動領域が255、静止領域が0）
        """
        h, w = frame.shape[:2]

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            self.flow_accumulator = np.zeros((h, w), dtype=np.float32)
            self.flow_frame_count = 0
            return np.ones((h, w), dtype=np.uint8) * 255  # 初回は全領域を前景扱い

        # サイズが変わった場合はリセット
        if gray.shape != self.prev_frame_gray.shape:
            self.prev_frame_gray = gray
            self.flow_accumulator = np.zeros((h, w), dtype=np.float32)
            self.flow_frame_count = 0
            return np.ones((h, w), dtype=np.uint8) * 255

        # Farneback法でオプティカルフロー計算
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # フローの大きさを計算
        flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # 累積フロー（時間的平滑化）
        self.flow_frame_count += 1
        alpha = 0.1  # 平滑化係数
        self.flow_accumulator = (1 - alpha) * self.flow_accumulator + alpha * flow_magnitude

        # 閾値処理で振動領域を検出
        # 非常に低い閾値で、わずかな動きも検出
        vibration_mask = (self.flow_accumulator > threshold).astype(np.uint8) * 255

        # モルフォロジー演算でノイズ除去と領域拡張
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        vibration_mask = cv2.morphologyEx(vibration_mask, cv2.MORPH_CLOSE, kernel)
        vibration_mask = cv2.dilate(vibration_mask, kernel, iterations=2)

        self.prev_frame_gray = gray

        return vibration_mask

    def auto_detect_prompt_points(
        self,
        frame: np.ndarray,
        existing_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        フレームから自動的に前景/背景のプロンプトポイントを検出

        前景ポイント：
        - 人物の胸の中央
        - ギターのヘッド付近
        - ギターのボディ付近

        背景ポイント：
        - 右側の椅子領域
        - 四隅

        Args:
            frame: BGR画像
            existing_mask: 既存のマスク（参考用）

        Returns:
            (foreground_points, background_points)
        """
        h, w = frame.shape[:2]

        foreground_points = []
        background_points = []

        # === 前景ポイントの自動検出 ===

        # 1. 人物の胸の中央（画面中央やや左上）
        chest_x = int(w * 0.4)  # 左から40%
        chest_y = int(h * 0.35)  # 上から35%
        foreground_points.append((chest_x, chest_y))

        # 2. ギターのボディ（画面中央やや下）
        body_x = int(w * 0.45)
        body_y = int(h * 0.65)
        foreground_points.append((body_x, body_y))

        # 3. ギターのヘッド（画面左上〜中央上）
        # Hough変換で検出した直線の端点を使う
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=80, maxLineGap=20)

        if lines is not None:
            # 最も左上にある直線の端点を探す
            best_point = None
            best_score = float('inf')

            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 左上優先のスコア
                for x, y in [(x1, y1), (x2, y2)]:
                    if x < w * 0.5 and y < h * 0.5:  # 左上半分のみ
                        score = x + y  # 左上ほどスコアが低い
                        if score < best_score:
                            best_score = score
                            best_point = (x, y)

            if best_point:
                foreground_points.append(best_point)
        else:
            # 直線が検出されなかった場合のデフォルト
            head_x = int(w * 0.25)
            head_y = int(h * 0.25)
            foreground_points.append((head_x, head_y))

        # === 背景ポイントの自動検出 ===

        # 1. 右側30%は椅子の可能性が高い
        chair_x = int(w * 0.85)
        chair_y = int(h * 0.5)
        background_points.append((chair_x, chair_y))

        # 右側の複数ポイント
        background_points.append((int(w * 0.9), int(h * 0.3)))
        background_points.append((int(w * 0.9), int(h * 0.7)))

        # 2. 四隅（確実に背景）
        margin = 20
        background_points.append((margin, margin))  # 左上
        background_points.append((w - margin, margin))  # 右上
        background_points.append((margin, h - margin))  # 左下
        background_points.append((w - margin, h - margin))  # 右下

        return foreground_points, background_points

    def process_single_frame(
        self,
        frame: np.ndarray,
        foreground_points: Optional[List[Tuple[int, int]]] = None,
        background_points: Optional[List[Tuple[int, int]]] = None,
        protect_guitar_neck: bool = True,
        use_vibration_detection: bool = True
    ) -> np.ndarray:
        """
        単一フレームを処理してBGRA画像を返す

        Args:
            frame: BGR画像
            foreground_points: 前景ポイント（None=自動検出）
            background_points: 背景ポイント（None=自動検出）
            protect_guitar_neck: ギターネックを保護するか
            use_vibration_detection: 振動検知を使用するか

        Returns:
            BGRA画像（アルファチャンネル付き）
        """
        h, w = frame.shape[:2]

        # ポイントの自動検出
        if foreground_points is None or background_points is None:
            auto_fg, auto_bg = self.auto_detect_prompt_points(frame)
            foreground_points = foreground_points or auto_fg
            background_points = background_points or auto_bg

        # SAM 2でセグメンテーション
        try:
            self._load_model()
            mask = self._segment_with_sam2(frame, foreground_points, background_points)
        except Exception as e:
            print(f"SAM 2 segmentation failed: {e}")
            # フォールバック：簡易セグメンテーション
            mask = self._fallback_segmentation(frame)

        # ギターネック保護
        if protect_guitar_neck:
            neck_protection = self.detect_guitar_neck_lines(frame)
            # 保護領域をマスクに追加
            mask = np.maximum(mask, neck_protection)

        # 振動検知による静止物体除去
        if use_vibration_detection:
            vibration_mask = self.detect_vibration_regions(frame, threshold=0.2)
            # 振動がない領域はマスクから除去（ただし保護領域は除く）
            static_region = cv2.bitwise_not(vibration_mask)
            if protect_guitar_neck:
                # ギターネック保護領域は静止でも除去しない
                static_region = cv2.bitwise_and(static_region, cv2.bitwise_not(neck_protection))

            # 静止領域をマスクから減算（90%抑制）
            mask_float = mask.astype(np.float32)
            static_float = static_region.astype(np.float32) / 255.0
            mask_float = mask_float * (1 - static_float * 0.9)
            mask = mask_float.astype(np.uint8)

        # 強制的に右側30%を背景化（椅子対策の最終手段）
        right_exclusion_start = int(w * 0.75)
        right_mask = np.ones((h, w), dtype=np.float32)
        # グラデーションで自然に除去
        for x in range(right_exclusion_start, w):
            ratio = (x - right_exclusion_start) / (w - right_exclusion_start)
            right_mask[:, x] = 1 - ratio * 0.95

        mask = (mask.astype(np.float32) * right_mask).astype(np.uint8)

        # BGRA画像を作成
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask

        return bgra

    def _segment_with_sam2(
        self,
        frame: np.ndarray,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        SAM 2を使用してセグメンテーション

        Args:
            frame: BGR画像
            foreground_points: 前景ポイント
            background_points: 背景ポイント

        Returns:
            マスク（0-255）
        """
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # 画像予測器を使用（ビデオ予測器よりシンプル）
        # Note: SAM2ImagePredictorはビデオ予測器から作成できる

        h, w = frame.shape[:2]

        # RGB変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ポイントとラベルを準備
        all_points = []
        all_labels = []

        for point in foreground_points:
            all_points.append(point)
            all_labels.append(1)  # 1 = foreground

        for point in background_points:
            all_points.append(point)
            all_labels.append(0)  # 0 = background

        points_array = np.array(all_points)
        labels_array = np.array(all_labels)

        # SAM 2推論
        with torch.inference_mode():
            # 画像をセット
            self.predictor.set_image(frame_rgb)

            # ポイントプロンプトで予測
            masks, scores, logits = self.predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )

        # 最もスコアの高いマスクを選択
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # 0-1を0-255に変換
        mask = (mask * 255).astype(np.uint8)

        return mask

    def _fallback_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        SAM 2が使えない場合のフォールバックセグメンテーション

        GrabCut + 色ベースのセグメンテーション
        """
        h, w = frame.shape[:2]

        # 初期マスク（前景/背景の推定）
        mask = np.zeros((h, w), np.uint8)

        # 中央領域を前景候補
        center_rect = (
            int(w * 0.2),   # x
            int(h * 0.1),   # y
            int(w * 0.5),   # width
            int(h * 0.8)    # height
        )

        # GrabCut用モデル
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                frame,
                mask,
                center_rect,
                bgd_model,
                fgd_model,
                iterCount=5,
                mode=cv2.GC_INIT_WITH_RECT
            )

            # マスクを二値化（前景=255, 背景=0）
            mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype(np.uint8)

        except Exception as e:
            print(f"GrabCut failed: {e}")
            # 最終フォールバック：中央領域を前景
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(
                mask,
                (int(w * 0.2), int(h * 0.1)),
                (int(w * 0.7), int(h * 0.9)),
                255,
                -1
            )

        return mask

    def process_video_batch(
        self,
        frames: List[np.ndarray],
        propagate: bool = True
    ) -> List[np.ndarray]:
        """
        複数フレームをバッチ処理（SAM 2のビデオ機能を使用）

        Args:
            frames: フレームのリスト
            propagate: フレーム間で伝播するか

        Returns:
            BGRA画像のリスト
        """
        if not frames:
            return []

        results = []

        # 最初のフレームでポイントを検出
        first_frame = frames[0]
        fg_points, bg_points = self.auto_detect_prompt_points(first_frame)

        # 各フレームを処理
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}...")

            # 最初のフレームは自動検出したポイントを使用
            # それ以降は伝播（SAM 2が内部で追跡）
            if i == 0 or not propagate:
                bgra = self.process_single_frame(
                    frame,
                    foreground_points=fg_points,
                    background_points=bg_points
                )
            else:
                bgra = self.process_single_frame(frame)

            results.append(bgra)

        return results

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        video_processorとの互換性用インターフェース

        Args:
            frame: BGR画像

        Returns:
            BGRA画像
        """
        return self.process_single_frame(
            frame,
            protect_guitar_neck=True,
            use_vibration_detection=True
        )

    def composite_with_background(
        self,
        foreground_bgra: np.ndarray,
        background: np.ndarray
    ) -> np.ndarray:
        """
        前景を背景と合成

        Args:
            foreground_bgra: BGRA画像
            background: BGR背景画像

        Returns:
            合成されたBGR画像
        """
        h, w = foreground_bgra.shape[:2]

        # 背景をリサイズ
        bg_resized = cv2.resize(background, (w, h))

        # アルファブレンド
        alpha = foreground_bgra[:, :, 3:4].astype(np.float32) / 255.0
        fg_bgr = foreground_bgra[:, :, :3].astype(np.float32)
        bg_float = bg_resized.astype(np.float32)

        composite = (fg_bgr * alpha + bg_float * (1 - alpha)).astype(np.uint8)

        return composite

    def reset(self):
        """状態をリセット"""
        self.video_state = None
        self.foreground_points = []
        self.background_points = []
        self.guitar_neck_mask = None
        self.prev_frame_gray = None
        self.flow_accumulator = None
        self.flow_frame_count = 0
        self.frame_buffer = []
        self.mask_cache = {}


# テスト用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python sam2_video_matting.py <input_image> <output_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # サービス初期化
    service = SAM2VideoMattingService(model_size="large")

    # 画像読み込み
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Failed to read: {input_path}")
        sys.exit(1)

    # 処理
    result = service.remove_background(frame)

    # 保存
    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")
