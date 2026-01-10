"""
Robust Video Matting (RVM) ベースの高品質ビデオマッティングサービス
ギター演奏動画専用の強力な後処理を含む

特徴:
- ビデオ専用に設計された時間的一貫性
- ガイデッドフィルタによるエッジ精緻化
- 静止物体（椅子・カーテン）の抑制
- GrabCutによる境界再計算
- 滑らかなアルファグラデーション
- 放送レベルの合成品質
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
import urllib.request
import os


class VideoMattingService:
    """Robust Video Matting を使用した高品質ビデオマッティング（エッジリファインメント版）"""

    MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript"
    MODEL_FILENAME = "rvm_mobilenetv3_fp32.torchscript"

    def __init__(self, model_dir: str = None, downsample_ratio: float = 0.4):
        """
        初期化

        Args:
            model_dir: モデル保存ディレクトリ
            downsample_ratio: 処理時のダウンサンプル比率（0.4=高品質）
        """
        self.model_dir = Path(model_dir) if model_dir else Path.home() / ".rvm"
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / self.MODEL_FILENAME
        self.downsample_ratio = downsample_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # RVMの再帰状態（時間的一貫性のため）
        self.rec = [None] * 4

        # 時間的平滑化用のバッファ
        self.alpha_buffer = []
        self.buffer_size = 5

        # 最後に処理したフレームサイズ（解像度変更検出用）
        self.last_frame_size = None

        # 静止物体検出用のフレームバッファ（30-60フレーム = 約1-2秒）
        self.frame_buffer: List[np.ndarray] = []
        self.frame_buffer_size = 45  # 約1.5秒分
        self.static_mask = None  # 静止領域マスク

        # 背景参照フレーム（奏者がいない状態）
        self.background_reference = None

        # MediaPipeセグメンテーション（アンサンブル用）
        self.mediapipe_segmenter = None

        # 強制排除ゾーン設定（右側の椅子対策）
        # 画面の右側何%を強制的に背景とするか
        self.exclusion_zone_right = 0.30  # 右30%
        self.exclusion_zone_left = 0.10   # 左10%
        self.exclusion_zone_top = 0.10    # 上10%
        self.exclusion_zone_bottom = 0.10 # 下10%

        # 動き検出用の前フレーム
        self.prev_frame_gray = None

        # インペインティングで生成した擬似背景（動的更新）
        self.pseudo_background = None
        self.pseudo_bg_update_interval = 30  # 30フレームごとに更新
        self.frame_count = 0

    def _create_high_confidence_mask(self, alpha: np.ndarray) -> np.ndarray:
        """
        高信頼度前景マスクを作成

        AIが「絶対に人物」と確信している領域のみを抽出
        収縮処理で背景の巻き込みを防止

        Args:
            alpha: 元のアルファマスク

        Returns:
            高信頼度のみの二値マスク
        """
        # 非常に高い閾値で二値化（alpha > 250）
        _, high_conf = cv2.threshold(alpha, 250, 255, cv2.THRESH_BINARY)

        # 強めの収縮で人物内部のみに限定
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        high_conf = cv2.erode(high_conf, kernel, iterations=3)

        # 小さなノイズを除去
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        high_conf = cv2.morphologyEx(high_conf, cv2.MORPH_OPEN, kernel_open)

        return high_conf

    def _generate_pseudo_background(self, frame: np.ndarray, high_conf_mask: np.ndarray) -> np.ndarray:
        """
        インペインティングで擬似背景を生成

        人物領域を周囲の背景情報で塗りつぶし、
        「人物がいない背景画像」を動的に生成

        Args:
            frame: 元のカラーフレーム
            high_conf_mask: 高信頼度前景マスク（塗りつぶす領域）

        Returns:
            擬似背景画像
        """
        # マスクを膨張させて確実に人物全体をカバー
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        inpaint_mask = cv2.dilate(high_conf_mask, kernel, iterations=2)

        # インペインティング（TELEA法 - 高速・高品質）
        # 半径を大きくして広い領域を塗りつぶせるように
        pseudo_bg = cv2.inpaint(frame, inpaint_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

        return pseudo_bg

    def _difference_attack(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        擬似背景との差分攻撃

        生成した背景と現在のフレームを比較し、
        差分が小さい領域（＝背景と同じ＝椅子など）を強制的に透明化

        Args:
            frame: 現在のフレーム
            alpha: 現在のアルファマスク

        Returns:
            差分攻撃後のアルファマスク
        """
        self.frame_count += 1

        # 高信頼度前景マスクを作成
        high_conf_mask = self._create_high_confidence_mask(alpha)

        # 擬似背景を生成/更新
        # 定期的に更新（毎フレームは重いため）
        if self.pseudo_background is None or self.frame_count % self.pseudo_bg_update_interval == 0:
            self.pseudo_background = self._generate_pseudo_background(frame, high_conf_mask)

        if self.pseudo_background is None:
            return alpha

        # 現在のフレームと擬似背景の差分を計算
        diff = cv2.absdiff(frame, self.pseudo_background)

        # グレースケールに変換
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 差分が小さい領域 = 背景と同じ = 椅子などの静止物体
        # 閾値を低めに設定（厳し目の判定）
        _, bg_match = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY_INV)

        # ノイズ除去（細かいノイズを消す）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bg_match = cv2.morphologyEx(bg_match, cv2.MORPH_OPEN, kernel)

        # 高信頼度前景領域は保護（差分攻撃から除外）
        # 人物の体は絶対に消さない
        protection_mask = cv2.dilate(high_conf_mask, kernel, iterations=3)
        bg_match = cv2.bitwise_and(bg_match, cv2.bitwise_not(protection_mask))

        # 背景と一致する領域をアルファから強制除去（95%抑制）
        suppression = bg_match.astype(np.float32) / 255.0 * 0.95
        alpha_float = alpha.astype(np.float32)
        alpha_attacked = alpha_float * (1 - suppression)

        return alpha_attacked.astype(np.uint8)

    def _download_model(self):
        """モデルをダウンロード"""
        if not self.model_path.exists():
            print(f"Downloading RVM model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("Download complete.")

    def _load_model(self):
        """モデルを読み込み"""
        if self.model is None:
            self._download_model()
            print(f"Loading RVM model from {self.model_path}...")
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print("Model loaded.")

    def reset_temporal_state(self):
        """時間的状態をリセット（新しい動画の開始時に呼ぶ）"""
        self.rec = [None] * 4
        self.alpha_buffer = []
        self.frame_buffer = []
        self.static_mask = None
        self.background_reference = None
        self.prev_frame_gray = None
        self.pseudo_background = None
        self.frame_count = 0
        # last_frame_sizeはリセットしない（サイズ変更検出に必要）

    def set_exclusion_zones(self, right: float = 0.30, left: float = 0.10,
                            top: float = 0.10, bottom: float = 0.10):
        """
        強制排除ゾーンを設定

        Args:
            right: 右側の排除率（0.0-0.5）
            left: 左側の排除率（0.0-0.5）
            top: 上側の排除率（0.0-0.5）
            bottom: 下側の排除率（0.0-0.5）
        """
        self.exclusion_zone_right = max(0.0, min(0.5, right))
        self.exclusion_zone_left = max(0.0, min(0.5, left))
        self.exclusion_zone_top = max(0.0, min(0.5, top))
        self.exclusion_zone_bottom = max(0.0, min(0.5, bottom))
        print(f"Exclusion zones set: R={right:.0%}, L={left:.0%}, T={top:.0%}, B={bottom:.0%}")

    def _brutal_exclusion_zone(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        【暴力的】座標ベースの強制排除

        画面端の指定領域を問答無用で透明にする
        ただし、大きな動きがある場合のみ保護（手やギターの移動）

        Args:
            frame: 現在のフレーム
            alpha: 現在のアルファマスク

        Returns:
            強制排除後のアルファマスク
        """
        h, w = alpha.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 動き検出（オプティカルフロー簡易版）
        motion_mask = np.zeros((h, w), dtype=np.float32)

        if self.prev_frame_gray is not None:
            # フレーム間差分で動きを検出
            diff = cv2.absdiff(gray, self.prev_frame_gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32) / 255.0

            # 動き領域を膨張（手の動きを広めに保護）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        self.prev_frame_gray = gray.copy()

        # === 強制排除ゾーンの計算 ===
        exclusion_mask = np.zeros((h, w), dtype=np.float32)

        # 右側排除ゾーン
        right_start = int(w * (1 - self.exclusion_zone_right))
        exclusion_mask[:, right_start:] = 1.0

        # 左側排除ゾーン
        left_end = int(w * self.exclusion_zone_left)
        exclusion_mask[:, :left_end] = 1.0

        # 上側排除ゾーン
        top_end = int(h * self.exclusion_zone_top)
        exclusion_mask[:top_end, :] = 1.0

        # 下側排除ゾーン
        bottom_start = int(h * (1 - self.exclusion_zone_bottom))
        exclusion_mask[bottom_start:, :] = 1.0

        # グラデーション境界（急激な切れ目を避ける）
        # 境界から内側20ピクセルにグラデーションをかける
        gradient_width = 30

        # 右側グラデーション
        for i in range(gradient_width):
            col = right_start - gradient_width + i
            if 0 <= col < w:
                exclusion_mask[:, col] = max(exclusion_mask[:, col].max(), i / gradient_width)

        # 左側グラデーション
        for i in range(gradient_width):
            col = left_end + i
            if 0 <= col < w:
                exclusion_mask[:, col] = max(exclusion_mask[:, col].max(), 1 - i / gradient_width)

        # === 動きによる保護 ===
        # 動きがある領域は排除しない（手やギターの保護）
        protection = motion_mask * 0.9  # 90%保護
        exclusion_mask = exclusion_mask * (1 - protection)

        # === アルファ値の強制排除 ===
        alpha_float = alpha.astype(np.float32)
        alpha_excluded = alpha_float * (1 - exclusion_mask)

        return alpha_excluded.astype(np.uint8)

    def _detect_mesh_texture(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        メッシュ状テクスチャの検出と除去

        椅子のメッシュのような規則的なパターンを検出し、
        そのような領域を背景として除去

        Args:
            frame: 現在のフレーム
            alpha: 現在のアルファマスク

        Returns:
            メッシュ領域を除去したアルファマスク
        """
        h, w = alpha.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 前景領域のみを分析
        fg_mask = alpha > 127

        # テクスチャ分析（局所的な分散）
        # ウィンドウサイズ15x15でローカルな標準偏差を計算
        window_size = 15

        # ローカル平均
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)

        # ローカル分散
        local_sq_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
        local_var = local_sq_mean - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))

        # 高周波成分の検出（エッジ密度）
        # Laplacianでエッジを検出
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_density = cv2.filter2D(np.abs(laplacian), -1, kernel)

        # メッシュ判定: 高いエッジ密度 + 中程度のテクスチャ分散
        # 人間の肌や服は滑らか、メッシュは規則的な高周波
        mesh_score = np.zeros((h, w), dtype=np.float32)

        # エッジ密度が高い（規則的なパターン）
        edge_threshold = np.percentile(edge_density, 70)
        high_edge = edge_density > edge_threshold

        # テクスチャ分散が中程度（人間の顔や髪は分散が高いか低い）
        std_low = np.percentile(local_std, 30)
        std_high = np.percentile(local_std, 70)
        medium_texture = (local_std > std_low) & (local_std < std_high)

        # メッシュスコア = 高エッジ AND 中分散 AND 前景領域
        mesh_candidate = high_edge & medium_texture & fg_mask
        mesh_score[mesh_candidate] = 1.0

        # メッシュ領域を膨張
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mesh_score = cv2.dilate(mesh_score, kernel_dilate, iterations=1)

        # スムージング
        mesh_score = cv2.GaussianBlur(mesh_score, (15, 15), 0)

        # メッシュ領域を80%抑制
        alpha_float = alpha.astype(np.float32)
        alpha_demeshed = alpha_float * (1 - mesh_score * 0.8)

        return alpha_demeshed.astype(np.uint8)

    def _init_mediapipe(self):
        """MediaPipeセグメンテーションを初期化"""
        if self.mediapipe_segmenter is None:
            try:
                import mediapipe as mp
                self.mediapipe_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                    model_selection=1  # 風景モデル（より正確）
                )
                print("MediaPipe Selfie Segmentation initialized.")
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
                self.mediapipe_segmenter = None

    def _get_mediapipe_human_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        MediaPipeで人物領域のみを検出

        Args:
            frame: BGR画像

        Returns:
            人物マスク（255=人物, 0=背景）
        """
        if self.mediapipe_segmenter is None:
            self._init_mediapipe()

        if self.mediapipe_segmenter is None:
            # MediaPipeが使えない場合は全体を人物として扱う
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mediapipe_segmenter.process(frame_rgb)

        if results.segmentation_mask is None:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # 0.3以上を人物として判定（厳し目）
        human_mask = (results.segmentation_mask > 0.3).astype(np.uint8) * 255

        return human_mask

    def set_background_reference(self, frame: np.ndarray):
        """
        背景参照フレームを設定（奏者がいない状態のフレーム）

        Args:
            frame: 背景のみのフレーム
        """
        self.background_reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Background reference frame set.")

    def _guided_filter(self, guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
        """
        ガイデッドフィルタによるエッジ精緻化
        元画像の色情報に基づいてマスクの境界を最適化

        Args:
            guide: ガイド画像（元のカラー画像）
            src: フィルタリング対象（アルファマスク）
            radius: フィルタ半径
            eps: 正則化パラメータ（小さいほどエッジ保持）

        Returns:
            精緻化されたマスク
        """
        try:
            # cv2.ximgprocが利用可能な場合
            result = cv2.ximgproc.guidedFilter(guide, src, radius, eps)
            return result
        except AttributeError:
            # ximgprocがない場合は手動実装
            return self._guided_filter_manual(guide, src, radius, eps)

    def _guided_filter_manual(self, guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """
        ガイデッドフィルタの手動実装（cv2.ximgprocがない場合）
        """
        # グレースケールに変換
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide.astype(np.float32) / 255.0

        src_float = src.astype(np.float32) / 255.0

        # ボックスフィルタで平均を計算
        ksize = (2 * radius + 1, 2 * radius + 1)

        mean_I = cv2.boxFilter(guide_gray, -1, ksize)
        mean_p = cv2.boxFilter(src_float, -1, ksize)
        mean_Ip = cv2.boxFilter(guide_gray * src_float, -1, ksize)
        mean_II = cv2.boxFilter(guide_gray * guide_gray, -1, ksize)

        # 共分散と分散
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # 線形係数
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # 平均化
        mean_a = cv2.boxFilter(a, -1, ksize)
        mean_b = cv2.boxFilter(b, -1, ksize)

        # 出力
        q = mean_a * guide_gray + mean_b
        return (np.clip(q, 0, 1) * 255).astype(np.uint8)

    def _detect_static_regions(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        静止物体（椅子・カーテン）を検出して【徹底的に】抑制

        過激な静止物体検出ロジック:
        - 狭いcenter_mask（体幹部分のみ保護）
        - 95-100%の強力な抑制
        - 45フレーム（約1.5秒）の長期バッファ
        - 右側領域への追加ペナルティ

        Args:
            frame: 現在のフレーム
            alpha: 現在のアルファマスク

        Returns:
            静止領域を抑制したマスク
        """
        h, w = alpha.shape[:2]

        # フレームバッファに追加
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray.copy())

        # 最大45フレーム保持（約1.5秒）
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)

        # 10フレーム以上ないと判定不可
        if len(self.frame_buffer) < 10:
            return alpha

        # === 長期的な静止領域検出 ===
        # フレーム間の差分を計算
        diffs = []
        for i in range(1, len(self.frame_buffer)):
            diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i-1])
            diffs.append(diff)

        # 差分の平均（動きの大きさ）
        avg_diff = np.mean(diffs, axis=0).astype(np.uint8)

        # 動きが非常に少ない領域 = 静止物体（閾値を厳しく: 3）
        _, static_regions = cv2.threshold(avg_diff, 3, 255, cv2.THRESH_BINARY_INV)

        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        static_regions = cv2.morphologyEx(static_regions, cv2.MORPH_OPEN, kernel)

        # === 体幹保護マスク（狭く！） ===
        # 奏者の体幹部分のみをカバー（幅を30%縮小）
        center_mask = np.zeros_like(alpha)
        # 幅: w//3 → w//5 (約40%縮小)
        # 高さ: h//2 → h*2//5 (20%縮小)
        cv2.ellipse(center_mask, (w//2, h//2), (w//5, h*2//5), 0, 0, 360, 255, -1)

        # === 地域別重み付け ===
        # 右側1/3には追加ペナルティ（椅子がある側）
        regional_weight = np.ones((h, w), dtype=np.float32)

        # 右側1/3: 抑制強度を1.5倍
        right_third_start = w * 2 // 3
        regional_weight[:, right_third_start:] = 1.5

        # 左側1/4: 抑制強度を1.2倍
        left_quarter_end = w // 4
        regional_weight[:, :left_quarter_end] = 1.2

        # 上下端: 抑制強度を1.3倍
        top_region = h // 6
        bottom_region = h * 5 // 6
        regional_weight[:top_region, :] *= 1.3
        regional_weight[bottom_region:, :] *= 1.3

        # === 抑制強度計算 ===
        # 中心マスク外の静止領域
        edge_static = cv2.bitwise_and(static_regions, cv2.bitwise_not(center_mask))

        # 基本抑制強度: 95%（中心部以外）
        base_suppression = 0.95

        # 抑制マップを作成
        suppression = edge_static.astype(np.float32) / 255.0 * base_suppression

        # 地域別重み付けを適用
        suppression = np.clip(suppression * regional_weight, 0, 1.0)

        # === 背景参照フレームによる追加抑制 ===
        if self.background_reference is not None:
            # 背景との差分が小さい領域 = 背景
            bg_diff = cv2.absdiff(gray, self.background_reference)
            _, bg_match = cv2.threshold(bg_diff, 25, 255, cv2.THRESH_BINARY_INV)

            # 背景と一致する領域は完全に消す
            bg_suppression = bg_match.astype(np.float32) / 255.0 * 0.98
            suppression = np.maximum(suppression, bg_suppression)

        # === アルファ抑制 ===
        alpha_float = alpha.astype(np.float32)
        alpha_suppressed = alpha_float * (1 - suppression)

        return alpha_suppressed.astype(np.uint8)

    def _ensemble_with_mediapipe(self, frame: np.ndarray, rvm_alpha: np.ndarray) -> np.ndarray:
        """
        MediaPipeとのアンサンブル（二重チェック）

        RVMが「前景」と判定したが、MediaPipeが「人物ではない」と判定した領域を除去
        ただし、ギター（細長い物体）は保護

        Args:
            frame: 元のカラーフレーム
            rvm_alpha: RVMによるアルファマスク

        Returns:
            アンサンブル後のアルファマスク
        """
        # MediaPipeで人物マスクを取得
        human_mask = self._get_mediapipe_human_mask(frame)

        # ギター保護: RVMマスクの中で「細長い」部分を検出
        # 輪郭分析で縦横比が大きい部分を保護
        guitar_protected = np.zeros_like(rvm_alpha)

        # RVMマスクの輪郭を検出
        _, binary = cv2.threshold(rvm_alpha, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 5:
                continue

            # 外接矩形の縦横比
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = max(cw, ch) / (min(cw, ch) + 1)

            # 縦横比が2.5以上 = ギターの可能性（細長い物体）
            if aspect_ratio > 2.5:
                cv2.drawContours(guitar_protected, [contour], -1, 255, -1)

        # 人物マスクを膨張（髪の毛などを保護）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        human_mask_dilated = cv2.dilate(human_mask, kernel, iterations=2)

        # アンサンブルマスク = 人物マスク OR ギター保護
        ensemble_mask = cv2.bitwise_or(human_mask_dilated, guitar_protected)

        # RVMマスクとアンサンブルマスクの交差部分のみを残す
        # RVMが前景と判定 AND (MediaPipeが人物 OR ギター保護) = 最終前景
        result = cv2.bitwise_and(rvm_alpha, ensemble_mask)

        # 元のRVMマスクと50%ブレンド（急激な変化を避ける）
        result_blended = cv2.addWeighted(rvm_alpha, 0.3, result, 0.7, 0)

        return result_blended

    def _grabcut_refinement(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        GrabCutによる境界再計算

        不確実な境界領域を色ベースで再分類

        Args:
            frame: 元のカラーフレーム
            alpha: 現在のアルファマスク

        Returns:
            精緻化されたマスク
        """
        h, w = frame.shape[:2]

        # トリマップを作成
        # 確実な前景(255) / 確実な背景(0) / 不確実(128)
        trimap = np.zeros((h, w), dtype=np.uint8)

        # 高信頼度の前景（alpha > 200）
        trimap[alpha > 200] = cv2.GC_FGD  # 確実な前景

        # 高信頼度の背景（alpha < 30）
        trimap[alpha < 30] = cv2.GC_BGD  # 確実な背景

        # 中間領域（不確実）
        uncertain = (alpha >= 30) & (alpha <= 200)
        trimap[uncertain] = cv2.GC_PR_FGD  # おそらく前景

        # GrabCut用のマスク
        mask = trimap.copy()

        # 不確実な領域がある場合のみGrabCut実行
        if np.sum(uncertain) > 100:
            try:
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)

                # GrabCutを実行（反復回数を少なく）
                cv2.grabCut(frame, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

                # 前景マスクを抽出
                refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

                # 元のマスクとブレンド（GrabCutの結果を50%反映）
                alpha_blend = cv2.addWeighted(alpha, 0.5, refined, 0.5, 0)
                return alpha_blend

            except cv2.error:
                # GrabCut失敗時は元のマスクを返す
                return alpha

        return alpha

    def _create_smooth_alpha_gradient(self, alpha: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        滑らかなアルファグラデーションを作成

        境界を「0か1か」ではなく連続的な透明度で表現

        Args:
            alpha: 入力マスク
            frame: 元のカラーフレーム

        Returns:
            滑らかなグラデーションを持つマスク
        """
        # エッジ検出で境界領域を特定
        edges = cv2.Canny(alpha, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edge_region = cv2.dilate(edges, kernel, iterations=2)

        # 距離変換で境界からの距離を計算
        dist_from_edge = cv2.distanceTransform(cv2.bitwise_not(edge_region), cv2.DIST_L2, 5)

        # 距離に基づいてグラデーションを作成
        max_dist = 15  # グラデーションの幅（ピクセル）
        gradient_weight = np.clip(dist_from_edge / max_dist, 0, 1)

        # 内部は元の値を保持、境界はガウシアンブラーでソフトに
        alpha_blurred = cv2.GaussianBlur(alpha, (21, 21), 0)

        # ブレンド
        alpha_float = alpha.astype(np.float32)
        blurred_float = alpha_blurred.astype(np.float32)

        result = alpha_float * gradient_weight + blurred_float * (1 - gradient_weight)

        return result.astype(np.uint8)

    def _morphological_cleanup(self, alpha: np.ndarray) -> np.ndarray:
        """
        精密なモルフォロジー演算でマスクを修正

        - 穴埋め（Closing）: フレット内の黒い穴を塗りつぶす
        - 膨張（Dilation）: ギターヘッドなど認識が甘い部分をカバー（抑制版）
        - 収縮（Erosion）: 過検出した背景を除去
        - オープニング: 孤立した背景ノイズを除去

        Args:
            alpha: 入力アルファマスク (0-255)

        Returns:
            修正されたアルファマスク
        """
        # ステップ1: クロージング処理（穴埋め）- 維持
        # フレット内の穴を確実に埋める
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # ステップ2: 小さい穴も埋める
        kernel_close_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close_small, iterations=2)

        # ステップ3: 膨張（Dilation）- 最小限
        # ギターヘッドをカバーしつつ、広がりすぎを最小化
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.dilate(alpha, kernel_dilate, iterations=1)  # 1回のみ

        # ステップ4: 収縮（Erosion）- 繊細な背景除去【重要】
        # 椅子やカーテンなど本体から離れた誤認識を除去
        # 非常に小さなカーネルで慎重に適用（ギターヘッドを消さないため）
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.erode(alpha, kernel_erode, iterations=1)

        # ステップ5: オープニング処理 - 孤立したノイズを除去
        # 本体から分離した小さな背景ノイズを消す
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # ステップ6: 最終クロージング - ギターの細かい穴を再度埋める
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_final, iterations=1)

        return alpha

    def _fill_holes_floodfill(self, alpha: np.ndarray) -> np.ndarray:
        """
        フラッドフィルで内部の穴を完全に埋める

        Args:
            alpha: 入力アルファマスク

        Returns:
            穴が埋められたマスク
        """
        # 二値化
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # 反転してフラッドフィル
        h, w = binary.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # 外側から塗りつぶし
        floodfilled = binary.copy()
        cv2.floodFill(floodfilled, mask, (0, 0), 255)

        # 反転して内部の穴を取得
        floodfilled_inv = cv2.bitwise_not(floodfilled)

        # 元のマスクと穴を結合
        filled = binary | floodfilled_inv

        return filled

    def _smooth_edges(self, alpha: np.ndarray, blur_size: int = 21) -> np.ndarray:
        """
        エッジを滑らかにしてジャギーを解消

        Args:
            alpha: 入力アルファマスク
            blur_size: ガウシアンブラーのカーネルサイズ

        Returns:
            エッジが滑らかになったマスク
        """
        # ガウシアンブラーでエッジを滑らかに
        if blur_size % 2 == 0:
            blur_size += 1

        smoothed = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

        return smoothed

    def _create_soft_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """
        滑らかなアルファグラデーションを作成（放送品質）

        Args:
            alpha: 入力アルファマスク

        Returns:
            ソフトなアルファマスク
        """
        # エッジ検出
        edges = cv2.Canny(alpha, 50, 150)

        # エッジ領域を拡張
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=3)

        # エッジ領域のみブラー
        blurred = cv2.GaussianBlur(alpha, (31, 31), 0)

        # エッジ領域はブラー版を使用、内部はシャープに
        edge_mask = edge_region.astype(np.float32) / 255.0
        result = alpha.astype(np.float32) * (1 - edge_mask) + blurred.astype(np.float32) * edge_mask

        return result.astype(np.uint8)

    def _temporal_smooth(self, alpha: np.ndarray) -> np.ndarray:
        """
        時間的平滑化（フレーム間のチラつき防止）
        """
        self.alpha_buffer.append(alpha.copy())

        if len(self.alpha_buffer) > self.buffer_size:
            self.alpha_buffer.pop(0)

        if len(self.alpha_buffer) < 3:
            return alpha

        # 重み付き平均（最新フレームに重みを置く）
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3][-len(self.alpha_buffer):])
        weights = weights / weights.sum()

        smoothed = np.zeros_like(alpha, dtype=np.float32)
        for i, buf_alpha in enumerate(self.alpha_buffer):
            smoothed += buf_alpha.astype(np.float32) * weights[i]

        return smoothed.astype(np.uint8)

    def extract_foreground(
        self,
        frame: np.ndarray,
        threshold: float = 0.1,  # 低い閾値で前景を広く拾う
        apply_morphology: bool = True,
        fill_holes: bool = True,
        smooth_edges: bool = True,
        temporal_smooth: bool = True,
        use_guided_filter: bool = True,
        suppress_static: bool = True,
        use_grabcut: bool = False,  # 重いので必要時のみ
        use_mediapipe_ensemble: bool = True,  # MediaPipeとの二重チェック
        use_brutal_exclusion: bool = True,    # 座標ベースの強制排除
        use_mesh_detection: bool = True,      # メッシュテクスチャ検出
        use_difference_attack: bool = True    # 【新】擬似背景との差分攻撃
    ) -> np.ndarray:
        """
        フレームから前景（人物+ギター）を抽出（最終決戦版）

        Args:
            frame: BGR形式のOpenCV画像
            threshold: マスク閾値（低いほど前景を広く拾う）
            apply_morphology: モルフォロジー演算を適用するか
            fill_holes: 穴埋め処理を適用するか
            smooth_edges: エッジ平滑化を適用するか
            temporal_smooth: 時間的平滑化を行うか
            use_guided_filter: ガイデッドフィルタでエッジ精緻化
            suppress_static: 静止物体（椅子・カーテン）を徹底抑制
            use_grabcut: GrabCutで境界を再計算（重い）
            use_mediapipe_ensemble: MediaPipeとの二重チェック
            use_brutal_exclusion: 座標ベースの強制排除（最終手段）
            use_mesh_detection: メッシュテクスチャを検出して除去
            use_difference_attack: 擬似背景との差分攻撃（AIの特性を逆手に取る）

        Returns:
            BGRA形式の画像（アルファチャンネル付き）
        """
        self._load_model()

        h, w = frame.shape[:2]
        current_size = (h, w)

        # フレームサイズが変わった場合、時間的状態をリセット
        if self.last_frame_size != current_size:
            self.reset_temporal_state()
            self.last_frame_size = current_size

        # BGR -> RGB、正規化、テンソル変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        src = src.to(self.device)

        # RVM推論
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)

        # アルファマスクをnumpy配列に変換
        alpha_raw = pha[0, 0].cpu().numpy()

        # 低い閾値で前景を広く拾う
        # 閾値より上を255に、閾値以下を0に（ただしグラデーションを保持）
        alpha = np.clip((alpha_raw - threshold) / (1.0 - threshold), 0, 1)
        alpha = (alpha * 255).astype(np.uint8)

        # マスクを元のサイズにリサイズ
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        # === 最終決戦：徹底背景排除処理 ===

        # 1. MediaPipeとのアンサンブル【重要】
        #    RVMが拾いすぎた背景をMediaPipeで除去
        if use_mediapipe_ensemble:
            alpha = self._ensemble_with_mediapipe(frame, alpha)

        # 2. 【新】擬似背景との差分攻撃【最重要】
        #    AIが「人物」と思う部分を背景色で塗りつぶし、
        #    それと実際のフレームを比較して背景を特定
        if use_difference_attack:
            alpha = self._difference_attack(frame, alpha)

        # 3. ガイデッドフィルタによるエッジ精緻化
        #    元画像の色境界に沿ってマスクを最適化
        if use_guided_filter:
            alpha = self._guided_filter(frame, alpha, radius=8, eps=0.001)

        # 4. 穴埋め（フラッドフィル）
        if fill_holes:
            alpha = self._fill_holes_floodfill(alpha)

        # 5. モルフォロジー演算（最小限）
        if apply_morphology:
            alpha = self._morphological_cleanup(alpha)

        # 6. 静止物体の徹底抑制【重要】
        #    椅子・カーテンなど動かない背景を95-100%除去
        if suppress_static:
            alpha = self._detect_static_regions(frame, alpha)

        # 7. メッシュテクスチャ検出
        #    椅子のメッシュのような規則的パターンを除去
        if use_mesh_detection:
            alpha = self._detect_mesh_texture(frame, alpha)

        # 8. 【暴力的】座標ベースの強制排除【最終手段】
        #    右側30%を問答無用で透明化（動きがある場合のみ保護）
        if use_brutal_exclusion:
            alpha = self._brutal_exclusion_zone(frame, alpha)

        # 9. GrabCutによる境界再計算（オプション）
        if use_grabcut:
            alpha = self._grabcut_refinement(frame, alpha)

        # 10. 滑らかなアルファグラデーション
        if smooth_edges:
            alpha = self._create_smooth_alpha_gradient(alpha, frame)

        # 11. 時間的平滑化
        if temporal_smooth:
            alpha = self._temporal_smooth(alpha)

        # BGRA画像を作成
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        return bgra

    def composite_with_background(
        self,
        foreground_bgra: np.ndarray,
        background: np.ndarray,
        decontaminate: bool = True
    ) -> np.ndarray:
        """
        前景を背景と合成（デコンタミネーション付き）
        """
        h, w = foreground_bgra.shape[:2]

        # 背景をリサイズ
        bg_resized = cv2.resize(background, (w, h))

        # アルファチャンネルを取得
        alpha = foreground_bgra[:, :, 3:4].astype(np.float32) / 255.0
        fg_bgr = foreground_bgra[:, :, :3].astype(np.float32)
        bg_float = bg_resized.astype(np.float32)

        if decontaminate:
            # エッジ検出
            edge_mask = cv2.Canny(foreground_bgra[:, :, 3], 30, 100)
            edge_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=3)
            edge_mask = cv2.GaussianBlur(edge_mask, (11, 11), 0)
            edge_weight = edge_mask.astype(np.float32) / 255.0

            # エッジ付近の色被りを除去
            decontam_factor = 0.4
            for c in range(3):
                fg_bgr[:, :, c] = fg_bgr[:, :, c] * (1 - edge_weight * decontam_factor) + \
                                  bg_float[:, :, c] * edge_weight * decontam_factor

        # アルファブレンド合成
        composite = (fg_bgr * alpha + bg_float * (1 - alpha)).astype(np.uint8)

        return composite

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        背景を除去（video_processorとの互換性用インターフェース）
        最終決戦版 - 全ての武器を投入（擬似背景差分攻撃追加）
        """
        return self.extract_foreground(
            frame,
            threshold=0.1,  # 低い閾値
            apply_morphology=True,
            fill_holes=True,
            smooth_edges=True,
            temporal_smooth=True,
            use_guided_filter=True,       # ガイデッドフィルタON
            suppress_static=True,         # 静止物体抑制ON（95-100%）
            use_grabcut=False,            # GrabCutはOFF（重いため）
            use_mediapipe_ensemble=True,  # MediaPipe二重チェックON
            use_brutal_exclusion=True,    # 座標ベース強制排除ON
            use_mesh_detection=True,      # メッシュテクスチャ検出ON
            use_difference_attack=True    # 【新】擬似背景との差分攻撃ON
        )


# 1フレームテスト用関数
def test_single_frame(image_path: str, output_path: str):
    """
    1フレームに対してマスク品質をテスト

    Args:
        image_path: 入力画像パス
        output_path: 出力画像パス
    """
    import cv2

    service = VideoMattingService(downsample_ratio=0.5)

    # 画像を読み込み
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read {image_path}")
        return

    print(f"Processing {image_path}...")
    print(f"Input size: {frame.shape}")

    # 前景抽出
    result = service.extract_foreground(
        frame,
        threshold=0.1,
        apply_morphology=True,
        fill_holes=True,
        smooth_edges=True,
        temporal_smooth=False  # 単一フレームなので無効
    )

    # 結果を保存
    cv2.imwrite(output_path, result)
    print(f"Saved to {output_path}")

    # マスクのみも保存
    mask_path = output_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_path, result[:, :, 3])
    print(f"Mask saved to {mask_path}")

    # 統計情報
    alpha = result[:, :, 3]
    print(f"Alpha stats: min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        test_single_frame(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python video_matting.py <input_image> <output_image>")
        print("\nRunning default test...")

        # デフォルトテスト
        service = VideoMattingService()
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.ellipse(test_img, (640, 400), (200, 350), 0, 0, 360, (100, 150, 200), -1)
        cv2.circle(test_img, (640, 100), 80, (200, 180, 160), -1)

        result = service.remove_background(test_img)
        print(f"Output shape: {result.shape}")
        print(f"Alpha range: {result[:,:,3].min()} - {result[:,:,3].max()}")
