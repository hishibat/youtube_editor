"""
ハイライト自動検出サービス

機能:
1. librosaによる音声解析
2. RMS（音量）とスペクトル重心によるピーク検出
3. ギターソロなどのクライマックス自動検出
4. 動画再構成（フック構造：ハイライトを冒頭に配置）
"""

import os
import subprocess
import shutil
import tempfile
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HighlightSegment:
    """ハイライトセグメント"""
    start_time: float  # 開始時間（秒）
    end_time: float    # 終了時間（秒）
    score: float       # スコア（0-1、高いほど重要）
    type: str = "peak"  # peak, climax, intro, etc.


@dataclass
class AudioFeatures:
    """音声特徴量"""
    rms: np.ndarray           # RMS（音量）
    spectral_centroid: np.ndarray  # スペクトル重心（明るさ）
    onset_strength: np.ndarray     # オンセット強度
    tempo: float                    # テンポ（BPM）
    beat_times: np.ndarray         # ビート時刻


class HighlightService:
    """
    ハイライト自動検出サービス

    音声解析によりギターソロやクライマックスを自動検出し、
    動画を再構成（フック構造）
    """

    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        self.sr = 22050  # サンプリングレート
        self.hop_length = 512  # ホップ長

        # librosaの遅延インポート
        self._librosa = None

        print("HighlightService initialized")

    def _load_librosa(self):
        """librosaを遅延読み込み"""
        if self._librosa is not None:
            return self._librosa

        try:
            import librosa
            self._librosa = librosa
            print("librosa loaded successfully")
            return librosa
        except ImportError:
            print("librosa not installed. Installing...")
            subprocess.run(["pip", "install", "librosa"], check=True)
            import librosa
            self._librosa = librosa
            return librosa

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """
        動画から音声を抽出（WAV形式）

        Args:
            video_path: 入力動画パス
            output_path: 出力音声パス

        Returns:
            成功したかどうか
        """
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sr),
            "-ac", "1",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return False

    def analyze_audio(self, audio_path: str) -> Optional[AudioFeatures]:
        """
        音声ファイルを解析

        Args:
            audio_path: 音声ファイルパス

        Returns:
            音声特徴量
        """
        librosa = self._load_librosa()

        try:
            # 音声を読み込み
            y, sr = librosa.load(audio_path, sr=self.sr)

            # RMS（音量）
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

            # スペクトル重心（音の明るさ）
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=self.hop_length
            )[0]

            # オンセット強度
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            )

            # テンポとビート
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=self.hop_length
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

            return AudioFeatures(
                rms=rms,
                spectral_centroid=spectral_centroid,
                onset_strength=onset_env,
                tempo=float(tempo),
                beat_times=beat_times
            )

        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return None

    def detect_highlights(
        self,
        video_path: str,
        min_duration: float = 2.0,
        max_duration: float = 5.0,
        top_k: int = 3
    ) -> List[HighlightSegment]:
        """
        動画からハイライトを自動検出

        Args:
            video_path: 動画パス
            min_duration: 最小ハイライト長（秒）
            max_duration: 最大ハイライト長（秒）
            top_k: 検出する上位ハイライト数

        Returns:
            ハイライトセグメントのリスト
        """
        librosa = self._load_librosa()

        with tempfile.TemporaryDirectory() as temp_dir:
            # 音声を抽出
            audio_path = os.path.join(temp_dir, "audio.wav")
            if not self.extract_audio(video_path, audio_path):
                print("Failed to extract audio")
                return []

            # 音声を解析
            features = self.analyze_audio(audio_path)
            if features is None:
                return []

            # 時間軸を作成
            times = librosa.frames_to_time(
                np.arange(len(features.rms)),
                sr=self.sr,
                hop_length=self.hop_length
            )

            # 複合スコアを計算
            # RMS（音量）を正規化
            rms_norm = (features.rms - features.rms.min()) / (features.rms.max() - features.rms.min() + 1e-6)

            # スペクトル重心を正規化
            sc_norm = (features.spectral_centroid - features.spectral_centroid.min()) / (features.spectral_centroid.max() - features.spectral_centroid.min() + 1e-6)

            # オンセット強度を正規化
            onset_norm = (features.onset_strength - features.onset_strength.min()) / (features.onset_strength.max() - features.onset_strength.min() + 1e-6)

            # 複合スコア（音量 + 明るさ + オンセット）
            composite_score = 0.4 * rms_norm + 0.3 * sc_norm + 0.3 * onset_norm[:len(rms_norm)]

            # スムージング
            window_size = int(0.5 * self.sr / self.hop_length)  # 0.5秒
            composite_smooth = np.convolve(
                composite_score,
                np.ones(window_size) / window_size,
                mode='same'
            )

            # ピークを検出
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(
                composite_smooth,
                distance=int(min_duration * self.sr / self.hop_length),
                prominence=0.1
            )

            # スコアでソート
            peak_scores = composite_smooth[peaks]
            sorted_indices = np.argsort(peak_scores)[::-1]

            # 上位k個を選択
            highlights = []
            used_ranges = []

            for idx in sorted_indices:
                if len(highlights) >= top_k:
                    break

                peak_idx = peaks[idx]
                peak_time = times[peak_idx]
                score = float(peak_scores[idx])

                # ハイライト範囲を計算（ピークの前後）
                start_time = max(0, peak_time - min_duration / 2)
                end_time = min(times[-1], peak_time + min_duration / 2)

                # 重複チェック
                overlaps = False
                for used_start, used_end in used_ranges:
                    if not (end_time <= used_start or start_time >= used_end):
                        overlaps = True
                        break

                if overlaps:
                    continue

                used_ranges.append((start_time, end_time))

                # ハイライトタイプを判定
                if score > 0.8:
                    hl_type = "climax"
                elif features.spectral_centroid[peak_idx] > np.percentile(features.spectral_centroid, 80):
                    hl_type = "bright_moment"
                else:
                    hl_type = "peak"

                highlights.append(HighlightSegment(
                    start_time=start_time,
                    end_time=end_time,
                    score=score,
                    type=hl_type
                ))

            # 時間順にソート
            highlights.sort(key=lambda x: x.start_time)

            return highlights

    def detect_guitar_solo(
        self,
        video_path: str,
        min_solo_duration: float = 3.0
    ) -> Optional[HighlightSegment]:
        """
        ギターソロを検出

        ギターソロの特徴:
        - 高いスペクトル重心（明るい音色）
        - 持続的な音量
        - 高いオンセット密度

        Args:
            video_path: 動画パス
            min_solo_duration: 最小ソロ長（秒）

        Returns:
            ギターソロセグメント（なければNone）
        """
        librosa = self._load_librosa()

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            if not self.extract_audio(video_path, audio_path):
                return None

            features = self.analyze_audio(audio_path)
            if features is None:
                return None

            times = librosa.frames_to_time(
                np.arange(len(features.rms)),
                sr=self.sr,
                hop_length=self.hop_length
            )

            # ギターソロの指標
            # 1. 高いスペクトル重心（高音域）
            sc_threshold = np.percentile(features.spectral_centroid, 70)
            high_freq_mask = features.spectral_centroid > sc_threshold

            # 2. 高い音量
            rms_threshold = np.percentile(features.rms, 60)
            loud_mask = features.rms > rms_threshold

            # 3. 複合マスク
            solo_mask = high_freq_mask & loud_mask[:len(high_freq_mask)]

            # 連続した領域を検出
            solo_regions = []
            start_idx = None

            for i, is_solo in enumerate(solo_mask):
                if is_solo and start_idx is None:
                    start_idx = i
                elif not is_solo and start_idx is not None:
                    duration = times[i] - times[start_idx]
                    if duration >= min_solo_duration:
                        solo_regions.append((times[start_idx], times[i], duration))
                    start_idx = None

            if start_idx is not None:
                duration = times[-1] - times[start_idx]
                if duration >= min_solo_duration:
                    solo_regions.append((times[start_idx], times[-1], duration))

            if not solo_regions:
                return None

            # 最も長いソロ領域を選択
            best_region = max(solo_regions, key=lambda x: x[2])

            return HighlightSegment(
                start_time=best_region[0],
                end_time=best_region[1],
                score=0.9,
                type="guitar_solo"
            )

    def restructure_video_with_hook(
        self,
        input_video: str,
        output_video: str,
        highlight: HighlightSegment,
        hook_duration: float = 3.0
    ) -> bool:
        """
        動画をフック構造に再構成

        構造: ハイライト（冒頭）→ 本編全体

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            highlight: フックとして使用するハイライト
            hook_duration: フック部分の長さ（秒）

        Returns:
            成功したかどうか
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # ハイライト部分を抽出
            hook_path = os.path.join(temp_dir, "hook.mp4")
            hook_start = highlight.start_time
            hook_end = min(highlight.end_time, hook_start + hook_duration)

            cmd_hook = [
                self.ffmpeg_path,
                "-y",
                "-i", input_video,
                "-ss", str(hook_start),
                "-t", str(hook_end - hook_start),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                hook_path
            ]

            result = subprocess.run(cmd_hook, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to extract hook: {result.stderr}")
                return False

            # トランジションエフェクト（フラッシュ）を追加
            hook_with_transition = os.path.join(temp_dir, "hook_trans.mp4")
            cmd_transition = [
                self.ffmpeg_path,
                "-y",
                "-i", hook_path,
                "-vf", "fade=out:st=" + str(hook_end - hook_start - 0.3) + ":d=0.3:color=white",
                "-c:v", "libx264",
                "-c:a", "copy",
                "-preset", "fast",
                hook_with_transition
            ]

            result = subprocess.run(cmd_transition, capture_output=True, text=True)
            if result.returncode != 0:
                # トランジション失敗時はフック単体を使用
                hook_with_transition = hook_path

            # 結合リストを作成
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, "w") as f:
                f.write(f"file '{hook_with_transition}'\n")
                f.write(f"file '{input_video}'\n")

            # 結合
            cmd_concat = [
                self.ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-movflags", "+faststart",
                output_video
            ]

            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to concat: {result.stderr}")
                return False

            return True

    def auto_restructure(
        self,
        input_video: str,
        output_video: str,
        hook_duration: float = 3.0
    ) -> Dict[str, Any]:
        """
        動画を自動的にフック構造に再構成

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            hook_duration: フック部分の長さ（秒）

        Returns:
            結果情報（検出したハイライトなど）
        """
        result = {
            "success": False,
            "highlights": [],
            "hook": None,
            "message": ""
        }

        # ハイライトを検出
        highlights = self.detect_highlights(
            input_video,
            min_duration=hook_duration,
            max_duration=hook_duration + 2,
            top_k=3
        )

        if not highlights:
            result["message"] = "No highlights detected"
            # 元動画をコピー
            shutil.copy(input_video, output_video)
            return result

        result["highlights"] = [
            {
                "start": h.start_time,
                "end": h.end_time,
                "score": h.score,
                "type": h.type
            }
            for h in highlights
        ]

        # 最高スコアのハイライトをフックとして使用
        best_highlight = max(highlights, key=lambda x: x.score)
        result["hook"] = {
            "start": best_highlight.start_time,
            "end": best_highlight.end_time,
            "score": best_highlight.score,
            "type": best_highlight.type
        }

        # 再構成
        success = self.restructure_video_with_hook(
            input_video,
            output_video,
            best_highlight,
            hook_duration
        )

        result["success"] = success
        result["message"] = "Video restructured with hook" if success else "Restructure failed"

        return result


# テスト用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python highlight_service.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    service = HighlightService()

    print("Detecting highlights...")
    highlights = service.detect_highlights(video_path)

    for i, h in enumerate(highlights):
        print(f"Highlight {i+1}: {h.start_time:.2f}s - {h.end_time:.2f}s (score: {h.score:.2f}, type: {h.type})")

    if highlights:
        output_path = video_path.replace(".mp4", "_hook.mp4")
        print(f"\nRestructuring video with hook...")
        success = service.restructure_video_with_hook(
            video_path, output_path, highlights[0]
        )
        if success:
            print(f"Output: {output_path}")
