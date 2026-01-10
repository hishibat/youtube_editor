"""
テロップ（字幕）生成サービス

機能:
1. 日本語フォント対応（Noto Sans JP）
2. FFmpeg/MoviePyベースのレンダリング
3. アウトライン付きテキスト
4. 位置指定（YouTube UIを避けて下部配置）
"""

import os
import subprocess
import shutil
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SubtitleEntry:
    """字幕エントリ"""
    start_time: float  # 開始時間（秒）
    end_time: float    # 終了時間（秒）
    text: str          # テキスト
    style: str = "default"  # スタイル名


@dataclass
class SubtitleStyle:
    """字幕スタイル"""
    font_name: str = "Noto Sans JP"
    font_size: int = 48
    primary_color: str = "&HFFFFFF"  # 白
    outline_color: str = "&H000000"  # 黒
    outline_width: int = 3
    shadow: bool = True
    bold: bool = True
    alignment: int = 2  # 1=左, 2=中央, 3=右
    margin_bottom: int = 60  # 下マージン（YouTube UIを避ける）


class SubtitleService:
    """
    字幕生成サービス

    FFmpegのdrawtextフィルターまたはASS形式を使用して
    高品質な字幕を動画に焼き付け
    """

    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe"

        # デフォルトスタイル
        self.styles: Dict[str, SubtitleStyle] = {
            "default": SubtitleStyle(),
            "emphasis": SubtitleStyle(
                font_size=56,
                primary_color="&H00FFFF",  # 黄色
                bold=True
            ),
            "small": SubtitleStyle(
                font_size=36,
                outline_width=2
            )
        }

        # フォントパスを検索
        self.font_path = self._find_japanese_font()
        print(f"SubtitleService initialized (font: {self.font_path})")

    def _find_japanese_font(self) -> Optional[str]:
        """日本語フォントを検索"""
        # 優先順位の高い日本語フォント
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/NotoSansJP-Bold.ttf",
            "C:/Windows/Fonts/NotoSansJP-Regular.ttf",
            "C:/Windows/Fonts/YuGothB.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
            # macOS
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/Library/Fonts/NotoSansJP-Bold.otf",
        ]

        for font_path in font_candidates:
            if os.path.exists(font_path):
                return font_path

        # システムフォントを検索
        try:
            if os.name == 'nt':  # Windows
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, value, _ = winreg.EnumValue(key, i)
                    if "noto" in name.lower() or "jp" in name.lower():
                        return f"C:/Windows/Fonts/{value}"
        except Exception:
            pass

        return None

    def add_style(self, name: str, style: SubtitleStyle):
        """カスタムスタイルを追加"""
        self.styles[name] = style

    def create_ass_file(
        self,
        subtitles: List[SubtitleEntry],
        output_path: str,
        video_width: int = 1080,
        video_height: int = 1920
    ) -> str:
        """
        ASS形式の字幕ファイルを作成

        Args:
            subtitles: 字幕エントリのリスト
            output_path: 出力パス
            video_width: 動画幅
            video_height: 動画高さ

        Returns:
            作成したファイルのパス
        """
        # ASSヘッダー
        ass_content = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

        # スタイル定義
        for name, style in self.styles.items():
            bold = -1 if style.bold else 0
            shadow = 2 if style.shadow else 0

            ass_content += f"Style: {name},{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},&H80000000,{bold},0,0,0,100,100,0,0,1,{style.outline_width},{shadow},{style.alignment},10,10,{style.margin_bottom},1\n"

        # ダイアログセクション
        ass_content += "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

        for subtitle in subtitles:
            start = self._format_ass_time(subtitle.start_time)
            end = self._format_ass_time(subtitle.end_time)
            style = subtitle.style if subtitle.style in self.styles else "default"

            # テキストのエスケープ
            text = subtitle.text.replace("\n", "\\N")

            ass_content += f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n"

        # ファイルに書き込み
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(ass_content)

        return output_path

    def _format_ass_time(self, seconds: float) -> str:
        """秒をASS時間形式に変換（H:MM:SS.cc）"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def burn_subtitles(
        self,
        input_video: str,
        output_video: str,
        subtitles: List[SubtitleEntry],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        字幕を動画に焼き付け

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            subtitles: 字幕エントリのリスト
            progress_callback: 進捗コールバック

        Returns:
            成功したかどうか
        """
        if not subtitles:
            # 字幕がない場合はコピー
            shutil.copy(input_video, output_video)
            return True

        # 動画情報を取得
        video_info = self._get_video_info(input_video)
        if not video_info:
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            # ASSファイルを作成
            ass_path = os.path.join(temp_dir, "subtitles.ass")
            self.create_ass_file(
                subtitles, ass_path,
                video_info['width'], video_info['height']
            )

            # FFmpegで字幕を焼き付け
            # assフィルターまたはsubtitlesフィルターを使用
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", input_video,
                "-vf", f"ass='{ass_path}'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "copy",
                "-movflags", "+faststart",
                output_video
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )

                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    return False

                return True

            except Exception as e:
                print(f"Subtitle burn failed: {e}")
                return False

    def burn_subtitles_drawtext(
        self,
        input_video: str,
        output_video: str,
        subtitles: List[SubtitleEntry]
    ) -> bool:
        """
        drawtextフィルターで字幕を焼き付け（ASS未対応環境用）

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            subtitles: 字幕エントリのリスト

        Returns:
            成功したかどうか
        """
        if not subtitles:
            shutil.copy(input_video, output_video)
            return True

        if not self.font_path:
            print("Warning: Japanese font not found")
            return False

        # drawtextフィルターを構築
        filter_parts = []

        for i, sub in enumerate(subtitles):
            style = self.styles.get(sub.style, self.styles["default"])

            # テキストのエスケープ
            text = sub.text.replace("'", r"\'").replace(":", r"\:")

            filter_part = (
                f"drawtext=fontfile='{self.font_path}':"
                f"text='{text}':"
                f"fontsize={style.font_size}:"
                f"fontcolor=white:"
                f"borderw={style.outline_width}:"
                f"bordercolor=black:"
                f"x=(w-text_w)/2:"
                f"y=h-{style.margin_bottom}-text_h:"
                f"enable='between(t,{sub.start_time},{sub.end_time})'"
            )
            filter_parts.append(filter_part)

        filter_complex = ",".join(filter_parts)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", input_video,
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Drawtext subtitle burn failed: {e}")
            return False

    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """動画情報を取得"""
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                stream = data.get("streams", [{}])[0]
                return {
                    "width": int(stream.get("width", 1080)),
                    "height": int(stream.get("height", 1920)),
                    "duration": float(stream.get("duration", 0))
                }
        except Exception as e:
            print(f"Failed to get video info: {e}")
        return None

    def generate_karaoke_subtitles(
        self,
        lyrics: List[Tuple[float, float, str]],
        highlight_color: str = "&H00FFFF"
    ) -> List[SubtitleEntry]:
        """
        カラオケ風字幕を生成

        Args:
            lyrics: (開始時間, 終了時間, 歌詞)のリスト
            highlight_color: ハイライト色

        Returns:
            字幕エントリのリスト
        """
        # カラオケスタイルを追加
        self.add_style("karaoke", SubtitleStyle(
            font_size=52,
            primary_color=highlight_color,
            bold=True,
            outline_width=4
        ))

        entries = []
        for start, end, text in lyrics:
            entries.append(SubtitleEntry(
                start_time=start,
                end_time=end,
                text=text,
                style="karaoke"
            ))

        return entries


# テスト用
if __name__ == "__main__":
    service = SubtitleService()

    # テスト字幕
    subtitles = [
        SubtitleEntry(0.0, 3.0, "こんにちは"),
        SubtitleEntry(3.0, 6.0, "テスト字幕です"),
        SubtitleEntry(6.0, 10.0, "日本語フォント対応！", style="emphasis"),
    ]

    # ASSファイル生成テスト
    service.create_ass_file(subtitles, "test_subtitles.ass", 1080, 1920)
    print("ASS file created: test_subtitles.ass")
