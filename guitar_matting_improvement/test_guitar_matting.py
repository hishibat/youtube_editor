#!/usr/bin/env python3
"""
ギター動画背景除去 品質テストスクリプト

放送品質レベルを達成するための自律改善ループ用テスト
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.robust_video_segmentation import RobustVideoSegmentation, KeyframeAnnotation


def get_sample_video_path() -> str:
    """サンプル動画のパスを取得"""
    sample_dir = Path(__file__).parent.parent / "sample"
    for f in sample_dir.iterdir():
        if f.suffix.lower() == ".mp4":
            return str(f)
    raise FileNotFoundError("No sample video found in sample directory")


def extract_test_frames(video_path: str, frame_indices: list) -> list:
    """テスト用フレームを抽出"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

    cap.release()
    return frames


def auto_detect_foreground_background_points(frame: np.ndarray) -> tuple:
    """
    フレームから自動的に前景/背景ポイントを検出

    この動画の実際のレイアウト（debug画像から確認）:
    - 人物は画面中央〜やや左
    - ギターは斜めに配置（左下から右上へ）
    - 椅子は右側
    - 背景には模様のあるカーテンと壁
    """
    h, w = frame.shape[:2]

    # 前景ポイント（人物+ギター）- より多くのポイントで確実にカバー
    foreground_points = [
        # 人物の体（中央）
        (int(w * 0.35), int(h * 0.20)),  # 頭/肩
        (int(w * 0.35), int(h * 0.35)),  # 胸
        (int(w * 0.35), int(h * 0.50)),  # 腹部
        (int(w * 0.35), int(h * 0.65)),  # 太もも
        (int(w * 0.30), int(h * 0.80)),  # 膝

        # ギターボディ（中央下）
        (int(w * 0.15), int(h * 0.45)),  # ギター左端
        (int(w * 0.20), int(h * 0.50)),
        (int(w * 0.25), int(h * 0.55)),

        # ギターネック（左上に伸びる）
        (int(w * 0.20), int(h * 0.35)),
        (int(w * 0.15), int(h * 0.30)),
        (int(w * 0.10), int(h * 0.25)),

        # 手（指板付近）
        (int(w * 0.50), int(h * 0.40)),
        (int(w * 0.55), int(h * 0.45)),

        # ギターヘッド（右上）
        (int(w * 0.70), int(h * 0.30)),
        (int(w * 0.75), int(h * 0.25)),
    ]

    # 背景ポイント - 椅子/機材を明示的に背景として指定
    # 重要: 椅子アームレストは x=0.55-0.70 にある（右端ではない）
    background_points = [
        # 四隅
        (20, 20),
        (w - 20, 20),
        (20, h - 20),
        (w - 20, h - 20),

        # 椅子のアームレスト（実際の位置: x=0.55-0.70, y=0.70-0.90）
        (int(w * 0.58), int(h * 0.72)),
        (int(w * 0.60), int(h * 0.75)),
        (int(w * 0.62), int(h * 0.78)),
        (int(w * 0.64), int(h * 0.80)),
        (int(w * 0.66), int(h * 0.82)),
        (int(w * 0.68), int(h * 0.85)),

        # 椅子座面
        (int(w * 0.50), int(h * 0.88)),
        (int(w * 0.55), int(h * 0.90)),

        # アンプ/機材（右側）
        (int(w * 0.85), int(h * 0.40)),
        (int(w * 0.90), int(h * 0.50)),
        (int(w * 0.92), int(h * 0.60)),

        # 右上の背景（壁）
        (int(w * 0.95), int(h * 0.10)),
        (int(w * 0.90), int(h * 0.15)),

        # 左上の背景
        (int(w * 0.05), int(h * 0.05)),
        (int(w * 0.05), int(h * 0.15)),

        # 床
        (int(w * 0.15), int(h * 0.95)),
        (int(w * 0.30), int(h * 0.95)),
    ]

    return foreground_points, background_points


def evaluate_mask_quality(mask: np.ndarray, frame: np.ndarray) -> dict:
    """
    マスクの品質を評価

    Returns:
        品質メトリクスの辞書
    """
    h, w = mask.shape[:2]

    # 基本的なマスク統計
    foreground_ratio = np.sum(mask > 127) / (h * w)

    # エッジの滑らかさを評価（Laplacianの分散）
    edges = cv2.Canny(mask, 50, 150)
    edge_pixels = np.sum(edges > 0)

    # マスクの連続性（穴の数）
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    num_components = len(contours)

    # エッジのギザギザ度（高周波成分）
    mask_float = mask.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(mask_float, cv2.CV_32F)
    edge_roughness = np.std(laplacian)

    return {
        "foreground_ratio": foreground_ratio,
        "edge_pixels": edge_pixels,
        "num_components": num_components,
        "edge_roughness": edge_roughness,
    }


def run_test():
    """メインテスト実行"""
    print("=" * 60)
    print("ギター動画背景除去 品質テスト")
    print("=" * 60)

    # 出力ディレクトリ作成
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # サンプル動画を取得
    try:
        video_path = get_sample_video_path()
        print(f"\n入力動画: {video_path}")
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return False

    # 動画情報を取得
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"解像度: {width}x{height}")
    print(f"フレーム数: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"長さ: {duration:.2f}秒")

    # テスト用フレームを抽出（最初、中間、最後）
    test_frame_indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4]
    test_frames = extract_test_frames(video_path, test_frame_indices)

    print(f"\nテストフレーム数: {len(test_frames)}")

    # 最初のフレームから前景/背景ポイントを検出
    first_frame = test_frames[0][1]
    fg_points, bg_points = auto_detect_foreground_background_points(first_frame)

    print(f"前景ポイント: {len(fg_points)}")
    print(f"背景ポイント: {len(bg_points)}")

    # RobustVideoSegmentation を初期化（auto = CPU: small, GPU: large）
    print("\nRobustVideoSegmentation を初期化中...")
    try:
        service = RobustVideoSegmentation(model_size="auto")
    except Exception as e:
        print(f"サービス初期化エラー: {e}")
        return False

    # 単一フレームテスト
    print("\n--- 単一フレームテスト ---")

    for idx, frame in test_frames:
        print(f"\nフレーム {idx} を処理中...")
        start_time = time.time()

        try:
            mask = service.segment_single_frame(
                frame,
                foreground_points=fg_points,
                background_points=bg_points
            )

            elapsed = time.time() - start_time
            print(f"  処理時間: {elapsed:.2f}秒")

            # 品質評価
            quality = evaluate_mask_quality(mask, frame)
            print(f"  前景比率: {quality['foreground_ratio']:.2%}")
            print(f"  エッジピクセル: {quality['edge_pixels']}")
            print(f"  連結成分数: {quality['num_components']}")
            print(f"  エッジ粗さ: {quality['edge_roughness']:.4f}")

            # マスクを保存
            mask_path = frames_dir / f"mask_{idx:06d}.png"
            cv2.imwrite(str(mask_path), mask)

            # 合成画像を保存（グリーンバック）
            alpha = mask.astype(np.float32) / 255.0
            alpha = alpha[:, :, np.newaxis]
            green_bg = np.zeros_like(frame)
            green_bg[:] = (0, 177, 64)
            composite = (frame.astype(np.float32) * alpha +
                        green_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            composite_path = frames_dir / f"composite_{idx:06d}.jpg"
            cv2.imwrite(str(composite_path), composite)

            # ポイントを可視化
            debug_frame = frame.copy()
            for px, py in fg_points:
                cv2.circle(debug_frame, (px, py), 8, (0, 255, 0), -1)
            for px, py in bg_points:
                cv2.circle(debug_frame, (px, py), 8, (0, 0, 255), -1)
            debug_path = frames_dir / f"debug_{idx:06d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)

            print(f"  保存: {mask_path.name}, {composite_path.name}")

        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()

    # 動画全体の処理テスト（最初の5秒のみ）
    print("\n--- 動画処理テスト（最初の5秒）---")

    output_video_path = output_dir / "test_output.mp4"
    test_duration_frames = min(int(fps * 5), total_frames)  # 最初の5秒

    # キーフレームアノテーション
    keyframe_annotations = [
        KeyframeAnnotation(
            frame_idx=0,
            foreground_points=fg_points,
            background_points=bg_points
        )
    ]

    def progress_callback(current, total, status, eta):
        if current % 30 == 0 or current == total:
            print(f"  進捗: {current}/{total} ({status}) ETA: {eta:.1f}s")

    start_time = time.time()

    try:
        # FFmpegで一時的な短い動画を作成（音声も保持）
        temp_video_path = output_dir / "temp_input.mp4"
        test_duration_sec = test_duration_frames / fps

        import subprocess
        import shutil
        ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"

        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-t", str(test_duration_sec),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "copy",
            str(temp_video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Failed to create test video")

        print(f"\nテスト動画を作成: {test_duration_frames}フレーム（音声付き）")

        # 動画処理
        service.process_video_with_propagation(
            input_path=str(temp_video_path),
            output_path=str(output_video_path),
            keyframe_annotations=keyframe_annotations,
            progress_callback=progress_callback
        )

        elapsed = time.time() - start_time
        fps_actual = test_duration_frames / elapsed if elapsed > 0 else 0

        print(f"\n処理完了!")
        print(f"  処理時間: {elapsed:.2f}秒")
        print(f"  処理速度: {fps_actual:.2f}fps")
        print(f"  出力: {output_video_path}")

        # 1分の動画を10分以内で処理できるか推定
        estimated_full_time = duration / fps_actual if fps_actual > 0 else float('inf')
        print(f"\n推定（1分動画）: {estimated_full_time:.2f}秒")

        if estimated_full_time <= 600:
            print("[PASS] 処理速度: 合格（10分以内）")
        else:
            print("[FAIL] 処理速度: 不合格（10分超過）")

        # 音声確認
        ffprobe_path = shutil.which("ffprobe") or "ffprobe"
        audio_check_cmd = [
            ffprobe_path, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            str(output_video_path)
        ]
        audio_result = subprocess.run(audio_check_cmd, capture_output=True, text=True)
        if audio_result.stdout.strip():
            print(f"[PASS] 音声保持: {audio_result.stdout.strip()}")
        else:
            print("[FAIL] 音声保持: 音声ストリームなし")

        # 一時ファイル削除
        temp_video_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"\n動画処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print(f"\n結果は {output_dir} を確認してください")

    return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
