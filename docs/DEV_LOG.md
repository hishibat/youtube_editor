# YouTube Shorts Editor 開発ログ

## 2026-01-10 セッション

### 目標
ギター演奏動画の背景除去において、時間的一貫性のあるセグメンテーションを実現する

### 現状の課題
- ギターのネック・ヘッド部分が背景として誤認識される
- フレーム間でマスクが不安定（フラフラする）
- 細いパーツ（ギターの弦、ネック）が消えてしまう

### 実施した対策

#### 1. cv2.ximgproc エラーの修正
- **問題**: `cv2.ximgproc.guidedFilter` が AttributeError
- **原因**: opencv-python-headless に ximgproc モジュールがない
- **解決**:
  - `guided_filter_fallback()` 関数を追加（純粋cv2.blur実装）
  - `try_guided_filter()` ラッパーで自動フォールバック
  - requirements.txt を opencv-contrib-python-headless に更新

#### 2. SAM 2 VideoPredictor 実装
- **ファイル**: `backend/services/robust_video_segmentation.py`
- **機能**:
  - `propagate_in_video()` による時間的マスク伝播
  - マルチキーフレームアノテーション対応
  - FP16高速推論

#### 3. マスク後処理パイプライン
- **MaskPostProcessor クラス**:
  - Morphological Closing: 小さな穴を埋める
  - Dilation: 細いパーツのエッジを保護
  - Connected Component Analysis: 主要物体に接続された部分を保護
  - Temporal Smoothing: 加重移動平均でフリッカー除去

#### 4. フロントエンド UI 改善
- 3つの処理モードを追加:
  - **高速**: SAM 2 + RVM（デフォルト）
  - **堅牢**: SAM 2 VideoPredictor + 時間的一貫性（ギター向け）
  - **高精度**: ProMatting（時間がかかる）
- ポイント選択のガイダンス追加

### 追加したファイル
```
backend/services/
├── robust_video_segmentation.py  # SAM 2 VideoPredictor + 後処理
├── hybrid_matting_service.py     # SAM 2 + RVM ハイブリッド
├── pro_matting_service.py        # 高精度マッティング
├── subtitle_service.py           # 字幕オーバーレイ
├── highlight_service.py          # ハイライト検出
└── fast_video_processor.py       # 高速フレーム処理
```

### APIエンドポイント
| エンドポイント | 用途 |
|---------------|------|
| POST /api/process-fast | 高速処理（SAM 2 + RVM） |
| POST /api/process-robust | 堅牢処理（VideoPredictor） |
| POST /api/process | 標準処理（ProMatting） |

### 未解決の課題

1. **セグメンテーション精度**
   - ギターネックが依然として不安定
   - 背景色に近い物体の分離が困難

2. **テスト不足**
   - robust モードの実際の動画でのテストが未完了
   - 処理時間のベンチマークが必要

3. **マルチキーフレーム UI**
   - フロントエンドで複数フレームにアノテーションを追加するUIが未実装
   - 現在は keyframes パラメータが undefined で渡される

### 次回のアクション案

1. 実際のギター動画で robust モードをテスト
2. ポイント選択の位置・数による精度変化を検証
3. Morphological パラメータのチューニング
4. 必要に応じてマルチキーフレーム UI を実装

### 技術スタック
- Backend: FastAPI + SAM 2 + RVM + OpenCV
- Frontend: React + TypeScript + Vite + Tailwind CSS
- 推論: PyTorch (CUDA FP16)

### Git コミット
```
6565ea2 feat: Add temporal consistency and multiple processing modes
```

---
*最終更新: 2026-01-10*
