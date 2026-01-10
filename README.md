# YouTube Shorts Editor

動画の背景を自由に差し替えられるWebアプリケーション。
AI背景除去（rembg）を使用して、人物を切り抜き、好きな背景と合成できます。

## 機能

- 動画アップロード（MP4, WebM, MOV, AVI対応）
- AI背景除去（rembgによる自動背景除去）
- 背景選択（プリセット or カスタムアップロード）
- リアルタイムプレビュー
- 動画処理・ダウンロード

## 必要環境

- Python 3.10+
- Node.js 20.19+ または 22.12+

## セットアップ

### 1. Backend

```bash
cd backend

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# サーバー起動
python main.py
```

バックエンドは http://localhost:8000 で起動します。

### 2. Frontend

```bash
cd frontend

# 依存関係インストール
npm install

# 開発サーバー起動
npm run dev
```

フロントエンドは http://localhost:5173 で起動します。

## 使い方

1. http://localhost:5173 にアクセス
2. 動画ファイルをドラッグ&ドロップ（またはクリックして選択）
3. 背景を選択（カスタム画像をアップロードも可能）
4. 「プレビュー生成」で1フレームのプレビューを確認
5. 「動画を処理」で全フレームを処理
6. 完了後、ダウンロードボタンで保存

## 背景素材の追加

`backend/backgrounds/` 以下にカテゴリ別フォルダを作成し、画像/動画を配置：

```
backend/backgrounds/
├── funny/        # ネタ系
├── stylish/      # おしゃれ系
└── realistic/    # 実写系
```

対応形式: `.jpg`, `.png`, `.mp4`, `.webm`, `.gif`

## プロジェクト構成

```
youtube_editor/
├── backend/
│   ├── main.py              # FastAPI エントリポイント
│   ├── requirements.txt     # Python依存関係
│   ├── services/
│   │   ├── background_removal.py  # rembg背景除去
│   │   └── video_processor.py     # OpenCV動画処理
│   └── backgrounds/         # 背景素材フォルダ
└── frontend/
    ├── src/
    │   ├── App.tsx          # メインUI
    │   └── api.ts           # APIクライアント
    └── package.json
```

## API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/upload` | 動画アップロード |
| GET | `/api/backgrounds` | 背景一覧取得 |
| POST | `/api/preview` | プレビュー生成 |
| POST | `/api/process` | 動画処理 |
| GET | `/api/download/{id}` | 動画ダウンロード |

## 注意事項

- 初回のrembgモデルダウンロードに時間がかかります（約170MB）
- 動画処理は長さに比例して時間がかかります
- メモリ使用量が多いため、長い動画や高解像度動画は注意
