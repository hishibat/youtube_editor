import { useState, useRef, useCallback, useEffect } from 'react';
import {
  uploadVideo,
  getBackgrounds,
  generatePreview,
  processVideo,
  processVideoFast,
  processVideoRobust,
  getProgress,
  getRawFrame,
  detectHighlights,
  restructureVideo,
} from './api';
import type { UploadResponse, Background, ProgressResponse, Stroke, Highlight } from './api';

type Step = 'upload' | 'edit' | 'points' | 'processing' | 'complete';

// クリックポイント型
interface ClickPoint {
  x: number;  // パーセンテージ (0-100)
  y: number;
  type: 'foreground' | 'background';
}

function App() {
  const [step, setStep] = useState<Step>('upload');
  const [videoData, setVideoData] = useState<UploadResponse | null>(null);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [selectedBg, setSelectedBg] = useState<string | null>(null);
  const [customBgFile, setCustomBgFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frameTime, setFrameTime] = useState(0);
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  // クリックポイント
  const [points, setPoints] = useState<ClickPoint[]>([]);
  const [firstFrameUrl, setFirstFrameUrl] = useState<string | null>(null);

  // ハイライト検出
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [isDetectingHighlights, setIsDetectingHighlights] = useState(false);

  // 処理モード: 'standard' | 'fast' | 'robust'
  const [processingMode, setProcessingMode] = useState<'standard' | 'fast' | 'robust'>('fast');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const bgFileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // 進捗ポーリング
  useEffect(() => {
    if (!taskId || step !== 'processing') return;

    const pollProgress = async () => {
      try {
        const prog = await getProgress(taskId);
        setProgress(prog);

        if (prog.status === '完了') {
          setStep('complete');
          setTaskId(null);
        } else if (prog.status.startsWith('エラー')) {
          setError(prog.status);
          setStep('edit');
          setTaskId(null);
        }
      } catch (err) {
        console.error('Progress poll error:', err);
      }
    };

    const interval = setInterval(pollProgress, 500);
    return () => clearInterval(interval);
  }, [taskId, step]);

  // キャンバスにポイントを描画
  const redrawCanvas = useCallback(() => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;

    // 画像を描画
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // ポイントを描画
    points.forEach((point, index) => {
      const x = (point.x / 100) * canvas.width;
      const y = (point.y / 100) * canvas.height;

      // 外側の円（白い縁取り）
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();

      // 内側の円（色付き）
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fillStyle = point.type === 'foreground'
        ? 'rgba(34, 197, 94, 0.9)'  // 緑
        : 'rgba(239, 68, 68, 0.9)'; // 赤
      ctx.fill();

      // 番号
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((index + 1).toString(), x, y);
    });
  }, [points]);

  useEffect(() => {
    if (step === 'points') {
      redrawCanvas();
    }
  }, [step, points, redrawCanvas]);

  const handleVideoUpload = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await uploadVideo(file);
      setVideoData(response);

      const bgResponse = await getBackgrounds();
      setBackgrounds(bgResponse.backgrounds);

      // 生フレームを取得
      const rawFrameResponse = await getRawFrame(response.file_id, 0);
      setFirstFrameUrl(rawFrameResponse.frame_url + '?t=' + Date.now());

      setStep('edit');
    } catch (err) {
      setError('動画のアップロードに失敗しました');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleVideoUpload(file);
    }
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('video/')) {
        handleVideoUpload(file);
      }
    },
    [handleVideoUpload]
  );

  // キャンバスクリック処理
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = ((e.clientX - rect.left) * scaleX / canvas.width) * 100;
    const y = ((e.clientY - rect.top) * scaleY / canvas.height) * 100;

    // 左クリック = 前景、右クリック = 背景
    const type = e.button === 2 ? 'background' : 'foreground';

    setPoints(prev => [...prev, { x, y, type }]);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    handleCanvasClick(e);
  };

  // ポイントをストローク形式に変換（API互換）
  const pointsToStrokes = (): Stroke[] => {
    const fgPoints = points.filter(p => p.type === 'foreground');
    const bgPoints = points.filter(p => p.type === 'background');

    const strokes: Stroke[] = [];

    if (fgPoints.length > 0) {
      strokes.push({
        points: fgPoints.map(p => ({ x: p.x, y: p.y })),
        type: 'foreground'
      });
    }

    if (bgPoints.length > 0) {
      strokes.push({
        points: bgPoints.map(p => ({ x: p.x, y: p.y })),
        type: 'background'
      });
    }

    return strokes;
  };

  const handlePreview = async () => {
    if (!videoData) return;
    setIsLoading(true);
    setError(null);
    try {
      const strokes = pointsToStrokes();
      const response = await generatePreview(
        videoData.file_id,
        selectedBg || undefined,
        customBgFile || undefined,
        frameTime,
        strokes.length > 0 ? strokes : undefined
      );
      setPreviewUrl(response.preview_url + '?t=' + Date.now());
    } catch (err) {
      setError('プレビューの生成に失敗しました');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProcess = async () => {
    if (!videoData) return;
    setError(null);
    setProgress({ current: 0, total: 0, status: '開始中', percent: 0 });
    setStep('processing');

    try {
      const strokes = pointsToStrokes();

      // モードに応じてAPIを選択
      let response;
      if (processingMode === 'robust') {
        // 堅牢モード: SAM 2 VideoPredictor + 時間的一貫性
        response = await processVideoRobust(
          videoData.file_id,
          selectedBg || undefined,
          customBgFile || undefined,
          strokes.length > 0 ? strokes : undefined,
          undefined  // keyframes (現在は未実装)
        );
      } else if (processingMode === 'fast') {
        // 高速モード: SAM 2 + RVM + 音声保持
        response = await processVideoFast(
          videoData.file_id,
          selectedBg || undefined,
          customBgFile || undefined,
          strokes.length > 0 ? strokes : undefined
        );
      } else {
        // 標準モード
        response = await processVideo(
          videoData.file_id,
          selectedBg || undefined,
          customBgFile || undefined,
          strokes.length > 0 ? strokes : undefined
        );
      }

      setTaskId(response.task_id);
      setOutputUrl(response.output_url);
      setDownloadUrl(response.download_url);
    } catch (err) {
      setError('動画処理の開始に失敗しました');
      setStep('edit');
      console.error(err);
    }
  };

  const handleDetectHighlights = async () => {
    if (!videoData) return;
    setIsDetectingHighlights(true);
    setError(null);

    try {
      const response = await detectHighlights(videoData.file_id);
      setHighlights(response.highlights);
    } catch (err) {
      setError('ハイライト検出に失敗しました');
      console.error(err);
    } finally {
      setIsDetectingHighlights(false);
    }
  };

  const handleRestructureVideo = async (highlight: Highlight) => {
    if (!videoData) return;
    setError(null);
    setProgress({ current: 0, total: 0, status: 'フック構造に再構成中...', percent: 50 });
    setStep('processing');

    try {
      const response = await restructureVideo(
        videoData.file_id,
        highlight.start_time,
        highlight.end_time
      );
      setOutputUrl(response.output_url);
      setDownloadUrl(response.download_url);
      setStep('complete');
    } catch (err) {
      setError('動画再構成に失敗しました');
      setStep('edit');
      console.error(err);
    }
  };

  const handleCustomBgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomBgFile(file);
      setSelectedBg(null);
    }
  };

  const handleBgSelect = (bgId: string) => {
    setSelectedBg(bgId);
    setCustomBgFile(null);
  };

  const undoLastPoint = () => {
    setPoints(prev => prev.slice(0, -1));
  };

  const clearPoints = () => {
    setPoints([]);
  };

  const resetAll = () => {
    setStep('upload');
    setVideoData(null);
    setBackgrounds([]);
    setSelectedBg(null);
    setCustomBgFile(null);
    setPreviewUrl(null);
    setOutputUrl(null);
    setDownloadUrl(null);
    setError(null);
    setFrameTime(0);
    setProgress(null);
    setTaskId(null);
    setPoints([]);
    setFirstFrameUrl(null);
    setHighlights([]);
  };

  const groupedBackgrounds = backgrounds.reduce(
    (acc, bg) => {
      if (!acc[bg.category]) {
        acc[bg.category] = [];
      }
      acc[bg.category].push(bg);
      return acc;
    },
    {} as Record<string, Background[]>
  );

  const fgPointCount = points.filter(p => p.type === 'foreground').length;
  const bgPointCount = points.filter(p => p.type === 'background').length;

  // ETA表示
  const formatEta = (seconds: number) => {
    if (seconds <= 0) return '';
    if (seconds < 60) return `残り約${Math.ceil(seconds)}秒`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.ceil(seconds % 60);
    return `残り約${mins}分${secs}秒`;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold text-center">
          YouTube Shorts Editor
        </h1>
        <p className="text-gray-400 text-center text-sm mt-1">
          プロ仕様の背景除去 + テロップ + ハイライト検出
        </p>
      </header>

      <main className="container mx-auto p-4 max-w-6xl">
        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {/* Step 1: Upload */}
        {step === 'upload' && (
          <div
            className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
            />
            {isLoading ? (
              <div className="flex flex-col items-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
                <p>アップロード中...</p>
              </div>
            ) : (
              <>
                <svg
                  className="w-16 h-16 mx-auto text-gray-500 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                <p className="text-xl mb-2">動画をドラッグ&ドロップ</p>
                <p className="text-gray-400">
                  または クリックしてファイルを選択
                </p>
                <p className="text-gray-500 text-sm mt-4">
                  対応形式: MP4, WebM, MOV, AVI
                </p>
              </>
            )}
          </div>
        )}

        {/* Step 2: Edit */}
        {step === 'edit' && videoData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Preview */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">プレビュー</h2>
              <div className="bg-gray-800 rounded-lg overflow-hidden aspect-[9/16] max-h-[500px] flex items-center justify-center">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : firstFrameUrl ? (
                  <img
                    src={firstFrameUrl}
                    alt="First frame"
                    className="max-w-full max-h-full object-contain opacity-70"
                  />
                ) : (
                  <p className="text-gray-500">
                    背景を選択してプレビューを生成
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <label className="block text-sm text-gray-400">
                  フレーム位置: {frameTime.toFixed(1)}秒
                </label>
                <input
                  type="range"
                  min={0}
                  max={videoData.info.duration}
                  step={0.1}
                  value={frameTime}
                  onChange={(e) => setFrameTime(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* ポイント指定ボタン */}
              <button
                onClick={() => setStep('points')}
                className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-3 rounded font-medium transition-colors"
              >
                🎯 切り抜き範囲を指定（3〜5点クリック）
              </button>

              {points.length > 0 && (
                <div className="bg-gray-800 rounded-lg p-3">
                  <p className="text-sm text-gray-400">
                    指定済み: {fgPointCount} 点（残す）/ {bgPointCount} 点（消す）
                  </p>
                </div>
              )}

              {/* 処理モード選択 */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">処理モード</h3>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setProcessingMode('fast')}
                    className={`px-3 py-2 rounded text-sm transition-colors ${
                      processingMode === 'fast'
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    ⚡ 高速
                    <span className="block text-xs opacity-70">SAM 2 + RVM</span>
                  </button>
                  <button
                    onClick={() => setProcessingMode('robust')}
                    className={`px-3 py-2 rounded text-sm transition-colors ${
                      processingMode === 'robust'
                        ? 'bg-orange-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    🎸 堅牢
                    <span className="block text-xs opacity-70">時間的一貫性</span>
                  </button>
                  <button
                    onClick={() => setProcessingMode('standard')}
                    className={`px-3 py-2 rounded text-sm transition-colors ${
                      processingMode === 'standard'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    🎯 高精度
                    <span className="block text-xs opacity-70">ProMatting</span>
                  </button>
                </div>
                {processingMode === 'robust' && (
                  <p className="mt-2 text-xs text-orange-300">
                    ギター等の細いパーツを持つ被写体向け。SAM 2 VideoPredictorによる時間的伝播で安定したマスクを生成します。
                  </p>
                )}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handlePreview}
                  disabled={isLoading}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
                >
                  {isLoading ? 'プレビュー生成中...' : 'プレビュー生成'}
                </button>
                <button
                  onClick={handleProcess}
                  disabled={isLoading}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
                >
                  動画を処理
                </button>
              </div>
              {!selectedBg && !customBgFile && (
                <p className="text-yellow-400 text-xs text-center">
                  💡 背景未選択の場合はグリーンバックで処理されます
                </p>
              )}

              {/* ハイライト検出 */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">🎸 ハイライト検出</h3>
                <button
                  onClick={handleDetectHighlights}
                  disabled={isDetectingHighlights}
                  className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-4 py-2 rounded font-medium transition-colors mb-2"
                >
                  {isDetectingHighlights ? '検出中...' : '🔍 ハイライトを自動検出'}
                </button>
                {highlights.length > 0 && (
                  <div className="space-y-2 mt-2">
                    <p className="text-xs text-gray-400">{highlights.length}件のハイライトを検出</p>
                    {highlights.map((h, i) => (
                      <div key={i} className="bg-gray-700 rounded p-2 flex justify-between items-center">
                        <div>
                          <span className="text-sm">
                            {h.start_time.toFixed(1)}s - {h.end_time.toFixed(1)}s
                          </span>
                          <span className="text-xs text-gray-400 ml-2">
                            ({h.type}, スコア: {(h.score * 100).toFixed(0)}%)
                          </span>
                        </div>
                        <button
                          onClick={() => handleRestructureVideo(h)}
                          className="bg-orange-600 hover:bg-orange-700 px-2 py-1 rounded text-xs"
                        >
                          冒頭に配置
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Background Selection */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">背景を選択</h2>

              {/* Custom Upload */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">
                  カスタム背景をアップロード
                </h3>
                <input
                  ref={bgFileInputRef}
                  type="file"
                  accept="image/*,video/*"
                  onChange={handleCustomBgChange}
                  className="hidden"
                />
                <button
                  onClick={() => bgFileInputRef.current?.click()}
                  className={`w-full border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                    customBgFile
                      ? 'border-blue-500 bg-blue-900/20'
                      : 'border-gray-600 hover:border-gray-500'
                  }`}
                >
                  {customBgFile ? (
                    <span className="text-blue-400">{customBgFile.name}</span>
                  ) : (
                    <span className="text-gray-400">
                      画像または動画を選択
                    </span>
                  )}
                </button>
              </div>

              {/* Preset Backgrounds */}
              {Object.entries(groupedBackgrounds).map(([category, bgs]) => (
                <div key={category} className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-3">
                    {category}
                  </h3>
                  <div className="grid grid-cols-3 gap-2">
                    {bgs.map((bg) => (
                      <button
                        key={bg.id}
                        onClick={() => handleBgSelect(bg.id)}
                        className={`relative aspect-[9/16] rounded overflow-hidden border-2 transition-all ${
                          selectedBg === bg.id
                            ? 'border-blue-500 ring-2 ring-blue-500/50'
                            : 'border-transparent hover:border-gray-500'
                        }`}
                      >
                        {bg.type === 'image' ? (
                          <img
                            src={bg.path}
                            alt={bg.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <video
                            src={bg.path}
                            className="w-full h-full object-cover"
                            muted
                            loop
                            onMouseEnter={(e) => e.currentTarget.play()}
                            onMouseLeave={(e) => {
                              e.currentTarget.pause();
                              e.currentTarget.currentTime = 0;
                            }}
                          />
                        )}
                        <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5">
                          <p className="text-xs truncate">{bg.name}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}

              {backgrounds.length === 0 && (
                <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500">
                  <p>プリセット背景がありません</p>
                  <p className="text-sm mt-1">
                    カスタム背景をアップロードしてください
                  </p>
                </div>
              )}

              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">
                  動画情報
                </h3>
                <div className="text-sm space-y-1">
                  <p>
                    ファイル名:{' '}
                    <span className="text-gray-300">
                      {videoData.original_name}
                    </span>
                  </p>
                  <p>
                    解像度:{' '}
                    <span className="text-gray-300">
                      {videoData.info.width} x {videoData.info.height}
                    </span>
                  </p>
                  <p>
                    長さ:{' '}
                    <span className="text-gray-300">
                      {videoData.info.duration.toFixed(1)}秒
                    </span>
                  </p>
                  <p>
                    FPS:{' '}
                    <span className="text-gray-300">
                      {videoData.info.fps.toFixed(0)}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step: Point Selection */}
        {step === 'points' && firstFrameUrl && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">🎯 クリックで指定</h2>
              <button
                onClick={() => setStep('edit')}
                className="text-gray-400 hover:text-white"
              >
                ← 戻る
              </button>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center text-xs font-bold">+</div>
                  <span className="text-sm">🟢 左クリック: 残す部分（緑）</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center text-xs font-bold">−</div>
                  <span className="text-sm">🔴 右クリック: 消す部分（赤）</span>
                </div>
              </div>

              {/* 詳細インストラクション */}
              <div className="bg-blue-900/30 border border-blue-600/50 rounded-lg p-3 mb-3">
                <p className="text-blue-200 text-sm font-medium mb-2">📌 精度を上げるコツ</p>
                <ul className="text-blue-200 text-xs space-y-1 list-disc list-inside">
                  <li><strong>「核をなぞる」</strong>: 物体の中心部分をクリックすると精度が上がります</li>
                  <li><strong>複数ポイント</strong>: 頭・胴体・手など複数箇所をクリックすると安定します</li>
                  <li><strong>境界は避ける</strong>: 輪郭ギリギリよりも内側をクリックしてください</li>
                </ul>
              </div>

              <div className="bg-green-900/30 border border-green-600/50 rounded-lg p-3 mb-4">
                <p className="text-green-200 text-sm">
                  💡 <strong>推奨:</strong>
                  「頭」「ギターのヘッド」「ギターのボディ」の3点を左クリック、
                  「椅子」「壁」を右クリックするだけでOK！
                </p>
              </div>

              <div className="bg-yellow-900/30 border border-yellow-600/50 rounded-lg p-2">
                <p className="text-yellow-200 text-xs">
                  ⚡ <strong>高速モード:</strong> SAM 2 + RVM で1分動画を約2-3分で処理（音声保持）
                </p>
              </div>
            </div>

            <div className="flex justify-center">
              <div className="relative inline-block">
                <img
                  ref={imageRef}
                  src={firstFrameUrl}
                  alt="First frame"
                  className="hidden"
                  onLoad={() => {
                    if (canvasRef.current && imageRef.current) {
                      const canvas = canvasRef.current;
                      const img = imageRef.current;
                      canvas.width = img.naturalWidth;
                      canvas.height = img.naturalHeight;
                      redrawCanvas();
                    }
                  }}
                />
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  onContextMenu={handleContextMenu}
                  className="max-h-[500px] cursor-crosshair rounded-lg"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              </div>
            </div>

            <div className="flex justify-center gap-4">
              <button
                onClick={undoLastPoint}
                disabled={points.length === 0}
                className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-700 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
              >
                ↩ 1つ戻す
              </button>
              <button
                onClick={clearPoints}
                disabled={points.length === 0}
                className="bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
              >
                🗑 全てクリア
              </button>
              <button
                onClick={() => setStep('edit')}
                className="bg-green-600 hover:bg-green-700 px-6 py-2 rounded font-medium transition-colors"
              >
                ✓ 確定して戻る
              </button>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-2">指定済みポイント</h3>
              <div className="flex flex-wrap gap-2">
                {points.map((point, index) => (
                  <span
                    key={index}
                    className={`px-3 py-1 rounded-full text-sm ${
                      point.type === 'foreground'
                        ? 'bg-green-900/50 text-green-300'
                        : 'bg-red-900/50 text-red-300'
                    }`}
                  >
                    #{index + 1} {point.type === 'foreground' ? '残す' : '消す'}
                  </span>
                ))}
                {points.length === 0 && (
                  <span className="text-gray-500 text-sm">まだポイントが指定されていません</span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Processing with Progress */}
        {step === 'processing' && (
          <div className="max-w-xl mx-auto text-center py-12">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-6"></div>
            <h2 className="text-xl font-semibold mb-4">動画を処理中...</h2>

            {/* Progress Bar */}
            <div className="bg-gray-800 rounded-lg p-6 mb-4">
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-gray-400">
                  {progress?.status || '開始中'}
                </span>
                <span className="text-blue-400">
                  {progress?.percent || 0}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                <div
                  className="bg-blue-500 h-4 rounded-full transition-all duration-300"
                  style={{ width: `${progress?.percent || 0}%` }}
                ></div>
              </div>
              {progress && progress.total > 0 && (
                <p className="text-gray-500 text-sm mt-2">
                  {progress.current} / {progress.total} フレーム
                </p>
              )}
            </div>

            <p className="text-gray-400 text-sm">
              {points.length > 0
                ? `🎯 ポイント指定モード: ${fgPointCount}点（残す）+ ${bgPointCount}点（消す）`
                : '🤖 自動検出モード: AIが前景を自動判定'
              }
            </p>
            <p className="text-gray-500 text-xs mt-2">
              処理中は音声も保持されます
            </p>
          </div>
        )}

        {/* Step 4: Complete */}
        {step === 'complete' && outputUrl && (
          <div className="max-w-2xl mx-auto space-y-6">
            <h2 className="text-xl font-semibold text-center">🎉 処理完了!</h2>

            <div className="bg-gray-800 rounded-lg overflow-hidden">
              <video
                src={outputUrl}
                controls
                className="w-full max-h-[500px]"
              />
            </div>

            <div className="flex gap-4 justify-center">
              <a
                href={downloadUrl || '#'}
                download
                className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded font-medium transition-colors"
              >
                📥 ダウンロード
              </a>
              <button
                onClick={resetAll}
                className="bg-gray-600 hover:bg-gray-700 px-6 py-3 rounded font-medium transition-colors"
              >
                新しい動画を編集
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
