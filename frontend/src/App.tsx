import { useState, useRef, useCallback, useEffect } from 'react';
import {
  uploadVideo,
  getBackgrounds,
  generatePreview,
  processVideo,
  getProgress,
} from './api';
import type { UploadResponse, Background, ProgressResponse } from './api';

type Step = 'upload' | 'edit' | 'processing' | 'complete';

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

  const fileInputRef = useRef<HTMLInputElement>(null);
  const bgFileInputRef = useRef<HTMLInputElement>(null);

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

  const handleVideoUpload = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await uploadVideo(file);
      setVideoData(response);

      const bgResponse = await getBackgrounds();
      setBackgrounds(bgResponse.backgrounds);

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

  const handlePreview = async () => {
    if (!videoData) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await generatePreview(
        videoData.file_id,
        selectedBg || undefined,
        customBgFile || undefined,
        frameTime
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
      const response = await processVideo(
        videoData.file_id,
        selectedBg || undefined,
        customBgFile || undefined
      );
      setTaskId(response.task_id);
      setOutputUrl(response.output_url);
      setDownloadUrl(response.download_url);
    } catch (err) {
      setError('動画処理の開始に失敗しました');
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

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold text-center">
          YouTube Shorts Editor
        </h1>
        <p className="text-gray-400 text-center text-sm mt-1">
          背景を自由に差し替えて動画を作成
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

              <div className="flex gap-2">
                <button
                  onClick={handlePreview}
                  disabled={isLoading || (!selectedBg && !customBgFile)}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
                >
                  {isLoading ? 'プレビュー生成中...' : 'プレビュー生成'}
                </button>
                <button
                  onClick={handleProcess}
                  disabled={isLoading || (!selectedBg && !customBgFile)}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded font-medium transition-colors"
                >
                  動画を処理
                </button>
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
              高速処理モード: 軽量AI + フレームスキップ
            </p>
          </div>
        )}

        {/* Step 4: Complete */}
        {step === 'complete' && outputUrl && (
          <div className="max-w-2xl mx-auto space-y-6">
            <h2 className="text-xl font-semibold text-center">処理完了!</h2>

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
                ダウンロード
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
