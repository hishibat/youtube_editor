import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration: number;
}

export interface UploadResponse {
  file_id: string;
  filename: string;
  original_name: string;
  path: string;
  info: VideoInfo;
}

export interface Background {
  id: string;
  name: string;
  category: string;
  category_id: string;
  path: string;
  type: 'video' | 'image';
}

export interface BackgroundsResponse {
  backgrounds: Background[];
}

export interface PreviewResponse {
  preview_id: string;
  preview_url: string;
}

export interface ProcessResponse {
  task_id: string;
  output_id: string;
  output_url: string;
  download_url: string;
}

export interface ProgressResponse {
  current: number;
  total: number;
  status: string;
  percent: number;
}

// SAM 2用プロンプトポイント
export interface PromptPoint {
  x: number;  // パーセンテージ (0-100)
  y: number;  // パーセンテージ (0-100)
  type: 'foreground' | 'background';
}

// ストローク（線）- 複数のポイントで構成
export interface Stroke {
  points: { x: number; y: number }[];  // パーセンテージ座標の配列
  type: 'foreground' | 'background';
}

// セグメンテーション入力（ポイントまたはストローク）
export interface SegmentationInput {
  strokes: Stroke[];
}

export interface RawFrameResponse {
  frame_url: string;
}

export const uploadVideo = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post<UploadResponse>('/upload', formData);
  return response.data;
};

export const getRawFrame = async (fileId: string, time: number = 0): Promise<RawFrameResponse> => {
  const response = await api.get<RawFrameResponse>(`/frame/${fileId}?time=${time}`);
  return response.data;
};

export const getBackgrounds = async (): Promise<BackgroundsResponse> => {
  const response = await api.get<BackgroundsResponse>('/backgrounds');
  return response.data;
};

export const generatePreview = async (
  fileId: string,
  backgroundId?: string,
  backgroundFile?: File,
  frameTime: number = 0,
  strokes?: Stroke[]
): Promise<PreviewResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);
  formData.append('frame_time', frameTime.toString());

  if (backgroundId) {
    formData.append('background_id', backgroundId);
  }
  if (backgroundFile) {
    formData.append('background_file', backgroundFile);
  }
  if (strokes && strokes.length > 0) {
    formData.append('strokes', JSON.stringify(strokes));
  }

  const response = await api.post<PreviewResponse>('/preview', formData);
  return response.data;
};

export const processVideo = async (
  fileId: string,
  backgroundId?: string,
  backgroundFile?: File,
  strokes?: Stroke[]
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);

  if (backgroundId) {
    formData.append('background_id', backgroundId);
  }
  if (backgroundFile) {
    formData.append('background_file', backgroundFile);
  }
  if (strokes && strokes.length > 0) {
    formData.append('strokes', JSON.stringify(strokes));
  }

  const response = await api.post<ProcessResponse>('/process', formData);
  return response.data;
};

export const getProgress = async (taskId: string): Promise<ProgressResponse> => {
  const response = await api.get<ProgressResponse>(`/progress/${taskId}`);
  return response.data;
};

// === ハイライト検出 ===

export interface Highlight {
  start_time: number;
  end_time: number;
  score: number;
  type: string;
}

export interface HighlightsResponse {
  highlights: Highlight[];
}

export const detectHighlights = async (fileId: string): Promise<HighlightsResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);
  const response = await api.post<HighlightsResponse>('/detect-highlights', formData);
  return response.data;
};

export const restructureVideo = async (
  fileId: string,
  hookStart: number,
  hookEnd: number
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);
  formData.append('hook_start', hookStart.toString());
  formData.append('hook_end', hookEnd.toString());
  const response = await api.post<ProcessResponse>('/restructure-video', formData);
  return response.data;
};

// === 字幕 ===

export interface SubtitleEntry {
  start_time: number;
  end_time: number;
  text: string;
  style?: string;
}

export const addSubtitles = async (
  fileId: string,
  subtitles: SubtitleEntry[]
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);
  formData.append('subtitles_json', JSON.stringify(subtitles));
  const response = await api.post<ProcessResponse>('/add-subtitles', formData);
  return response.data;
};

// === 高速処理（SAM 2 + RVM + 音声保持） ===

export const processVideoFast = async (
  fileId: string,
  backgroundId?: string,
  backgroundFile?: File,
  strokes?: Stroke[]
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);

  if (backgroundId) {
    formData.append('background_id', backgroundId);
  }
  if (backgroundFile) {
    formData.append('background_file', backgroundFile);
  }
  if (strokes && strokes.length > 0) {
    formData.append('strokes', JSON.stringify(strokes));
  }

  const response = await api.post<ProcessResponse>('/process-fast', formData);
  return response.data;
};

// === 堅牢処理（SAM 2 VideoPredictor + 時間的一貫性） ===

export interface KeyframeAnnotation {
  frame_idx: number;
  foreground_points: Array<{ x: number; y: number }>;  // パーセンテージ座標
  background_points: Array<{ x: number; y: number }>;  // パーセンテージ座標
}

export const processVideoRobust = async (
  fileId: string,
  backgroundId?: string,
  backgroundFile?: File,
  strokes?: Stroke[],
  keyframes?: KeyframeAnnotation[]
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);

  if (backgroundId) {
    formData.append('background_id', backgroundId);
  }
  if (backgroundFile) {
    formData.append('background_file', backgroundFile);
  }
  if (strokes && strokes.length > 0) {
    formData.append('strokes', JSON.stringify(strokes));
  }
  if (keyframes && keyframes.length > 0) {
    formData.append('keyframes_json', JSON.stringify(keyframes));
  }

  const response = await api.post<ProcessResponse>('/process-robust', formData);
  return response.data;
};

export default api;
