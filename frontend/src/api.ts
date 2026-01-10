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

export const uploadVideo = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post<UploadResponse>('/upload', formData);
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
  frameTime: number = 0
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

  const response = await api.post<PreviewResponse>('/preview', formData);
  return response.data;
};

export const processVideo = async (
  fileId: string,
  backgroundId?: string,
  backgroundFile?: File
): Promise<ProcessResponse> => {
  const formData = new FormData();
  formData.append('file_id', fileId);

  if (backgroundId) {
    formData.append('background_id', backgroundId);
  }
  if (backgroundFile) {
    formData.append('background_file', backgroundFile);
  }

  const response = await api.post<ProcessResponse>('/process', formData);
  return response.data;
};

export const getProgress = async (taskId: string): Promise<ProgressResponse> => {
  const response = await api.get<ProgressResponse>(`/progress/${taskId}`);
  return response.data;
};

export default api;
