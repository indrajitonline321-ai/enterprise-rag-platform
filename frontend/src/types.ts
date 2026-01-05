// src/types.ts

export interface SourceChunk {
  id: number;
  score: number;
  content: string;
  file_name: string;
  document_id: string;
  page: number;
  chunk_index: number;
}

export interface ChatResponseAPI {
  answer: string;
  sources: SourceChunk[];
}

export interface ChatMessage {
  query: string;
}

// For uploaded file list in left panel
export interface FileInfo {
  name: string;
  documentId: string;
  uploadedAt: string;
  chunks?: number;  // Number of indexed chunks
  blobUrl?: string;
}
