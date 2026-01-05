// FileList.tsx
import React from 'react';
import type { FileInfo } from '../types';

interface FileListProps {
  files: FileInfo[];
}

const FileList: React.FC<FileListProps> = ({ files }) => (
  <div className="file-list">
    {files.map((file: FileInfo, index: number) => (
      <div key={index} className="file-item">
        📄 {file.name}
        <br />
        <small>
          ID: {file.documentId} • 
          {file.chunks ? `${file.chunks} chunks` : 'Indexing...'}
        </small>
      </div>
    ))}
  </div>
);


export default FileList;
