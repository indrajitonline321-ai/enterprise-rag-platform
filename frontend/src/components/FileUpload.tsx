// FileUpload.tsx
import React, { useState } from 'react';
import type { FileInfo } from '../types';

interface FileUploadProps {
  setFiles: React.Dispatch<React.SetStateAction<FileInfo[]>>;
}

const FileUpload: React.FC<FileUploadProps> = ({ setFiles }) => {
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setStatus('Uploading to Azure...');
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('userID', "Manager");

    try {
      // 1. STEP 1: Upload to Spring Boot → Azure Blob
      const uploadRes = await fetch('http://localhost:8080/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      
      const uploadJson = await uploadRes.json();
      const documentId = uploadJson.name || 'unknown';
      const blobUrl = uploadJson.url;  // Spring Boot returns blob URL

      // const documentId = "9"
      //  const blobUrl = "https://enterpriseragstorage.blob.core.windows.net/documents/pieChart.pdf";  // Spring Boot returns blob URL
     const userId="Manager"
      // setStatus('✅ Uploaded! Indexing chunks...');

      // 2. STEP 2: Trigger Python /ingest
      const ingestRes = await fetch('http://localhost:8080/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: documentId,
          blob_url: blobUrl,
          user_id: userId, // Python downloads from Azure
        }),
      });

      const ingestJson = await ingestRes.json();
      
      // 3. Add to file list
      setFiles(prev => [
        ...prev,
        {
          name: file.name,
          documentId,
          uploadedAt: new Date().toLocaleString(),
          chunks: ingestJson.vectors_stored || 0,  // Bonus: chunk count
          blobUrl
        },
      ]);

      setStatus(`✅ Indexed ${ingestJson.vectors_stored || 0} chunks! Ready for chat.`);
      
    } catch (err) {
      console.error('Upload/ingest failed', err);
      setStatus('❌ Failed: ' + (err as Error).message);
    } finally {
      setUploading(false);
      // Clear status after 3s
      setTimeout(() => setStatus(''), 3000);
    }
  };

  return (
    <div className="upload-section">
      <input 
        type="file" 
        accept=".pdf,.docx,.xlsx,.xls,.pptx,.txt,.rtf" 
        onChange={handleUpload} 
        disabled={uploading}
      />
      {uploading && <p className="status">{status}</p>}
    </div>
  );
};

export default FileUpload;
