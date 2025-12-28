import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [message, setMessage] = useState<string>("Ready to upload");
  const [loading, setLoading] = useState(false);

  // Load files on start
  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const res = await axios.get<string[]>("http://localhost:8080/api/files/list");
      setFiles(res.data);
    } catch (e) {
      console.log("Backend not running? Start with ./mvnw spring-boot:run");
    }
  };

  const uploadFile = async () => {
    if (!file) {
      setMessage("Select a file first");
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const res = await axios.post("http://localhost:8080/api/files/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      
      setMessage(`✅ ${res.data.name} uploaded! ${res.data.url}`);
      loadFiles(); // Refresh list
    } catch (e: any) {
      setMessage(`❌ Upload failed: ${e.response?.data?.error || e.message}`);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 40, maxWidth: 600, margin: 'auto' }}>
      <h1>🚀 Enterprise RAG - File Upload</h1>
      <p>Upload PDFs → They go to Azure Blob Storage → Ready for RAG processing</p>
      
      <div style={{ 
        border: '2px dashed #ccc', 
        padding: 30, 
        borderRadius: 10, 
        textAlign: 'center',
        marginBottom: 30 
      }}>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          style={{ marginBottom: 20 }}
        />
        <div>
          <button 
            onClick={uploadFile} 
            disabled={!file || loading}
            style={{
              padding: '12px 24px',
              fontSize: 16,
              background: loading ? '#ccc' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: 5,
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Uploading...' : `Upload ${file?.name || ''}`}
          </button>
          <button 
            onClick={loadFiles}
            style={{
              padding: '12px 24px',
              marginLeft: 10,
              fontSize: 16,
              background: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: 5
            }}
          >
            Refresh List
          </button>
        </div>
      </div>

      <div style={{ 
        background: '#f8f9fa', 
        padding: 20, 
        borderRadius: 10,
        minHeight: 100 
      }}>
        <h3>📁 Files in Azure Storage:</h3>
        <div style={{ color: '#6c757d', fontSize: 14 }}>
          {message}
        </div>
        {files.length === 0 ? (
          <p style={{ color: '#999' }}>No files yet. Upload your first PDF!</p>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {files.map((f, i) => (
              <li key={i} style={{ 
                padding: 10, 
                borderBottom: '1px solid #eee',
                fontFamily: 'monospace'
              }}>
                📄 {f}
              </li>
            ))}
          </ul>
        )}
        <div style={{ marginTop: 20, fontSize: 12, color: '#666' }}>
          Backend: http://localhost:8080 | {files.length} files total
        </div>
      </div>
    </div>
  );
}

export default App;

