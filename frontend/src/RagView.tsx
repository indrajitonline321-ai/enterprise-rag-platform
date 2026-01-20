import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import FileList from './components/FileList';
import ChatInput from './components/ChatInput';
import ChatResponse from './components/ChatResponse';
import type { ChatResponseAPI, FileInfo } from './types';

interface HistoryItem {
  query: string;
  response: ChatResponseAPI;
}

interface RagViewProps {
  onLogout: () => void;
}

const RagView: React.FC<RagViewProps> = ({ onLogout }) => {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showFilePanel, setShowFilePanel] = useState(false);
  

  const handleSend = (query: string, response: ChatResponseAPI) => {
    setHistory(prev => [...prev, { query, response }]);
  };

  return (
    <div className="rag-dashboard">
      {/* FULL SCREEN CHAT */}
      <div className="main-chat">
        <div className="header-bar">
          <h2>💬 RAG Chat</h2>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              className="file-toggle-btn"
              onClick={() => setShowFilePanel(!showFilePanel)}
            >
              📁 {showFilePanel ? 'Hide' : 'Files'}
            </button>
            <button className="logout-btn" onClick={onLogout}>
              Logout
            </button>
          </div>
        </div>

       <div className={`chat-messages ${showFilePanel ? 'shrink-chat' : ''}`}>
          <ChatResponse history={history} />
        </div>

        <div className={`chat-input ${showFilePanel ? 'shrink' : ''}`}>
          <div className="chat-input-container">
            <ChatInput onSend={handleSend} lastQuery={history.length > 0 ? history[history.length - 1].query : undefined} />
          </div>
        </div>
      </div>

      {/* SLIDE-DOWN FILE PANEL - Right side */}
      <div className={`file-panel ${showFilePanel ? 'open' : ''}`}>
        <div className="file-panel-header">
          <h3>📁&nbsp;Upload&nbsp;Documents</h3>
          <button 
            className="panel-close-btn"
            onClick={() => setShowFilePanel(false)}
          >
            ×
          </button>
        </div>
        <div className="file-panel-content">
          <div className="upload-section">
            <FileUpload setFiles={setFiles} />
          </div>
          <div className="file-list">
            <FileList files={files} />
          </div>
        </div>
      </div>
    </div>
  );
};


export default RagView;
