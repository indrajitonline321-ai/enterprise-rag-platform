import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import FileList from './components/FileList';
import ChatInput from './components/ChatInput';
import ChatResponse from './components/ChatResponse';
import type { ChatResponseAPI, ChatMessage, FileInfo } from './types';
import './App.css';

interface HistoryItem {
  query: string;
  response: ChatResponseAPI;
}

function App() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const handleSend = (query: string, response: ChatResponseAPI) => {
    setHistory(prev => [...prev, { query, response }]); // append, do not replace
  };

  return (
    <div className="app">
      <div className="left-panel">
        <h2>📁 Documents</h2>
        <FileUpload setFiles={setFiles} />
        <FileList files={files} />
      </div>

      <div className="right-panel">
        <h2>💬 RAG Chat</h2>
        <ChatResponse history={history} />
        <ChatInput onSend={handleSend} />
      </div>
    </div>
  );
}

export default App;
