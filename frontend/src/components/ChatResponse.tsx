// src/components/ChatResponse.tsx
import React from 'react';
import type { ChatResponseAPI, SourceChunk } from '../types';

interface HistoryItem {
  query: string;
  response: ChatResponseAPI;
}

interface ChatResponseProps {
  history: HistoryItem[];
}

const ChatResponse: React.FC<ChatResponseProps> = ({ history }) => {
  if (!history.length) {
    return <div className="chat-response">Ask a question to get started.</div>;
  }

const handleDownload = async () => {
    try {
    const response = await fetch('https://api.example.com/log-click');
    const data = await response.json();
    console.log("Action logged:", data);
  } catch (err) {
    console.error("API call failed", err);
  }
};

const getFileName = (file_name: string) => {
    return file_name.split('/').pop();
  };

  return (
    <div className="chat-response">
      {history.map((item, i) => (
        <div key={i} className="message">
          <div>
            <strong>Q:</strong> {item.query}
          </div>

          <div className="answer">
            <strong>A:</strong> {item.response.answer}
          </div>

          <div className="sources-title">📚 Sources:</div>
          {item.response.sources.slice(0, 3).map((chunk: SourceChunk, j: number) => (
            <div key={j} className="source">
              <div className="source-header">
                📄 <strong><a href="#" onClick={handleDownload}>{getFileName(chunk.file_name)}</a></strong>
                <span className="source-meta">
                  Page {chunk.page} • Score {chunk.score.toFixed(3)}
                </span>
              </div>
              <div className="source-content">
                {chunk.content.substring(0, 140)}...
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default ChatResponse;
