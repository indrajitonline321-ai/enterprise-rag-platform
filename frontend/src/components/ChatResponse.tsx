// src/components/ChatResponse.tsx
import React, { useEffect, useRef } from 'react';
import type { ChatResponseAPI, SourceChunk } from '../types';

interface HistoryItem {
  query: string;
  response: ChatResponseAPI;
}

interface ChatResponseProps {
  history: HistoryItem[];
}

const ChatResponse: React.FC<ChatResponseProps> = ({ history }) => {
 const scrollEndRef = useRef<HTMLDivElement>(null);

  // 2. Setup an effect that runs whenever 'history' changes
  useEffect(() => {
    if (scrollEndRef.current) {
      scrollEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [history]); // This triggers every time a new message is added

  if (!history.length) {
    return <div className="chat-response">Ask a question to get started.</div>;
  }


const handleDownload = async (fileName: string) => {
  try {
    const response = await fetch(`http://localhost:8080/api/files/download/${(getFileName(fileName))}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = (getFileName(fileName) as string);
    a.click();

    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download failed', err);
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
            <strong>&gt;</strong> {item.query}
          </div>

          <div className="answer">
            {item.response.answer}
          </div>

          <div className="sources-title">📚 Sources:</div>
          {item.response.sources.slice(0, 2).map((chunk: SourceChunk, j: number) => (
            <div key={j} className="source">
              <div className="source-header">
                📄 <strong>
            <a
              href="#"
              onClick={(e) => {
                   e.preventDefault();
      handleDownload(chunk.file_name);   // pass full name here
               }}
              >
             {getFileName(chunk.file_name)}
          </a>
          </strong>
                <span className="source-meta">
                  Page {chunk.page} • Score {chunk.score.toFixed(2)}
                </span>
              </div>
              <div className="source-content">
                {chunk.content.substring(0, 500)}...
              </div>
            </div>
          ))}
        </div>
      ))}
        <div ref={scrollEndRef} />
    </div>
  );
};

export default ChatResponse;
