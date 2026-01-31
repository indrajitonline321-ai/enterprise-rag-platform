// src/components/ChatInput.tsx
import React, { useState } from 'react';
import type { ChatResponseAPI } from '../types';

interface ChatInputProps {
  onSend: (query: string, response: ChatResponseAPI) => void;
  lastQuery?: string; 
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend,lastQuery }) => {
    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); 
      handleSubmit(e);    
    }
  };
  const [query, setQuery] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const q = query.trim();
    
    const currentQuery = query.trim();
    const combinedQuery = lastQuery 
      ? `${lastQuery} | ${currentQuery}` 
      : currentQuery;

    setQuery('');

    try {
      const res = await fetch('http://localhost:8080/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: combinedQuery, userId: localStorage.getItem('userName')!})
      });

      const  ChatResponseAPI = await res.json();
      onSend(q, ChatResponseAPI);
    } catch (err) {
      console.error('Chat error', err);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input">
       <textarea
        value={query}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about your documents..." className="inputTextArea"
      />
      <button type="submit"><span>↑</span></button>
    </form>
  );
};

export default ChatInput;
