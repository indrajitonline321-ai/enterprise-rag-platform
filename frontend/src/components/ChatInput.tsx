// src/components/ChatInput.tsx
import React, { useState } from 'react';
import type { ChatResponseAPI } from '../types';

interface ChatInputProps {
  onSend: (query: string, response: ChatResponseAPI) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const q = query.trim();
    setQuery('');

    try {
      const res = await fetch('http://localhost:8080/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, userId: "Employee" })
      });

      const  ChatResponseAPI = await res.json();
      onSend(q, ChatResponseAPI);
    } catch (err) {
      console.error('Chat error', err);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input">
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Ask about your documents..."
      />
      <button type="submit">Send</button>
    </form>
  );
};

export default ChatInput;
