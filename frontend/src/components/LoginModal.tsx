import React, { useState, useEffect } from 'react';

const LoginModal = ({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    setTimeout(() => {
      if ((username.toUpperCase === 'admin'.toUpperCase || username.toUpperCase === 'Manager'.toUpperCase 
        || username.toUpperCase === 'Employee'.toUpperCase
      ) && password === 'rag0419') {
        localStorage.setItem('userName', username);
        onSuccess(); // Close modal → RAG dashboard
      } else {
        setError('Invalid username or password');
      }
      setIsLoading(false);
    }, 800);
  };

  // ESC key closes modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  // Prevent scroll behind modal
  useEffect(() => {
    document.body.classList.add('no-scroll');
    return () => document.body.classList.remove('no-scroll');
  }, []);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>ENTERPRISE - RAG - PLATFORM</h2>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group1">
            <label>UserName</label>
            <input
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              disabled={isLoading}
              autoFocus
            />
          </div>

          <div className="input-group2">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" disabled={isLoading} className="submit-btn">
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginModal;
