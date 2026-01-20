import React, { useState } from 'react';
import type { FC } from 'react';

interface LoginViewProps {
  onLoginSuccess: () => void;
  onBack: () => void;
}

const LoginView: FC<LoginViewProps> = ({ onLoginSuccess, onBack }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (username === 'admin' && password === 'rag2026') {
      // you can also set localStorage token here
      onLoginSuccess();
    } else {
      setError('Invalid username or password');
    }
  };

  return (
    <div className="login-container">
      <button className="back-btn" onClick={onBack}>← Back</button>
      <div className="login-card">
        <h1>Login to Enterprise Rag Platform</h1>
        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-group">
            <label>Username</label>
            <input
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <div className="error-message">{error}</div>}
          <button type="submit">Login</button>
        </form>
        <div className="demo-info">Demo: admin / rag2026</div>
      </div>
    </div>
  );
};

export default LoginView;
