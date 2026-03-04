import React, { useState } from 'react';

type ContactModalProps = {
  onClose: () => void;
  onSuccess: () => void;
};

const ContactModal: React.FC<ContactModalProps> = ({ onClose, onSuccess }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [query, setQuery] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    const data = { name, email, query };

    try {
      const uploadRes = await fetch('https://forgerag.com/api/files/contactUS', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
        if (uploadRes.ok) {
        setSuccessMsg('Got It! We’ll Get Back to You Shortly! 😊');
        }
      setTimeout(() => {
        onSuccess();
      }, 2200); 
    } catch (err) {
      console.error(err);
      setSuccessMsg('Something went wrong. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content contact-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <button className="modal-close1" onClick={onClose}>
          ✕
        </button>
        <h2 className="modal-title">Contact Us</h2>

        <form onSubmit={handleSubmit} className="contact-form">
          <div className="input-group1">
            <label>Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={50}
              required className="contactTextBox"
            />
          </div>

          <div className="input-group1">
            <label>Email ID</label>
            <input
              type="email"
              value={email}
              maxLength={50}
              onChange={(e) => setEmail(e.target.value)}
              required className="contactTextBox"
            />
          </div>

          <div className="input-group1">
            <label>Query</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={4} 
              maxLength={500}
              required className="textArea"
            />
          </div>

          <button
            type="submit"
            className="submit-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Sending...' : 'Submit'}
          </button>

          {successMsg && <p className="success-text">{successMsg}</p>}
        </form>
      </div>
    </div>
  );
};

export default ContactModal;