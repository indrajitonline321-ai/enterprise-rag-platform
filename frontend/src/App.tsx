// Updated App.tsx - Login as MODAL popup (keeps your exact landing + CSS)
import React, { useState } from 'react';
import './App.css';
import LoginModal from './components/LoginModal'; // New modal component
import RagView from './RagView'; // Your existing RAG component
import archDiagram from './ArchDigram.png'; 
import ContactModal from './components/ContactModel';
import { FaFacebook, FaInstagram, FaLinkedin } from "react-icons/fa";

type ViewMode = 'landing' | 'rag' | 'architecture';

function App() {
  const [mode, setMode] = useState<ViewMode>('landing');
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showContactModal, setShowContactModal] = useState(false); 

  const handleGoToContact = () => {
    setShowContactModal(true);
  };

  const handleContactSuccess = () => {
    setShowContactModal(false);
  };


  const handleGoToLogin = () => {
    setShowLoginModal(true); // Open popup modal (keeps landing visible)
  };

  const handleGoToArchitecture = () => {
    setMode('architecture');
  };

  const handleWatchDemo = () => {
    window.open('https://your-demo-link.com', '_blank', 'noopener,noreferrer');
  };

  const handleLoginSuccess = () => {
    setShowLoginModal(false); // Close modal
    setMode('rag'); // Go to RAG dashboard
  };

  const handleLogout = () => {
    setMode('landing');
  };

  // === MAIN LANDING WITH 3 OPTIONS (YOUR EXACT LAYOUT) ===
  if (mode === 'landing') {
    return (
      <>
        <div className="landing">
          <div className="card-grid">
            <a 
              onClick={(e) => {
                e.preventDefault();
                handleGoToArchitecture(); 
              }}
              className="card block no-underline text-inherit cursor-pointer hover:shadow-lg transition-shadow"
            >
              <h2>Architecture View</h2>
            </a>

            <a 
              onClick={(e) => {
                e.preventDefault();
                handleGoToLogin(); // Now opens MODAL popup
              }}
              className="card block no-underline text-inherit cursor-pointer hover:shadow-lg transition-shadow"
            >
              <h2>Login to Application</h2>
            </a>

            <a 
              onClick={(e) => {
                e.preventDefault();
                handleWatchDemo(); 
              }}
              className="card block bg-white no-underline text-inherit cursor-pointer hover:shadow-lg transition-shadow duration-300"
            >
              <h2>Watch Live Demo</h2>
            </a>
          </div>

          
          <div className="Heading">
            <h1>ENTERPRISE - RAG - PLATFORM</h1>
             <div className="ContactUs flex gap-4">
             <a
              onClick={(e) => {
                e.preventDefault();
                handleGoToContact();
              }}
              className="no-underline text-inherit cursor-pointer hover:shadow-lg transition-shadow"
            >
              <span className="contact-us-btn">Contact US</span>
            </a>
            <span> | Social </span>
            <a href="https://www.facebook.com/profile.php?id=61586574128434" className="iconF" target="_blank" 
  rel="noopener noreferrer">
                <FaFacebook color="#1877F2" size={20} />
            </a>
             <a href="#" className="iconI" target="_blank" 
  rel="noopener noreferrer"
            >
                <FaInstagram style={{ color: "#E4405F", fontSize: "20px" }} />
            </a>
             <a href="https://www.linkedin.com/in/indrajit-singh-243a7b85/" className="iconL" target="_blank" 
  rel="noopener noreferrer"
            >
                <FaLinkedin style={{ color: "#0A66C2", fontSize: "20px" }} />
            </a>
            </div>
            </div>
          </div>
        {/* LOGIN POPUP MODAL - overlays landing page */}
        {showLoginModal && (
          <LoginModal 
            onClose={() => setShowLoginModal(false)}
            onSuccess={handleLoginSuccess}
          />
        )}

         {showContactModal && (
          <ContactModal
            onClose={() => setShowContactModal(false)}
            onSuccess={handleContactSuccess}
          />
        )}
        
      </>
      
    );

    
  }

  // === ARCHITECTURE DOCUMENT VIEW (unchanged) ===
  if (mode === 'architecture') {
    return (
      <div className="architecture-page">
        <button className="back-btn" onClick={() => setMode('landing')}>
          ← Back
        </button>
        <h1 className="h1Arch">Enterprise - Rag - Platform Architecture</h1>
        <img
          src={archDiagram}
          className="architecture-image"
        />
        <p className="architecture-text">
        </p>
      </div>
    );
  }

  // === RAG DASHBOARD (unchanged) ===
  if (mode === 'rag') {
    return (
      <RagView onLogout={handleLogout} />
    );
  }

  return null;

  
}

export default App;
