import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './components/Login';
import ChatBot from './components/ChatBot';
import Report from './components/Report';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(true);

  const handleLogin = (username, password) => {
    // 여기에 실제 로그인 로직을 구현합니다.
    console.log('Login attempt:', username, password);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
  };

  return (
    <Router>
      <div className="min-h-screen flex justify-center bg-gray-100">
        <div className="w-full max-w-3xl bg-white shadow-lg">  {/* 여기서 최대 너비를 제어합니다 */}
          <Routes>
            <Route path="/" element={
              isLoggedIn ? <Navigate to="/chat" replace /> : <Login onLogin={handleLogin} />
            } />
            <Route 
              path="/chat" 
              element={isLoggedIn ? <ChatBot onLogout={handleLogout} /> : <Navigate to="/" replace />} 
            />
            <Route 
              path="/report" 
              element={isLoggedIn ? <Report onLogout={handleLogout} /> : <Navigate to="/" replace />} 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;