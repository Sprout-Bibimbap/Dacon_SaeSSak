import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import axios from 'axios';
import { UserProvider, useUser } from './UserContext';
import Home from './components/Home';
import Login from './components/Login';
import Signup from './components/SignUp';
import ChatBot from './components/ChatBot';
import Report from './components/Report';
import MainPage from './components/MainPage';

const LOGIN_URL = process.env.REACT_APP_LOGIN_URL;
const SIGNUP_URL = process.env.REACT_APP_SIGNUP_URL;
const USERINFO_URL = process.env.REACT_APP_USERINFO_URL;

function App() {
  return (
    <UserProvider>
      <AppContent />
    </UserProvider>
  );
}

function AppContent() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [token, setToken] = useState(null);
  const { setUser } = useUser();

  useEffect(() => {
    const storedToken = localStorage.getItem('token');
    if (storedToken) {
      setToken(storedToken);
      setIsLoggedIn(true);
    }
  }, []);

  const fetchUserInfo = async (token) => {
    try {
      const response = await axios.get(USERINFO_URL, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      const userData = response.data;
      
      // 필요한 필드가 있는지 확인
      if (!userData.username) {
        throw new Error('Invalid user data received');
      }
      
      setUser(userData);
    } catch (error) {
      console.error('Error fetching user info:', error);
      if (error.response && error.response.status === 401) {
        // 인증 에러인 경우에만 로그아웃
        handleLogout();
      } else {
        // 다른 종류의 에러는 사용자에게 알림
        alert('Failed to fetch user information. Please try again.');
      }
    }
  };
  const handleLogin = async (username, password) => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      const response = await axios.post(LOGIN_URL, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const { access_token } = response.data;
      setToken(access_token);
      setIsLoggedIn(true);
      localStorage.setItem('token', access_token);

      await fetchUserInfo(access_token);
    } catch (error) {
      console.error('Login error:', error.response?.data?.detail || error.message);
      alert(error.response?.data?.detail || 'An error occurred during login');
    }
  };

  const handleSignup = async (userData) => {
    try {
      await axios.post(SIGNUP_URL, userData);
      alert('Signup successful. Please login.');
    } catch (error) {
      console.error('Signup error:', error.response?.data?.detail || error.message);
      alert(error.response?.data?.detail || 'An error occurred during signup');
    }
  };



  const handleLogout = () => {
    setIsLoggedIn(false);
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
  };

  // axios 인터셉터 설정
  axios.interceptors.request.use(
    (config) => {
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  return (
    <Router>
      <div className="min-h-screen flex justify-center bg-gray-100">
        <div className="w-full max-w-3xl bg-white shadow-lg">
          <Routes>
            <Route path="/" element={
              isLoggedIn ? <Navigate to="/main" replace /> : <Home />
            } />
            <Route path="/login" element={
              isLoggedIn ? <Navigate to="/main" replace /> : <Login onLogin={handleLogin} />
            } />
            <Route path="/signup" element={
              isLoggedIn ? <Navigate to="/main" replace /> : <Signup onSignup={handleSignup} />
            } />
            <Route 
              path="/main" 
              element={isLoggedIn ? <MainPage onLogout={handleLogout} /> : <Navigate to="/" replace />} 
            />
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