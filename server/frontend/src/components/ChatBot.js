import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import AudioVisualizer from './AudioVisualizer';
import RecordingIndicator from './RecordingIndicator';
import ErrorMessage from './ErrorMessage';
import { useOpenAITTS } from '../hooks/tts.js';
import { useAudioRecorder } from '../hooks/useAudioRecorder.js';
import { useUser } from '../UserContext';
import '../App.css';
import './ChatBot.css';

const STT_URL = process.env.REACT_APP_STT_URL || 'http://localhost:8000/api/v1/response/stt';

function ChatBot({ onLogout }) {
  const navigate = useNavigate();
  const { user } = useUser();
  const [transcript, setTranscript] = useState('');
  const [modelAnswer, setModelAnswer] = useState('');
  const [error, setError] = useState(null);

  const { isRecording, audioData, startRecording, stopRecording } = useAudioRecorder({ mimeType: 'audio/webm;codecs=opus' });
  const { isPlaying, isLoading, error: ttsError, getTTS, stopAudio } = useOpenAITTS();

  useEffect(() => {
    if (ttsError) {
      setError('TTS Error: ' + ttsError);
    }
  }, [ttsError]);

  useEffect(() => {
    console.log('isRecording changed:', isRecording);
  }, [isRecording]);

  const sendAudioToServer = useCallback(async (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    formData.append('user_id', user.username);
    formData.append('request_id', uuidv4());
    formData.append('timestamp', new Date().toISOString());

    try {
      const response = await fetch(STT_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setTranscript(data.transcription);
      setModelAnswer(data.model_answer);

      await getTTS(data.model_answer);
    } catch (err) {
      setError('Error sending audio to server: ' + err.message);
    }
  }, [user, getTTS, setError, setTranscript, setModelAnswer]);

  const handleStartRecording = useCallback(async () => {
    try {
      stopAudio();
      await startRecording();
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Microphone permission denied. Please allow microphone access and try again.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No microphone found. Please check your audio input devices and try again.');
      } else {
        setError('Failed to start recording: ' + err.message);
      }
    }
  }, [startRecording, stopAudio]);

  const handleStopRecording = useCallback(async () => {
    try {
      const audioBlob = await stopRecording();
      if (audioBlob) {
        await sendAudioToServer(audioBlob);
      }
    } catch (err) {
      setError('Failed to stop recording: ' + err.message);
    }
  }, [stopRecording, sendAudioToServer]);

  const memoizedAudioVisualizer = useMemo(() => (
    <AudioVisualizer isRecording={isRecording} audioData={audioData} />
  ), [isRecording, audioData]);

  const handleBackToMain = () => {
    navigate('/'); 
  };

  return (
    <div className="chatbot-container">
      <nav className="chatbot-nav">
        <button onClick={handleBackToMain} className="back-button">Back</button>
        <Link to="/report" className="nav-link">View Report</Link>
      </nav>
      <div className={`chat-area ${isRecording ? 'recording' : ''}`}>
        <div className="visualizer-container">
          {isRecording ? memoizedAudioVisualizer : <div className="initial-circle"></div>}
        </div>
        <RecordingIndicator isListening={isRecording} isRecording={isRecording} />
        <button
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          disabled={isLoading}
        >
          {isRecording ? 'Stop' : 'Start'}
        </button>
        {error && <ErrorMessage message={error} />}
        <div>
          <h2>Question:</h2>
          <p>{transcript}</p>
        </div>
        <div>
          <h2>Answer:</h2>
          <p>{modelAnswer}</p>
        </div>
        {isLoading && <p>Generating audio...</p>}
        {isPlaying && <p>Playing audio...</p>}
      </div>
    </div>
  );
}

export default ChatBot;