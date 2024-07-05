import React, { useState, useRef, useEffect } from 'react';
import AudioVisualizer from './components/AudioVisualizer';
import RecordingIndicator from './components/RecordingIndicator';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [modelanswer, setModelAnswer] = useState('');
  const [audioData, setAudioData] = useState(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = handleDataAvailable;

      mediaRecorderRef.current.start();
      setIsRecording(true);
      updateAudioData();
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const updateAudioData = () => {
    if (!isRecording) return;

    const bufferLength = analyserRef.current.fftSize;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteTimeDomainData(dataArray);
    setAudioData(dataArray);

    animationFrameIdRef.current = requestAnimationFrame(updateAudioData);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      cancelAnimationFrame(animationFrameIdRef.current);
      setAudioData(null);
    }
  };

  const handleDataAvailable = async (event) => {
    if (event.data.size > 0) {
        const audioBlob = new Blob([event.data], { type: 'audio/wav' });
        await sendAudioToServer(audioBlob);
    }
  };

  const sendAudioToServer = async (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    try {
      const response = await fetch(`${process.env.REACT_APP_STT_URL}`, { 
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setTranscript(data.transcription);
      setModelAnswer(data.modelanswer);
    } catch (error) {
      console.error('Error sending audio to server:', error);
    }
  };

  return (
    <div className={`App ${isRecording ? 'recording' : ''}`}>
      <div className="visualizer-container">
        {isRecording ? (
          <AudioVisualizer isRecording={isRecording} audioData={audioData} />
        ) : (
          <div className="initial-circle"></div>
        )}
      </div>
      <RecordingIndicator isListening={isRecording} isRecording={isRecording} />
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? 'Stop' : 'Start'}
      </button>
      <div>
        <h2>Question:</h2>
        <p>{transcript}</p>
      </div>
      <div>
        <h2>Answer:</h2>
        <p>{modelanswer}</p>
      </div>
    </div>
  );
}

export default App;