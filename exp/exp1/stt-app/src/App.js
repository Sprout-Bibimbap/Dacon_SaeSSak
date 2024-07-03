import React, { useState, useRef, useEffect, useCallback } from 'react';
import AudioVisualizer from './components/AudioVisualizer';
import RecordingIndicator from './components/RecordingIndicator';
import './App.css';

function App() {
  const [isListening, setIsListening] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [audioData, setAudioData] = useState(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const silenceTimeoutRef = useRef(null);
  const animationFrameIdRef = useRef(null); // 추가: 애니메이션 프레임 ID 참조

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current); // 애니메이션 프레임 정리
      }
    };
  }, []);

  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      mediaRecorderRef.current = new MediaRecorder(stream);
      setIsListening(true);
      setIsRecording(true); // 녹음 시작 상태 설정
      detectSound();
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const detectSound = useCallback(() => {
    const bufferLength = analyserRef.current.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const checkAudioLevel = () => {
      analyserRef.current.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += Math.abs(dataArray[i] - 128);
      }
      const average = sum / bufferLength;

      if (average > 10) { // Adjust this threshold as needed
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'recording') {
          startRecording();
        }
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
        }
        silenceTimeoutRef.current = setTimeout(stopRecording, 1500); // Stop after 1.5s of silence

        setAudioData(dataArray); // 음성이 감지되었을 때만 audioData 업데이트
      } else {
        setAudioData(null); // 음성이 감지되지 않으면 null로 설정
      }

      if (isListening) {
        animationFrameIdRef.current = requestAnimationFrame(checkAudioLevel);
      }
    };

    checkAudioLevel();
  }, [isListening]);

  const startRecording = () => {
    audioChunksRef.current = [];
    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };
    mediaRecorderRef.current.onstop = sendAudioToServer;
    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const sendAudioToServer = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    try {
      const response = await fetch(`${process.env.REACT_APP_STT_URL}`, { 
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setTranscript(prev => prev + ' ' + data.transcription);
    } catch (error) {
      console.error('Error sending audio to server:', error);
    }
  };

  const stopListening = () => {
    setIsListening(false);
    setIsRecording(false);
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
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
      <RecordingIndicator isListening={isListening} isRecording={isRecording} />
      <button onClick={isListening ? stopListening : startListening}>
        {isListening ? 'Stop' : 'Start'}
      </button>
      <div>
        <h2>Result:</h2>
        <p>{transcript}</p>
      </div>
    </div>
  );
}

export default App;
