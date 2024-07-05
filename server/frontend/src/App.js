import React, { useState, useCallback, useMemo } from 'react';
import { useAudioRecorder } from './hooks/useAudioRecorder.js';
import AudioVisualizer from './components/AudioVisualizer';
import RecordingIndicator from './components/RecordingIndicator';
import ErrorMessage from './components/ErrorMessage';
import { useOpenAITTS } from './hooks/tts.js';
import './App.css';

const STT_URL = process.env.REACT_APP_STT_URL || 'http://localhost:8000/api/v1/response/stt';

function App() {
  const [transcript, setTranscript] = useState('');
  const [modelAnswer, setModelAnswer] = useState('');
  const [error, setError] = useState(null);

  const { isRecording, audioData, startRecording, stopRecording } = useAudioRecorder({ mimeType: 'audio/webm;codecs=opus' });
  const { audioUrl, getTTS } = useOpenAITTS();

  const sendAudioToServer = useCallback(async (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

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
  }, [getTTS]);

  const handleStartRecording = useCallback(async () => {
    try {
      await startRecording();
    } catch (err) {
      setError('Failed to start recording: ' + err.message);
    }
  }, [startRecording]);

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

  return (
    <div className={`App ${isRecording ? 'recording' : ''}`}>
      <div className="visualizer-container">
        {isRecording ? memoizedAudioVisualizer : <div className="initial-circle"></div>}
      </div>
      <RecordingIndicator isListening={isRecording} isRecording={isRecording} />
      <button
        onClick={isRecording ? handleStopRecording : handleStartRecording}
        aria-label={isRecording ? 'Stop recording' : 'Start recording'}
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
      {audioUrl && (
        <audio controls>
          <source src={audioUrl} type="audio/mpeg" />
          Your browser does not support the audio element.
        </audio>
      )}
    </div>
  );
}

export default App;
