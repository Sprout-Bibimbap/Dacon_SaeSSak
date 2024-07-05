import { useState, useRef, useCallback, useEffect } from 'react';

export const useAudioRecorder = (options = {}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const [error, setError] = useState(null);

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      const mimeType = options.mimeType || 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        throw new Error(`MIME type ${mimeType} is not supported`);
      }

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current.start();
      setIsRecording(true);

      const updateAudioData = () => {
        const bufferLength = analyserRef.current.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteTimeDomainData(dataArray);
        setAudioData(dataArray);
        animationFrameIdRef.current = requestAnimationFrame(updateAudioData);
      };
      updateAudioData();
    } catch (err) {
      setError(err.message);
    }
  }, [options.mimeType]);

  const stopRecording = useCallback(() => {
    return new Promise((resolve) => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        cancelAnimationFrame(animationFrameIdRef.current);
        setAudioData(null);
        
        mediaRecorderRef.current.ondataavailable = (event) => {
          resolve(event.data);
        };
      } else {
        resolve(null);
      }
    });
  }, []);

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

  return {
    isRecording,
    audioData,
    error,
    startRecording,
    stopRecording,
  };
};