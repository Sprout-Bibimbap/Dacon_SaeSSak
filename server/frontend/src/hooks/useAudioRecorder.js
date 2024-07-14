import { useState, useCallback, useRef } from 'react';

export const useAudioRecorder = ({ mimeType = 'audio/webm;codecs=opus' }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setAudioData(blob);
        chunksRef.current = [];
      };
      await new Promise(resolve => {
        mediaRecorderRef.current.onstart = resolve;
        mediaRecorderRef.current.start();
      });
  
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      throw error; // 에러를 상위로 전파
    }
  }, [mimeType]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      return new Promise((resolve) => {
        mediaRecorderRef.current.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          setAudioData(blob);
          chunksRef.current = [];
          resolve(blob);
        };
      });
    }
    return Promise.resolve(null);
  }, [isRecording, mimeType]);

  return { isRecording, audioData, startRecording, stopRecording };
};