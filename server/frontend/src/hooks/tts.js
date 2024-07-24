import { useState, useCallback, useRef, useEffect } from 'react';

export const useOpenAITTS = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const playAudioBuffer = useCallback((audioBuffer) => {
    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
    }
    sourceNodeRef.current = audioContextRef.current.createBufferSource();
    sourceNodeRef.current.buffer = audioBuffer;
    sourceNodeRef.current.connect(audioContextRef.current.destination);
    sourceNodeRef.current.start();
    setIsPlaying(true);
    sourceNodeRef.current.onended = () => setIsPlaying(false);
  }, []);

  const getTTS = useCallback(async (text) => {
    setIsLoading(true);
    setError(null);

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    
    try {
      const response = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.REACT_APP_OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: 'tts-1',
          voice: 'alloy',
          input: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const streamAudio = new ReadableStream({
        async start(controller) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
          controller.close();
        },
      });

      const audioArrayBuffer = await new Response(streamAudio).arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(audioArrayBuffer);
      playAudioBuffer(audioBuffer);

      setIsLoading(false);
    } catch (err) {
      console.error('TTS Error:', err);
      setError('TTS Error: ' + err.message);
      setIsLoading(false);
    }
  }, [playAudioBuffer]);

  const stopAudio = useCallback(() => {
    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
    }
    setIsPlaying(false);
    setIsLoading(false);
  }, []);

  return { isPlaying, isLoading, error, getTTS, stopAudio };
};