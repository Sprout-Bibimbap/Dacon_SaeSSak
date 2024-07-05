import { useState, useCallback, useRef, useEffect } from 'react';

export const useWebSocket = (url) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const websocketRef = useRef(null);

  const connect = useCallback(() => {
    websocketRef.current = new WebSocket(url);

    websocketRef.current.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    websocketRef.current.onmessage = (event) => {
      setLastMessage(event.data);
    };

    websocketRef.current.onerror = (event) => {
      setError('WebSocket error: ' + event.message);
    };

    websocketRef.current.onclose = () => {
      setIsConnected(false);
    };
  }, [url]);

  const disconnect = useCallback(() => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }
  }, []);

  const sendMessage = useCallback((message) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(message);
    } else {
      setError('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
  };
};