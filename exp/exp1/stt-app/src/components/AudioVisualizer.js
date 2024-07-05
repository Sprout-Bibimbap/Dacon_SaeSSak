import React, { useEffect, useRef } from 'react';
import './AudioVisualizer.css';

const AudioVisualizer = ({ isRecording, audioData }) => {
  const barsRef = useRef([]);

  useEffect(() => {
    if (audioData && audioData.length > 0) {
      for (let i = 0; i < barsRef.current.length; i++) {
        const frequencyIndex = Math.floor(Math.random() * audioData.length);
        const baseHeight = 100; // 기본 타원의 최소 길이
        const maxHeight = 300; // 기본 타원의 최대 길이
        const randomFactor = Math.random(); // 0에서 1 사이의 랜덤 값
        const barHeight = baseHeight + (audioData[frequencyIndex] / 255) * (maxHeight - baseHeight) * randomFactor;
        barsRef.current[i].style.height = `${barHeight}px`;
        barsRef.current[i].style.transform = `translateY(${-(barHeight - baseHeight) / 2}px)`; // 위아래로 균일하게 늘어나도록 설정
      }
    } else {
      // 음성이 감지되지 않으면 타원을 최소 길이로 설정
      for (let i = 0; i < barsRef.current.length; i++) {
        barsRef.current[i].style.height = '100px';
        barsRef.current[i].style.transform = 'translateY(0)';
      }
    }
  }, [audioData]);

  return (
    <div className="visualizer">
      {[...Array(4)].map((_, index) => (
        <div
          key={index}
          className="bar"
          ref={el => (barsRef.current[index] = el)}
        ></div>
      ))}
    </div>
  );
};

export default AudioVisualizer;
