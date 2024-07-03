import React from 'react';

export default function RecordingIndicator({ isListening, isRecording }) {
  return (
    <div className={`recording-indicator ${isRecording ? 'active' : ''}`}>
      {isListening ? 'Recording...' : 'Not Recording'}
    </div>
  );
}
