import React from 'react';
import './AttachmentRelationship.css'; 

const AttachmentRelationship = ({ humanText, aiText }) => {
  return (
    <div className="attachment-relationship-container">
      {/* 사람 말풍선 */}
      <div className="attachment-bubble human-bubble">
        <svg className="attachment-icon attachment-relationship-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="#4A5568" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="#4A5568" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div className="attachment-text">
          <p className="attachment-relationship-text">{humanText}</p>
        </div>
      </div>
      
      {/* AI 말풍선 */}
      <div className="attachment-bubble ai-bubble">
        <div className="attachment-text">
          <p className="attachment-relationship-text">{aiText}</p>
        </div>
        <svg className="attachment-icon attachment-relationship-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="2" y="2" width="20" height="20" rx="2" stroke="#4A5568" strokeWidth="2"/>
          <circle cx="8" cy="8" r="2" fill="#4A5568"/>
          <circle cx="16" cy="8" r="2" fill="#4A5568"/>
          <path d="M7 16h10" stroke="#4A5568" strokeWidth="2" strokeLinecap="round"/>
          <path d="M12 12v4" stroke="#4A5568" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </div>
    </div>
  );
};

export default AttachmentRelationship;