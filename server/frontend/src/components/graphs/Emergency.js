import React from 'react';
import yellowSiren from '../img/yellow.svg';
import redSiren from '../img/red.svg';
import greenSiren from '../img/green.svg';
import './Emergency.css';

const SirenStatus = ({ siren, aiText }) => {
  const sirenImages = {
    '위험': redSiren,
    '주의': yellowSiren,
    '안전': greenSiren
  };

  const sirenColorClasses = {
    '위험': 'siren-status__indicator--red',
    '주의': 'siren-status__indicator--yellow',
    '안전': 'siren-status__indicator--green'
  };

  return (
    <div className="siren-status">
      <div className="siren-status__content">
        <div className="siren-status__image-container">
          <img 
            src={sirenImages[siren]} 
            alt={`${siren} 상태`} 
            className="siren-status__image"
          />
        </div>
        
        <div className="siren-status__legend">
          {Object.entries(sirenColorClasses).map(([key, colorClass]) => (
            <div key={key} className="siren-status__legend-item">
              <div className={`siren-status__indicator ${colorClass}`}></div>
              <span className="siren-status__legend-text">{key}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="siren-status__ai-text">
        <p>{aiText || '데이터를 불러오는 중입니다...'}</p>
      </div>
    </div>
  );
};

export default SirenStatus;