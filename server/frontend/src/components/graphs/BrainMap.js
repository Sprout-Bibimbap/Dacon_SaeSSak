import React from 'react';
import { ReactComponent as BrainSVG } from '../img/brain2.svg';
import './BrainMap.css';

const BrainMap = ({ 
  brainAreas, 
  userBrainData,
  otherUsersBrainData,
  textPositionAdjust = { x: 0, y: 0 }, 
  fontSizeAdjusts = Array(7).fill('12px'), 
  userName = "새싹"
}) => {
  const renderBrain = (brainData, subtitle) => (
    <div className="brain-map-content">
      <BrainSVG className="brain-svg" />
      <div className="brain-areas">
        {brainData.map((text, index) => {
          const area = brainAreas[index];
          if (!area) return null;
          return (
            <div
              key={area.id}
              className="brain-area-text"
              style={{
                left: `${area.x + textPositionAdjust.x}px`,
                top: `${area.y + textPositionAdjust.y}px`,
                fontSize: fontSizeAdjusts[index]
              }}
            >
              {text}
            </div>
          );
        })}
      </div>
      <h3 className="brain-map-subtitle">
        {subtitle}
      </h3>
    </div>
  );

  return (
    <div className="brain-map-container">
      <div className="brain-maps-wrapper">
        {renderBrain(userBrainData, `${userName}이의 두뇌 탐험`)}
        {renderBrain(otherUsersBrainData, '비슷한 사용자들의 두뇌 탐험')}
      </div>
    </div>
  );
};

export default BrainMap;