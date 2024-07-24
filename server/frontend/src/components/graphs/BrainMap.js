import React from 'react';
import BrainMapWithText from './BrainMapWithText';
import './BrainMap.css';

const BrainMap = ({ 
  userBrainAreas,
  otherUsersBrainAreas,
  userBrainData,
  otherUsersBrainData,
  fontSizeAdjusts = Array(7).fill('12px'), 
  textRotations = Array(7).fill(0),
  userName = "새싹"
}) => {
  return (
    <div className="brain-map-container">
      <div className="brain-maps-wrapper">
        <div className="brain-map">
          <div className="brain-map-content">
            <BrainMapWithText
              brainAreas={userBrainAreas}
              brainData={userBrainData}
              fontSizeAdjusts={fontSizeAdjusts}
              textRotations={textRotations}
            />
          </div>
          <h3 className="brain-map-subtitle">{`${userName}이의 두뇌 탐험`}</h3>
        </div>
        <div className="brain-map">
          <div className="brain-map-content">
            <BrainMapWithText
              brainAreas={otherUsersBrainAreas}
              brainData={otherUsersBrainData}
              fontSizeAdjusts={fontSizeAdjusts}
              textRotations={textRotations}
            />
          </div>
          <h3 className="brain-map-subtitle">비슷한 사용자들의 두뇌 탐험</h3>
        </div>
      </div>
    </div>
  );
};

export default BrainMap;