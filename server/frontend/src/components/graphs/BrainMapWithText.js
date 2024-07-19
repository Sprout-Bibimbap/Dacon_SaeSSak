import React from 'react';
import { ReactComponent as BrainSVG } from '../img/brain2.svg';

const BrainMapWithText = ({ 
  brainAreas, 
  brainData,
  fontSizeAdjusts = Array(7).fill('12px'), 
  textRotations = Array(7).fill(0),
}) => {
  return (
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
                left: `${area.x}%`,
                top: `${area.y}%`,
                fontSize: fontSizeAdjusts[index],
                transform: `translate(-50%, -50%) rotate(${textRotations[index]}deg)`
              }}
            >
              {text}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BrainMapWithText;