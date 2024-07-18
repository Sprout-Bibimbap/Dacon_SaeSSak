import React from 'react';
import { ReactComponent as BrainSVG } from '../img/brain.svg';

const BrainMap = ({ brainAreas, brainData, textPositionAdjust = { x: 0, y: 0 }, fontSizeAdjust = '12px', userName = "새싹" }) => {
  return (
    <div className="relative border-2 border-gray-300 rounded-lg p-4 pt-8 pb-12 mb-8">
      <h2 className="absolute top-[1px] -translate-y-[calc(50%+25px)] bg-white px-2 text-2xl font-bold">
        관심 주제
      </h2>
      <div className="relative w-full max-w-md mx-auto">
        <BrainSVG className="w-full h-auto" />
        <div className="absolute inset-0">
          {brainData.map((text, index) => {
            const area = brainAreas[index];
            if (!area) return null;
            return (
              <div
                key={area.id}
                className="absolute text-center"
                style={{
                  left: `${area.x + textPositionAdjust.x}px`,
                  top: `${area.y + textPositionAdjust.y}px`,
                  fontSize: fontSizeAdjust,
                  transform: 'translate(-50%, -50%)'
                }}
              >
                {text}
              </div>
            );
          })}
        </div>
      </div>
      <h3 className="mt-4 text-lg font-semibold text-center">
        {userName}이의 두뇌 탐험
      </h3>
    </div>
  );
};

export default BrainMap;
