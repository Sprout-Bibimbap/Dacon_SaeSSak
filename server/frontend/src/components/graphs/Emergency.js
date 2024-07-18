import React from 'react';

const SirenStatus = ({ title, siren, aiText }) => {
  const sirenColors = {
    '위험': 'red',
    '주의': 'yellow',
    '안전': 'green'
  };

  return (
    <div className="component-container">
      <h2 className="component-title">{title}</h2>
      
      <div className="flex justify-between">
        <div className="flex-grow flex justify-center items-center">
          <img 
            src={`../img/${sirenColors[siren]}_siren.svg`} 
            alt={`${siren} 상태`} 
            className="w-24 h-24"
          />
        </div>
        
        <div className="flex flex-col items-start space-y-2">
          {Object.entries(sirenColors).map(([key, value]) => (
            <div key={key} className="flex items-center">
              <div className={`w-4 h-4 rounded-full bg-${value}-500 mr-2`}></div>
              <span>{key}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="text-center p-2 bg-gray-100 rounded mt-4">
        <p>{aiText || '데이터를 불러오는 중입니다...'}</p>
      </div>
    </div>
  );
};

export default SirenStatus;